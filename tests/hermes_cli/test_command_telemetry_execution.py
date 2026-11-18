"""Office Agent command telemetry on the real execution path.

Commands run through ``model_tools.handle_function_call`` (the real ``pre_tool_call`` /
``post_tool_call`` dispatch and the real terminal tool) against a temp ``HERMES_HOME`` whose root
``.env`` holds ``OA_TOKEN``. Contracts:
- each command sends ``command.started`` then ``command.finished`` with one identity triple and two
  different ``event_id``s, and nothing from Hermes's side beyond the contract's fields
- ``source.agent`` is the name of the profile the command ran in (``default`` for the root), in both
  events, even when ``finished`` is sent after that profile's scope has ended
- ``finished`` is sent however the command ends: timeout, kill (background), failure to start, block
- a hanging or unreachable OA endpoint never slows a command or changes its result
- ``telemetry.commands.enabled: false`` sends nothing
"""

import json
import logging
import os
import socket
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import pytest

import model_tools
from hermes_cli.config_defaults import DEFAULT_CONFIG
from hermes_cli.observability import command_telemetry as ct
from hermes_cli.observability.command_telemetry_background import BackgroundWatch
from hermes_cli.observability.command_telemetry_sender import CommandTelemetrySender, Response
from hermes_cli.observability.command_telemetry_tools import command_for, outcome_for
from tests.waiting import wait_until

TOKEN = "oa-exec-test-token"
ENDPOINT = "https://oa.test/api/v1/telemetry/commands"

ALLOWED = {
    "schema_version", "event_id", "event_type", "timestamp", "source", "command", "execution",
    "source.agent", "source.instance_id", "source.session_id", "source.tool_call_id",
    "command.id", "command.tool", "command.text", "command.cwd",
    "execution.status", "execution.exit_code", "execution.duration_ms", "execution.output",
    "execution.output.preview", "execution.output.total_bytes", "execution.output.truncated",
}


def _paths(node, prefix=""):
    for key, value in node.items():
        path = f"{prefix}{key}"
        yield path
        if isinstance(value, dict):
            yield from _paths(value, f"{path}.")


def _configure(*, enabled: bool = True, base_url: str = "https://oa.test/") -> None:
    home = Path(os.environ["HERMES_HOME"])
    (home / "config.yaml").write_text(
        f"telemetry:\n  commands:\n    enabled: {'true' if enabled else 'false'}\n    base_url: {base_url}\n")


@pytest.fixture(autouse=True)
def oa_install(monkeypatch):
    """Root ``.env`` with the token, telemetry on, a fresh pairing table; real sender closed after."""
    (Path(os.environ["HERMES_HOME"]) / ".env").write_text(f"OA_TOKEN={TOKEN}\n")
    _configure()
    monkeypatch.setattr(ct, "_PENDING", OrderedDict())
    monkeypatch.setitem(ct._STATE, "sender", None)
    monkeypatch.setitem(ct._STATE, "watch", None)
    yield
    if ct._STATE["sender"] is not None:
        ct._STATE["sender"].close()


class Captured:
    def __init__(self, monkeypatch):
        self.posts: list[tuple[str, bytes, str]] = []
        self.sender = CommandTelemetrySender(post_fn=self._post, start_workers=False)
        self.watch = BackgroundWatch(ct._report_finished, start_thread=False)
        monkeypatch.setitem(ct._STATE, "sender", self.sender)
        monkeypatch.setitem(ct._STATE, "watch", self.watch)

    def _post(self, endpoint, body, token):
        self.posts.append((endpoint, body, token))
        return Response(202)

    def events(self) -> list[dict]:
        self.sender.pump()
        return [json.loads(body.decode("utf-8")) for _endpoint, body, _token in self.posts]


@pytest.fixture
def captured(monkeypatch) -> Captured:
    return Captured(monkeypatch)


def _run(args: dict, call_id: str = "call_1") -> str:
    return model_tools.handle_function_call("terminal", args, task_id="task_1", session_id="session_1",
                                            tool_call_id=call_id)


def _identity(event: dict) -> tuple:
    return event["source"]["instance_id"], event["source"]["session_id"], event["command"]["id"]


def test_a_command_sends_started_then_finished_under_one_identity(captured):
    result = json.loads(_run({"command": "echo hello"}))
    assert result["output"] == "hello"

    started, finished = captured.events()

    assert [endpoint for endpoint, _b, _t in captured.posts] == [ENDPOINT, ENDPOINT]
    assert {token for _e, _b, token in captured.posts} == {TOKEN}
    assert started["event_type"] == "command.started" and finished["event_type"] == "command.finished"
    assert _identity(started) == _identity(finished)
    assert started["event_id"] != finished["event_id"]
    assert started["source"] == {"agent": "default", "instance_id": started["source"]["instance_id"],
                                 "session_id": "session_1", "tool_call_id": "call_1"}
    assert started["command"]["tool"] == "terminal" and started["command"]["text"] == "echo hello"
    assert finished["execution"]["status"] == "completed" and finished["execution"]["exit_code"] == 0
    assert finished["execution"]["output"] == {"preview": "hello", "total_bytes": 5, "truncated": False}
    assert finished["execution"]["duration_ms"] >= 0


@contextmanager
def _in_profile(name: str):
    """Run as the Desktop backend and the multiplex gateway run an agent's turn: the profile's home is
    bound for this context only, while the process's ``HERMES_HOME`` stays the root."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home = Path(os.environ["HERMES_HOME"]) / "profiles" / name
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("telemetry:\n  commands:\n    base_url: https://oa.test/\n")
    token = set_hermes_home_override(home)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def test_a_command_in_a_named_profile_reports_that_profile_as_the_agent(captured):
    with _in_profile("geo-agent"):
        _run({"command": "echo hi"})

    started, finished = captured.events()

    assert started["source"]["agent"] == finished["source"]["agent"] == "geo-agent"


def test_a_background_command_reports_its_own_profile_after_that_scope_ends(captured):
    from tools.process_registry import process_registry

    with _in_profile("leads-agent"):
        proc_id = json.loads(_run({"command": "sleep 30", "background": True}))["session_id"]

    process_registry.kill_process(proc_id)
    wait_until(lambda: captured.watch.poll_once() == 0, what="the watch to see the kill")

    started, finished = captured.events()
    assert finished["execution"]["status"] == "killed"
    assert started["source"]["agent"] == finished["source"]["agent"] == "leads-agent"


def test_two_commands_never_share_a_command_id(captured):
    _run({"command": "echo one"}, "call_1")
    _run({"command": "echo two"}, "call_2")

    ids = {event["command"]["id"] for event in captured.events()}

    assert len(ids) == 2


def test_unknown_args_and_result_fields_never_reach_the_body(captured):
    _run({"command": "echo hi", "pty": False, "secret_extra": "zz-extra-zz", "heartbeat": 0})

    for event in captured.events():
        assert set(_paths(event)) <= ALLOWED
        assert "zz-extra-zz" not in json.dumps(event)


def test_a_command_with_a_secret_is_sent_masked(captured):
    _run({"command": "echo $X; curl -H 'Authorization: Bearer sk-ant-api03-abcdefghijklmnopqrstuvwxyz012345' x"})

    started = captured.events()[0]

    assert "sk-ant-api03-abcdefghijklmnopqrstuvwxyz012345" not in started["command"]["text"]
    assert started["command"]["text"].startswith("echo $X; curl -H 'Authorization: Bearer ")


def test_a_timed_out_command_still_sends_finished(captured):
    _run({"command": "sleep 5", "timeout": 1})

    finished = captured.events()[-1]

    assert finished["execution"]["status"] == "timed_out"
    assert finished["execution"]["exit_code"] is None


def test_a_command_that_fails_to_start_still_sends_finished(captured, monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise RuntimeError("terminal backend unavailable")

    monkeypatch.setattr("tools.terminal_tool._acquire_env", unavailable)

    _run({"command": "echo never"})
    started, finished = captured.events()

    assert _identity(started) == _identity(finished)
    assert finished["execution"]["status"] == "error"
    assert finished["execution"]["exit_code"] is None


def test_a_killed_background_command_sends_finished_when_it_ends(captured):
    from tools.process_registry import process_registry

    proc_id = json.loads(_run({"command": "sleep 30", "background": True}))["session_id"]
    assert [event["event_type"] for event in captured.events()] == ["command.started"]
    assert captured.watch.watching() == 1

    process_registry.kill_process(proc_id)
    wait_until(lambda: captured.watch.poll_once() == 0, what="the watch to see the kill")

    started, finished = captured.events()
    assert _identity(started) == _identity(finished)
    assert finished["execution"]["status"] == "killed"
    assert finished["execution"]["exit_code"] is None


def test_a_blocked_call_still_sends_finished(captured):
    hook = {"tool_name": "terminal", "args": {"command": "rm -rf build"}, "session_id": "s", "task_id": "t",
            "tool_call_id": "call_9"}
    ct.observe_lifecycle("pre_tool_call", **hook)
    ct.observe_lifecycle("post_tool_call", **hook, result=json.dumps({"error": "Blocked by policy"}),
                         status="blocked", duration_ms=0)

    started, finished = captured.events()

    assert _identity(started) == _identity(finished)
    assert finished["execution"] == {"status": "blocked", "exit_code": None, "duration_ms": finished["execution"]["duration_ms"]}


def test_a_finished_call_with_no_started_event_reports_both(captured):
    ct.observe_lifecycle("post_tool_call", tool_name="terminal", args={"command": "make test"}, session_id="s",
                         task_id="t", tool_call_id="call_late",
                         result=json.dumps({"output": "ok", "exit_code": 2, "error": None}), duration_ms=40)

    started, finished = captured.events()

    assert started["command"]["text"] == "make test" and _identity(started) == _identity(finished)
    assert finished["execution"]["status"] == "failed" and finished["execution"]["exit_code"] == 2


def test_the_off_switch_sends_nothing(captured):
    _configure(enabled=False)

    _run({"command": "echo quiet"})

    assert not ct.handles_hook("post_tool_call")
    assert captured.events() == []


def test_without_a_token_nothing_is_sent(captured):
    (Path(os.environ["HERMES_HOME"]) / ".env").write_text("")

    _run({"command": "echo no-token"})

    assert captured.events() == []


def test_the_shipped_default_is_on_and_points_at_office_agent():
    (Path(os.environ["HERMES_HOME"]) / "config.yaml").write_text("{}\n")

    assert DEFAULT_CONFIG["telemetry"]["commands"] == {"enabled": True, "base_url": "https://data.officeagent.us/"}
    assert ct.settings() == ct.Settings(True, "https://data.officeagent.us/api/v1/telemetry/commands")


@pytest.mark.parametrize(("base_url", "endpoint"), [
    ("https://data.officeagent.us/", "https://data.officeagent.us/api/v1/telemetry/commands"),
    ("https://data.officeagent.us", "https://data.officeagent.us/api/v1/telemetry/commands"),
    ("https://oa.example/api/", "https://oa.example/api/v1/telemetry/commands"),
    ("http://127.0.0.1:8080", "http://127.0.0.1:8080/api/v1/telemetry/commands"),
    ("http://data.officeagent.us/", None),
    ("not a url", None),
])
def test_the_endpoint_is_built_from_the_base_url(base_url, endpoint):
    assert ct.endpoint_for(base_url) == endpoint


@pytest.mark.parametrize(("tool", "args", "text"), [
    ("terminal", {"command": "ls -la", "workdir": "/srv/app"}, "ls -la"),
    ("execute_code", {"code": "import os\nos.remove('x')"}, "import os\nos.remove('x')"),
    ("browser_exec", {"code": "await page.goto('https://x.test')"}, "await page.goto('https://x.test')"),
    ("process_manage", {"action": "submit", "session_id": "proc_1", "data": "rm -rf /tmp/x"}, "rm -rf /tmp/x"),
    ("process_manage", {"action": "poll", "session_id": "proc_1"}, None),
    ("read_file", {"path": "/etc/passwd"}, None),
    ("terminal", {"command": "   "}, None),
])
def test_every_command_running_tool_is_covered(tool, args, text):
    call = command_for(tool, args, "task")

    assert (call.text if call else None) == text
    if tool == "terminal" and call:
        assert call.cwd == "/srv/app"


@pytest.mark.parametrize(("result", "hook_status", "status", "exit_code"), [
    ({"output": "", "exit_code": 0, "error": None}, None, "completed", 0),
    ({"output": "boom", "exit_code": 1, "error": None}, None, "failed", 1),
    ({"output": "x\n[Command timed out after 3s]", "exit_code": 124, "error": None}, None, "timed_out", None),
    ({"output": "[Command interrupted]", "exit_code": 130, "error": None}, None, "cancelled", None),
    ({"output": "", "exit_code": -9, "error": None}, None, "killed", None),
    ({"output": "", "exit_code": -1, "error": "Command execution failed"}, None, "error", None),
    ({"output": "", "exit_code": -1, "error": "denied", "status": "blocked"}, None, "blocked", None),
    ({"status": "success", "output": "1\n"}, None, "completed", None),
    ({"status": "timeout", "output": ""}, None, "timed_out", None),
    ({"success": False, "exit_code": 3, "output": "err"}, None, "failed", 3),
    ({"output": "partial"}, "cancelled", "cancelled", None),
    ({"output": "partial"}, "timeout", "timed_out", None),
])
def test_outcomes_map_onto_the_oa_vocabulary(result, hook_status, status, exit_code):
    outcome = outcome_for(json.dumps(result), hook_status=hook_status)

    assert (outcome.status, outcome.exit_code) == (status, exit_code)


# -- the OA endpoint misbehaving ----------------------------------------------------------------


@contextmanager
def _hanging_endpoint():
    """A local endpoint that accepts connections and never answers."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(16)
    server.settimeout(0.2)
    accepted: list[socket.socket] = []
    stop = threading.Event()

    def accept_forever():
        while not stop.is_set():
            try:
                accepted.append(server.accept()[0])
            except TimeoutError:
                continue
            except OSError:
                return

    thread = threading.Thread(target=accept_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.getsockname()[1]}", accepted
    finally:
        stop.set()
        thread.join(5)
        for connection in accepted:
            connection.close()
        server.close()


def _closed_port_url() -> str:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return f"http://127.0.0.1:{port}"


def _timed_commands(count: int = 3) -> tuple[float, list[str]]:
    start = time.monotonic()
    results = [_run({"command": f"echo run-{n}"}, f"call_{n}") for n in range(count)]
    return time.monotonic() - start, results


def test_a_hanging_endpoint_does_not_slow_commands():
    _configure(enabled=False)
    off_seconds, off_results = _timed_commands()

    with _hanging_endpoint() as (base_url, accepted):
        _configure(base_url=base_url)
        on_seconds, on_results = _timed_commands()
        wait_until(lambda: len(accepted) >= 1, what="telemetry to reach the hanging endpoint")

    assert on_results == off_results
    assert on_seconds < off_seconds + 2.0


def test_an_unreachable_endpoint_leaves_results_unchanged(caplog):
    _configure(enabled=False)
    _seconds, off_results = _timed_commands()

    _configure(base_url=_closed_port_url())
    with caplog.at_level(logging.DEBUG):
        _seconds, on_results = _timed_commands()
        wait_until(lambda: ct.sender().next_due() is not None and ct.sender().waiting() >= 1,
                   what="failed sends queued for retry")

    assert on_results == off_results
    telemetry_logs = "\n".join(record.getMessage() for record in caplog.records
                               if record.name.startswith("hermes_cli.observability.command_telemetry"))
    assert TOKEN not in telemetry_logs and "echo run-" not in telemetry_logs
