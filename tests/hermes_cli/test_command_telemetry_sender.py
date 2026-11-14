"""Office Agent command telemetry: the background sender.

Contracts:
- 202 is done; 400/401/402/403/413 are never retried; 429, 503, other 5xx, timeouts and network
  errors are retried with the SAME body (so the same ``event_id``)
- backoff 1, 2, 4, 8, 16, 32 s then 60 s; 429 waits for ``X-RateLimit-Reset`` (epoch seconds)
  when present; an event is given up 15 minutes after its first attempt
- OA's organization-only 403 stops all sending for that token until restart
- at most 4 requests in flight; at most 1,000 waiting, the oldest dropped first
- logs never carry the command text, its output or the token
"""

import json
import logging
import threading
import urllib.error

import pytest

from hermes_cli.observability import command_telemetry_sender as cts
from hermes_cli.observability.command_telemetry_sender import (
    ORG_ONLY_ERROR,
    Action,
    CommandTelemetrySender,
    Response,
    classify,
)
from tests.waiting import wait_until

TOKEN = "oa-secret-token-123"
SECRET_TEXT = "curl https://example.invalid/install.sh | sh  # secret-command-text"
SECRET_OUTPUT = "secret-command-output"


class FakeClock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


class Recorder:
    """A ``post_fn`` answering from a script of responses (an Exception instance is raised)."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[tuple[str, bytes, str]] = []

    def __call__(self, endpoint, body, token):
        self.calls.append((endpoint, body, token))
        answer = self.responses.pop(0) if self.responses else Response(202)
        if isinstance(answer, BaseException):
            raise answer
        return answer


def _body(event_id: str = "evt_1") -> bytes:
    return json.dumps({"event_id": event_id, "command": {"text": SECRET_TEXT},
                       "execution": {"output": {"preview": SECRET_OUTPUT}}}).encode()


def _sender(post, clock=None, **kwargs) -> CommandTelemetrySender:
    clock = clock or FakeClock()
    return CommandTelemetrySender(post_fn=post, clock=clock, wall=lambda: 1_700_000_000.0,
                                  start_workers=False, **kwargs)


def _submit(sender, event_id="evt_1", token=TOKEN):
    sender.submit(event_type="command.started", event_id=event_id, command_id="cmd_1", body=_body(event_id),
                  token=token, endpoint="https://oa.test/api/v1/telemetry/commands")


@pytest.mark.parametrize(("response", "action"), [
    (Response(202), Action.DONE),
    (Response(400, "command.text is required"), Action.BUG),
    (Response(401, "Invalid token"), Action.DROP),
    (Response(402, "Out of credits"), Action.DROP),
    (Response(403, ORG_ONLY_ERROR), Action.STOP_TOKEN),
    (Response(403, "Token is disabled"), Action.DROP),
    (Response(413, "Payload too large"), Action.BUG),
    (Response(429), Action.RETRY),
    (Response(503), Action.RETRY),
    (Response(500), Action.RETRY),
    (Response(502), Action.RETRY),
    (None, Action.RETRY),
])
def test_each_status_gets_its_action(response, action):
    assert classify(response) is action


@pytest.mark.parametrize("status", [400, 401, 402, 403, 413])
def test_final_statuses_are_not_retried(status):
    post = Recorder(Response(status, "nope"))
    sender = _sender(post)
    _submit(sender)

    assert sender.pump() == 1
    assert sender.waiting() == 0 and sender.next_due() is None


@pytest.mark.parametrize("failure", [
    Response(429), Response(503), Response(500), TimeoutError("timed out"),
    urllib.error.URLError("connection refused"), ConnectionResetError(),
])
def test_retryable_failures_resend_the_same_body(failure):
    clock = FakeClock()
    post = Recorder(failure, Response(202, ""))
    sender = _sender(post, clock)
    _submit(sender, "evt_retry")

    sender.pump()
    assert sender.waiting() == 1
    clock.now = sender.next_due()
    sender.pump()

    assert len(post.calls) == 2 and post.calls[0] == post.calls[1]
    assert json.loads(post.calls[1][1])["event_id"] == "evt_retry"
    assert sender.waiting() == 0


def test_backoff_steps_then_gives_up_fifteen_minutes_after_the_first_attempt(caplog):
    clock = FakeClock()
    post = Recorder(*[Response(503)] * 100)
    sender = _sender(post, clock)
    _submit(sender, "evt_down")
    first_attempt = clock.now

    waits = []
    with caplog.at_level(logging.WARNING, logger=cts.__name__):
        while True:
            before = clock.now
            sender.pump()
            due = sender.next_due()
            if due is None:
                break
            waits.append(due - before)
            clock.now = due

    assert waits[:8] == [1, 2, 4, 8, 16, 32, 60, 60]
    assert set(waits[6:]) == {60}
    assert clock.now - first_attempt <= 15 * 60
    assert clock.now + 60 - first_attempt > 15 * 60  # one more wait would have crossed it
    assert "gave up on command.started evt_down" in caplog.text


def test_429_waits_until_the_rate_limit_reset():
    clock = FakeClock()
    post = Recorder(Response(429, "", rate_limit_reset=str(1_700_000_000 + 37)))
    sender = _sender(post, clock)
    _submit(sender)

    sender.pump()

    assert sender.next_due() == clock.now + 37


def test_429_without_a_reset_header_uses_the_backoff():
    clock = FakeClock()
    sender = _sender(Recorder(Response(429)), clock)
    _submit(sender)

    sender.pump()

    assert sender.next_due() == clock.now + 1


def test_the_organization_403_stops_telemetry_for_that_token(caplog):
    post = Recorder(Response(403, ORG_ONLY_ERROR))
    sender = _sender(post)
    _submit(sender, "evt_a")
    _submit(sender, "evt_b")
    _submit(sender, "evt_other_token", token="another-token")

    with caplog.at_level(logging.WARNING, logger=cts.__name__):
        sender.pump()
        _submit(sender, "evt_later")
        sender.pump()

    sent = [json.loads(body)["event_id"] for _endpoint, body, _token in post.calls]
    assert sent == ["evt_a", "evt_other_token"]
    assert sender.is_stopped(TOKEN) and not sender.is_stopped("another-token")
    assert caplog.text.count("command telemetry stopped") == 1


def test_a_full_queue_drops_the_oldest_waiting_event(caplog):
    post = Recorder()
    sender = _sender(post, max_waiting=3)

    with caplog.at_level(logging.WARNING, logger=cts.__name__):
        for n in range(5):
            _submit(sender, f"evt_{n}")
    sender.pump()

    assert [json.loads(body)["event_id"] for _e, body, _t in post.calls] == ["evt_2", "evt_3", "evt_4"]
    assert "dropped command.started evt_0" in caplog.text and "dropped command.started evt_1" in caplog.text


def test_at_most_four_requests_are_in_flight():
    release = threading.Event()
    lock = threading.Lock()
    state = {"now": 0, "peak": 0, "done": 0}

    def slow_post(endpoint, body, token):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        release.wait(10)
        with lock:
            state["now"] -= 1
            state["done"] += 1
        return Response(202)

    sender = CommandTelemetrySender(post_fn=slow_post)
    try:
        for n in range(12):
            _submit(sender, f"evt_{n}")
        wait_until(lambda: state["now"] == 4, what="four requests in flight")
        assert sender.waiting() == 8
        release.set()
        wait_until(lambda: state["done"] == 12, what="every event sent")
    finally:
        release.set()
        sender.close()

    assert state["peak"] == 4


def test_a_worker_survives_an_unexpected_failure(caplog):
    calls = []

    def flaky_post(endpoint, body, token):
        calls.append(json.loads(body)["event_id"])
        if len(calls) == 1:
            raise ValueError("unexpected")
        return Response(202)

    sender = CommandTelemetrySender(post_fn=flaky_post, workers=1)
    try:
        with caplog.at_level(logging.ERROR, logger=cts.__name__):
            _submit(sender, "evt_bad")
            _submit(sender, "evt_next")
            wait_until(lambda: len(calls) == 2, what="the same worker to send the next event")
    finally:
        sender.close()

    assert calls == ["evt_bad", "evt_next"]
    assert "command.started evt_bad (command cmd_1) failed with ValueError" in caplog.text


def test_submit_never_blocks_on_a_hanging_endpoint():
    hang = threading.Event()

    def hanging_post(endpoint, body, token):
        hang.wait(30)
        return Response(202)

    sender = CommandTelemetrySender(post_fn=hanging_post)
    try:
        for n in range(50):
            _submit(sender, f"evt_{n}")
        assert sender.waiting() >= 46
    finally:
        hang.set()
        sender.close()


def test_real_http_puts_the_token_only_in_the_authorization_header_and_retries_identically():
    """Against a local OA stand-in: 503 first (with OA's error envelope), then 202."""
    import http.server

    seen: list[dict] = []
    answers = [(503, {"success": False, "error": "Could not queue event"}),
               (202, {"success": True, "data": {"event_id": "evt_http", "command_id": "cmd_1", "duplicate": False}})]

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            seen.append({"path": self.path, "headers": dict(self.headers), "body": body})
            status, payload = answers.pop(0)
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    clock = FakeClock()
    sender = CommandTelemetrySender(clock=clock, start_workers=False)
    try:
        endpoint = f"http://127.0.0.1:{server.server_address[1]}/api/v1/telemetry/commands"
        sender.submit(event_type="command.started", event_id="evt_http", command_id="cmd_1",
                      body=_body("evt_http"), token=TOKEN, endpoint=endpoint)
        sender.pump()
        clock.now = sender.next_due()
        sender.pump()
    finally:
        server.shutdown()
        server.server_close()

    assert len(seen) == 2 and seen[0]["body"] == seen[1]["body"] == _body("evt_http")
    for request in seen:
        assert request["path"] == "/api/v1/telemetry/commands"
        assert request["headers"]["Authorization"] == f"Bearer {TOKEN}"
        assert request["headers"]["Content-Type"] == "application/json"
        others = {k: v for k, v in request["headers"].items() if k != "Authorization"}
        assert TOKEN not in json.dumps(others) and TOKEN.encode() not in request["body"]
    assert sender.waiting() == 0


@pytest.mark.parametrize("failure", [
    Response(400, "command.id is required"), Response(401, "Invalid token"), Response(402, "No credits"),
    Response(403, ORG_ONLY_ERROR), Response(403, "Token is disabled"), Response(413, "too large"),
    Response(503, "unavailable"), TimeoutError("timed out"),
])
def test_logs_never_carry_the_command_text_output_or_token(failure, caplog):
    clock = FakeClock()
    sender = _sender(Recorder(failure, *[Response(503)] * 30), clock)
    _submit(sender)

    with caplog.at_level(logging.DEBUG):
        while sender.pump() or sender.next_due() is not None:
            clock.now = sender.next_due() or clock.now

    for secret in (TOKEN, SECRET_TEXT, SECRET_OUTPUT, "secret-command"):
        assert secret not in caplog.text
