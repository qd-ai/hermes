"""Office Agent model requests name the agent, session and installation that sent them.

Contracts: only requests to model.officeagent.us carry the headers; the values are the ones command telemetry
reports for the same work; nothing about them can break or change the rest of a request.
"""

import os
import socket
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import auxiliary_client as aux
from agent import opencode_affinity
from agent.chat_completion_helpers import build_api_kwargs
from agent.office_agent_headers import office_agent_headers
from hermes_cli.install_identity import get_install_id
from hermes_cli.observability import command_telemetry as ct
from run_agent import AIAgent

BASE = "https://model.officeagent.us/deepseek/v4-flash/v1"
OTHER = "https://openrouter.ai/api/v1"
MSGS = [{"role": "user", "content": "hi"}]
AGENT, SESSION, INSTANCE = "X-OA-Agent", "X-OA-Session-Id", "X-OA-Instance-Id"


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    """No network (building an agent probes its host), no pinned profile, no cached id, no ambient turn."""
    def no_network(*_a, **_k):
        raise socket.gaierror("no network in tests")

    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    monkeypatch.delenv("HERMES_PROFILE_NAME", raising=False)
    monkeypatch.setitem(ct._STATE, "instance_id", None)
    token = aux._RUNTIME_MAIN_CONTEXT.set(None)
    yield
    aux._RUNTIME_MAIN_CONTEXT.reset(token)


def _agent(session_id="sess-A"):
    return AIAgent(api_key="test-key", base_url=BASE, model="v4-flash", provider="custom", quiet_mode=True,
                   skip_context_files=True, skip_memory=True, session_id=session_id)


def _oa(kwargs):
    return {k: v for k, v in (kwargs.get("extra_headers") or {}).items() if k in (AGENT, SESSION, INSTANCE)}


def _main(agent):
    return _oa(build_api_kwargs(agent, MSGS))


def _side(session_id=""):
    token = aux.set_runtime_main("custom", "v4-flash", base_url=BASE, session_id=session_id)
    try:
        return _oa(aux._build_call_kwargs("custom", "v4-flash", MSGS, base_url=BASE))
    finally:
        aux.reset_runtime_main(token)


@contextmanager
def _in_profile(name):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home = Path(os.environ["HERMES_HOME"]) / "profiles" / name
    home.mkdir(parents=True)
    token = set_hermes_home_override(home)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def test_main_and_side_requests_name_the_agent_session_and_installation():
    expected = {AGENT: "default", SESSION: "sess-A", INSTANCE: get_install_id()}

    assert _main(_agent()) == expected
    assert _side("sess-A") == expected


@pytest.mark.parametrize("url", ["HTTPS://MODEL.OfficeAgent.US:443/deepseek/v4-pro/v1/", "https://eu.model.officeagent.us/v1"])
def test_every_spelling_of_the_office_agent_host_gets_them(url):
    assert set(office_agent_headers(url, "sess-A")) == {AGENT, SESSION, INSTANCE}


@pytest.mark.parametrize("url", [
    OTHER, "https://model.officeagent.us.example.com/v1", "https://example.com/model.officeagent.us/v1",
    "https://notmodel.officeagent.us/v1", "https://officeagent.us/v1", "http://localhost:3000/v1", "", None,
])
def test_no_other_host_gets_them(url):
    assert office_agent_headers(url, "sess-A") == {}


def test_switching_away_from_office_agent_drops_them():
    agent = _agent()
    assert set(_main(agent)) == {AGENT, SESSION, INSTANCE}

    agent.base_url = OTHER

    assert _main(agent) == {}


def test_a_call_outside_any_session_leaves_out_only_the_session():
    assert set(_side()) == {AGENT, INSTANCE}


def test_each_profile_in_one_process_names_itself():
    with _in_profile("geo-agent"):
        geo = _side("s1")
    with _in_profile("leads-agent"):
        leads = _side("s2")

    assert (geo[AGENT], leads[AGENT]) == ("geo-agent", "leads-agent")
    assert geo[INSTANCE] == leads[INSTANCE]


@pytest.mark.parametrize("session_id", ["sess-A", "s" * 300])
def test_the_values_are_the_ones_command_telemetry_reports(monkeypatch, session_id):
    (Path(os.environ["HERMES_HOME"]) / ".env").write_text("OA_TOKEN=oa-test-token\n")
    sources = []
    monkeypatch.setattr(ct, "_PENDING", OrderedDict())
    monkeypatch.setitem(ct._STATE, "sender", SimpleNamespace(is_stopped=lambda _token: False))
    monkeypatch.setattr(ct, "_submit", lambda _pending, _type, event: sources.append(event["source"]))
    ct._on_pre_tool_call({"tool_name": "terminal", "args": {"command": "true"}, "task_id": "t1",
                          "session_id": session_id, "tool_call_id": "c1"})

    headers = office_agent_headers(BASE, session_id)

    assert headers == {AGENT: sources[0]["agent"], SESSION: sources[0]["session_id"], INSTANCE: sources[0]["instance_id"]}


def test_an_unreadable_install_sends_the_id_telemetry_falls_back_to(monkeypatch):
    monkeypatch.setattr("hermes_cli.install_identity.get_install_id", lambda: None)

    sent = office_agent_headers(BASE, "sess-A")[INSTANCE]

    assert sent.startswith("process-") and sent == ct.instance_id()


@pytest.mark.parametrize("session_id", ["sesión", "a\nb", "x\x7f", " padded"])
def test_an_unsafe_session_id_is_left_out_and_the_rest_still_goes(session_id):
    assert set(office_agent_headers(BASE, session_id)) == {AGENT, INSTANCE}


@pytest.mark.parametrize("name", ["日本", "a" * 257])
def test_an_unsafe_agent_name_is_left_out_and_the_rest_still_goes(monkeypatch, name):
    monkeypatch.setenv("HERMES_PROFILE_NAME", name)

    assert set(office_agent_headers(BASE, "sess-A")) == {SESSION, INSTANCE}


def test_a_failure_building_them_is_logged_and_the_request_goes_without_them(monkeypatch, caplog):
    def boom():
        raise RuntimeError("no identity")

    monkeypatch.setattr(ct, "instance_id", boom)

    assert _main(_agent()) == {}
    assert "Office Agent headers" in caplog.text


def test_nothing_else_about_the_request_changes(monkeypatch):
    agent = _agent()
    agent.request_overrides = {"extra_headers": {"X-Mine": "1"}}
    with_headers = build_api_kwargs(agent, MSGS)
    monkeypatch.setattr(opencode_affinity, "office_agent_headers", lambda *_a: {})
    without = build_api_kwargs(agent, MSGS)

    sent = with_headers.pop("extra_headers")
    assert without.pop("extra_headers") == {"X-Mine": "1"}
    assert with_headers == without
    assert sent["X-Mine"] == "1" and "Authorization" not in sent


LONG_CHAT = [m for i in range(30) for m in ({"role": "user", "content": f"q{i} " + "x" * 4000},
                                            {"role": "assistant", "content": f"a{i} " + "y" * 4000})]


@pytest.fixture
def sent(monkeypatch):
    """The X-OA headers of every side request, served by a fake client on the Office Agent route."""
    requests = []

    def create(**kwargs):
        requests.append(_oa(kwargs))
        message = SimpleNamespace(content="Summary: ok " * 30)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], model="v4-flash", usage=None)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), base_url=BASE)
    monkeypatch.setattr(aux, "_resolve_task_provider_model", lambda *_a, **_k: ("custom", "v4-flash", BASE, "test-key", None))
    monkeypatch.setattr(aux, "_get_cached_client", lambda *_a, **_k: (client, "v4-flash"))
    return requests


def _side_call_with_its_own_runtime():
    """What the compactors do: a side call that passes a runtime holding no session id."""
    aux.call_llm(task="compression", messages=MSGS,
                 main_runtime={"provider": "custom", "model": "v4-flash", "base_url": BASE, "api_key": "test-key"})


def _turn(monkeypatch, bodies):
    """Run real turns whose model loop is replaced by ``bodies[agent.session_id]``."""
    import agent.conversation_loop as loop

    def fake_loop(agent, *_a, **_k):
        bodies[agent.session_id](agent)
        return {"final_response": "ok", "messages": []}

    monkeypatch.setattr(loop, "run_conversation", fake_loop)


def test_compaction_outside_a_turn_names_the_session_it_compacts(sent):
    _agent("sess-A")._compress_context(list(LONG_CHAT), "sys", approx_tokens=900_000, force=True)

    assert sent and {h.get(SESSION) for h in sent} == {"sess-A"}


def test_side_calls_inside_a_turn_name_the_running_session(monkeypatch, sent):
    def body(agent):
        agent._compress_context(list(LONG_CHAT), "sys", approx_tokens=900_000, force=True)
        _side_call_with_its_own_runtime()

    _turn(monkeypatch, {"sess-A": body})
    _agent("sess-A").run_conversation("hello")

    assert len(sent) >= 2 and {h.get(SESSION) for h in sent} == {"sess-A"}


def test_a_subagent_names_itself_and_the_parent_is_itself_again_after(monkeypatch, sent):
    def parent(_agent_):
        _side_call_with_its_own_runtime()
        _agent("sess-child").run_conversation("child work")
        _side_call_with_its_own_runtime()

    _turn(monkeypatch, {"sess-A": parent, "sess-child": lambda _a: _side_call_with_its_own_runtime()})
    _agent("sess-A").run_conversation("hello")

    assert [h.get(SESSION) for h in sent] == ["sess-A", "sess-child", "sess-A"]


def test_a_turn_that_raises_leaves_no_session_behind(monkeypatch, sent):
    def body(_agent_):
        raise RuntimeError("turn failed")

    _turn(monkeypatch, {"sess-A": body})
    with pytest.raises(RuntimeError, match="turn failed"):
        _agent("sess-A").run_conversation("hello")
    _side_call_with_its_own_runtime()

    assert SESSION not in sent[-1]
