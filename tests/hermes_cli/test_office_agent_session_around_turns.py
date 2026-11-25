"""Model requests a session makes just before or after its turn name that session to Office Agent.

The CLI describes attached images before the turn starts, and the ``/goal`` and ``/loop --until`` judges run after
it ends, so no turn is there to bind the session for them.
"""
import json
import time

import pytest

from agent.office_agent_headers import office_agent_headers
from hermes_cli import goals
from hermes_cli.cli_stream_mixin import CLIStreamMixin
from hermes_cli.goals import GoalManager
from hermes_cli.loops import LoopManager

DONE = ("done", "finished", False, None, False)


@pytest.fixture(autouse=True)
def fresh_goal_store():
    goals._DB_CACHE.clear()
    yield
    goals._DB_CACHE.clear()


@pytest.fixture
def named():
    """The session ids Office Agent would be given, one per request."""
    return []


def _record(named):
    named.append(office_agent_headers("https://model.officeagent.us/deepseek/v4-flash/v1").get("X-OA-Session-Id"))


def test_the_goal_judge_names_the_session_the_goal_belongs_to(monkeypatch, named):
    monkeypatch.setattr(goals, "judge_goal", lambda *_a, **_k: _record(named) or DONE)
    manager = GoalManager(session_id="sess-goal")
    manager.set("ship it")

    manager.evaluate_after_turn("shipped")

    assert named == ["sess-goal"]


def test_the_loop_judge_names_the_session_the_loop_belongs_to(monkeypatch, named):
    monkeypatch.setattr(goals, "judge_goal", lambda *_a, **_k: _record(named) or DONE)
    manager = LoopManager(session_id="sess-loop")
    state = manager.set("poll", interval_seconds=300, until="ci is green")
    state.next_due_at = time.time() - 1
    manager.fire_tick()

    manager.complete_tick("ci is green")

    assert named == ["sess-loop"]


def test_an_image_attached_in_the_cli_is_described_under_its_session(monkeypatch, tmp_path, named):
    async def analyze(**_kw):
        _record(named)
        return json.dumps({"success": True, "analysis": "a pixel"})

    class Cli(CLIStreamMixin):
        session_id = "sess-cli"

    monkeypatch.setattr("tools.vision_tools.vision_analyze_tool", analyze)
    image = tmp_path / "a.png"
    image.write_bytes(b"png")

    Cli()._preprocess_images_with_vision("look", [image], announce=False)

    assert named == ["sess-cli"]
