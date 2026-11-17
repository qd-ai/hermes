"""The Office Agent token is mirrored from the default profile's ``.env`` into every agent's ``.env``.

Contracts:
- every launch (``hermes gateway ensure``) leaves each servable profile with exactly default's
  ``OA_TOKEN``: overwritten in place, appended, or created; every other line of the file survives
- with no token in default, each profile's copy is removed
- ghost dirs and crashed-create shells (an empty ``.env`` only) are never seeded
- a profile that cannot be written is skipped; ensure still answers
"""

import argparse
import json

import pytest

from agent.secret_scope import load_env_file
from hermes_cli import gateway_runtime, version_info
from hermes_cli.gateway_runtime import GatewayDiscovery
from hermes_cli.gateway_runtime_cli import cmd_gateway_ensure
from hermes_cli.office_agent_token_mirror import mirror_office_agent_token

CENTRAL = "oad_central"


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A custom-root install (``<root>/.env`` + ``<root>/profiles/*``), the shape the install-wide
    secret reader resolves for every profile."""
    home = tmp_path / "root"
    (home / "profiles").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _profile(root, name, env=None):
    home = root / "profiles" / name
    home.mkdir()
    (home / "config.yaml").write_text("model:\n  provider: office-agent\n")
    if env is not None:
        (home / ".env").write_text(env)
    return home


def _lines(path):
    return path.read_text().splitlines()


def test_every_profile_gets_defaults_token_and_keeps_its_other_lines(root):
    (root / ".env").write_text(f"OA_TOKEN={CENTRAL}\nOPENAI_API_KEY=sk-root-only\n")
    stale = _profile(root, "geo-agent", "FOO=1\nOA_TOKEN=oad_stale\nexport OA_TOKEN=oad_older\nBAR=2")
    unset = _profile(root, "deal-agent", "FOO=1")
    bare = _profile(root, "fresh-agent")
    crashed = root / "profiles" / "crashed"
    crashed.mkdir()
    (crashed / ".env").write_text("")
    ghost = root / "profiles" / "ghost"
    (ghost / "logs").mkdir(parents=True)

    assert sorted(mirror_office_agent_token()) == ["deal-agent", "fresh-agent", "geo-agent"]

    assert _lines(stale / ".env") == ["FOO=1", f"OA_TOKEN={CENTRAL}", "BAR=2"]
    assert _lines(unset / ".env") == ["FOO=1", f"OA_TOKEN={CENTRAL}"]
    assert _lines(bare / ".env") == [f"OA_TOKEN={CENTRAL}"]
    assert (crashed / ".env").read_text() == ""
    assert not (ghost / ".env").exists()
    assert load_env_file(root / ".env") == {"OA_TOKEN": CENTRAL, "OPENAI_API_KEY": "sk-root-only"}
    # Already mirrored: the next launch rewrites nothing.
    assert mirror_office_agent_token() == []


@pytest.mark.parametrize("default_env", ["OPENAI_API_KEY=sk-root-only\n", "OA_TOKEN=\n", None])
def test_without_a_default_token_each_profile_copy_is_removed(root, default_env):
    if default_env is not None:
        (root / ".env").write_text(default_env)
    agent = _profile(root, "geo-agent", "FOO=1\nOA_TOKEN=oad_agent_only\nBAR=2\n")
    clean = _profile(root, "deal-agent", "FOO=1\n")

    assert mirror_office_agent_token() == ["geo-agent"]

    assert _lines(agent / ".env") == ["FOO=1", "BAR=2"]
    assert _lines(clean / ".env") == ["FOO=1"]


def test_gateway_ensure_mirrors_before_it_answers(root, monkeypatch, capsys):
    (root / ".env").write_text(f"OA_TOKEN={CENTRAL}\n")
    agent = _profile(root, "geo-agent", "OA_TOKEN=oad_stale\n")
    seen_at_ensure = []

    def fake_ensure(profile_home, *, timeout):
        seen_at_ensure.append(load_env_file(agent / ".env").get("OA_TOKEN"))
        return GatewayDiscovery(state="absent")

    monkeypatch.setattr(gateway_runtime, "ensure_gateway_runtime", fake_ensure)
    monkeypatch.setattr(version_info, "get_code_identity", lambda: {"sha": None})

    with pytest.raises(SystemExit):
        cmd_gateway_ensure(argparse.Namespace(timeout=1))

    assert seen_at_ensure == [CENTRAL]
    assert json.loads(capsys.readouterr().out)["state"] == "absent"


def test_a_profile_that_cannot_be_written_is_skipped(root):
    (root / ".env").write_text(f"OA_TOKEN={CENTRAL}\n")
    broken = _profile(root, "broken-agent")
    (broken / ".env").mkdir()  # unreadable as a file on every OS
    agent = _profile(root, "geo-agent", "OA_TOKEN=oad_stale\n")

    assert mirror_office_agent_token() == ["geo-agent"]
    assert _lines(agent / ".env") == [f"OA_TOKEN={CENTRAL}"]
