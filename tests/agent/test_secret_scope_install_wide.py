"""Install-wide secrets: the Office Agent token lives once, in the root home's ``.env``.

Contracts, whatever profile or process asks:
- ``OA_TOKEN`` (and the provider blocks' former ``key_env`` name) resolve from ``<root>/.env``;
  a copy in the asking profile's ``.env`` or in ``os.environ`` is never used
- the file is read at lookup time, so a rotated token is seen by the next lookup
- every other secret keeps its profile scope (multiplex isolation is unchanged)
"""

import pytest

from agent import secret_scope as ss

CENTRAL = "oa-central-token"
STALE = "oa-stale-agent-copy"


@pytest.fixture(autouse=True)
def _reset_multiplex():
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


@pytest.fixture
def install(tmp_path, monkeypatch):
    """A custom-root install: ``<root>/.env`` plus a ``geo-agent`` profile holding a stale copy."""
    root = tmp_path / "root"
    profile = root / "profiles" / "geo-agent"
    profile.mkdir(parents=True)
    (root / ".env").write_text(f"OA_TOKEN={CENTRAL}\nOPENAI_API_KEY=sk-root-only\n")
    (profile / ".env").write_text(f"OA_TOKEN={STALE}\nHERMES_CUSTOM_OFFICE_AGENT_API_KEY={STALE}\n")
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setenv("OA_TOKEN", STALE)  # what a profile process loaded into its env at startup
    return root, profile


def test_a_profile_reads_the_root_token_not_its_own_copy(install):
    assert ss.get_secret("OA_TOKEN") == CENTRAL


def test_a_multiplexed_profile_turn_reads_the_root_token(install):
    ss.set_multiplex_active(True)
    token = ss.set_secret_scope({"OA_TOKEN": STALE}, profile_home=str(install[1]))
    try:
        assert ss.get_secret("OA_TOKEN") == CENTRAL
    finally:
        ss.reset_secret_scope(token)
    # No scope while multiplexing: an install-wide secret is not a leak, so no fail-closed raise.
    assert ss.get_secret("OA_TOKEN") == CENTRAL


def test_a_rotated_token_is_seen_on_the_next_lookup(install):
    root, _ = install
    assert ss.get_secret("OA_TOKEN") == CENTRAL
    (root / ".env").write_text("OA_TOKEN=oa-rotated\n")
    assert ss.get_secret("OA_TOKEN") == "oa-rotated"


def test_the_former_provider_key_env_reads_the_central_token(install):
    from hermes_cli.runtime_provider_custom import _key_env_secret

    assert ss.get_secret("HERMES_CUSTOM_OFFICE_AGENT_API_KEY") == CENTRAL
    assert _key_env_secret({"key_env": "HERMES_CUSTOM_OFFICE_AGENT_API_KEY"}, "office-agent") == CENTRAL
    assert _key_env_secret({"key_env": "OA_TOKEN"}, "office-agent") == CENTRAL


def test_no_root_token_is_no_token_even_with_copies_elsewhere(install):
    root, _ = install
    (root / ".env").write_text("OPENAI_API_KEY=sk-root-only\n")
    assert ss.get_secret("OA_TOKEN") is None
    assert ss.get_secret("OA_TOKEN", "fallback") == "fallback"


def test_other_secrets_keep_their_profile_scope(install):
    ss.set_multiplex_active(True)
    token = ss.set_secret_scope({}, profile_home=str(install[1]))
    try:
        assert ss.get_secret("OPENAI_API_KEY") is None
    finally:
        ss.reset_secret_scope(token)
    with pytest.raises(ss.UnscopedSecretError):
        ss.get_secret("OPENAI_API_KEY")


def test_readers_that_prefer_a_profile_env_still_get_the_central_token(install):
    # These read the asking profile's own .env first for every other key. The model client for a
    # custom provider resolves its key_env through the first one; a stale copy used to win there.
    from agent.credential_pool import get_env_prefer_dotenv
    from hermes_cli.config import get_env_value, get_env_value_prefer_dotenv

    for name in ("OA_TOKEN", "HERMES_CUSTOM_OFFICE_AGENT_API_KEY"):
        assert get_env_prefer_dotenv(name) == CENTRAL
        assert get_env_value_prefer_dotenv(name) == CENTRAL
        assert get_env_value(name) == CENTRAL


def test_without_a_root_token_those_readers_find_none_either(install):
    from agent.credential_pool import get_env_prefer_dotenv
    from hermes_cli.config import get_env_value, get_env_value_prefer_dotenv

    root, _ = install
    (root / ".env").write_text("OPENAI_API_KEY=sk-root-only\n")
    assert get_env_prefer_dotenv("OA_TOKEN") == ""
    assert not get_env_value_prefer_dotenv("OA_TOKEN")
    assert get_env_value("HERMES_CUSTOM_OFFICE_AGENT_API_KEY") is None
