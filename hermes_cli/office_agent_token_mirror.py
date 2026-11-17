"""Mirror the Office Agent token from the default profile's ``.env`` into every agent's own ``.env``.

The default profile's ``.env`` (the root home's) holds THE token; every other profile carries an
exact copy. Desktop runs ``hermes gateway ensure`` on every launch, and that command runs
:func:`mirror_office_agent_token` before it ensures the gateway:

- default's non-empty value is written into each profile's ``.env``, overwriting a different one;
- with no token in default, each profile's copy is removed, so no profile keeps a token default
  no longer has. An older build's token that lives only in an agent is removed too (owner
  decision 2026-11-17: re-entering it once is acceptable).

The copies are not what Hermes authenticates with: ``agent.secret_scope`` still answers ``OA_TOKEN``
from the root ``.env`` for every profile, so a token changed mid-session reaches every agent at once
and the copies catch up at the next launch.

Only profiles with a servable identity receive a copy (``named_profile_has_servable_identity``): an
empty ``.env`` is all a crashed ``profile create`` leaves, and seeding one would make the host count
it as a second tenant at its next boot.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TOKEN_ENV = "OA_TOKEN"


def mirror_office_agent_token() -> list[str]:
    """Make every named profile's ``OA_TOKEN`` match the default profile's. Returns the profiles
    whose ``.env`` changed. A profile that cannot be written is logged and skipped, never raised:
    this runs on the launch path and must not stop the gateway from coming up."""
    from agent.secret_scope import load_env_file
    from hermes_cli.config import env_write_refusal
    from hermes_cli.profiles import _iter_named_profile_dirs
    from hermes_constants import get_default_hermes_root, named_profile_has_servable_identity

    refusal = env_write_refusal(TOKEN_ENV, "mirror")
    if refusal:
        logger.info("Not mirroring %s into profiles: %s", TOKEN_ENV, refusal)
        return []

    try:
        profile_homes = _iter_named_profile_dirs()
    except OSError as exc:
        logger.warning("Could not list profiles to mirror %s into: %s", TOKEN_ENV, exc)
        return []

    token = load_env_file(get_default_hermes_root() / ".env").get(TOKEN_ENV) or ""
    changed = []
    for profile_home in profile_homes:
        if not named_profile_has_servable_identity(profile_home):
            continue
        try:
            if _mirror_into(profile_home / ".env", token):
                changed.append(profile_home.name)
        except (OSError, TimeoutError) as exc:
            logger.warning("Could not mirror %s into profile %s: %s", TOKEN_ENV, profile_home.name, exc)
    return changed


def _mirror_into(env_path: Path, token: str) -> bool:
    """Leave exactly one ``OA_TOKEN`` line holding ``token`` (none when ``token`` is empty), in place
    of the first existing one; every other line is kept as it is. True when the file was rewritten."""
    from hermes_cli.config import (
        _env_line_defines_key, _env_write_lock, _quote_env_value, _read_env_lines, _write_env_lines,
    )

    with _env_write_lock(env_path):
        exists = env_path.exists()
        lines = _read_env_lines(env_path) if exists else []
        first = next((i for i, line in enumerate(lines) if _env_line_defines_key(line, TOKEN_ENV)), None)
        mirrored = [line for line in lines if not _env_line_defines_key(line, TOKEN_ENV)]
        if token:
            assignment = f"{TOKEN_ENV}={_quote_env_value(token)}\n"
            if first is None:
                if mirrored and not mirrored[-1].endswith("\n"):
                    mirrored[-1] += "\n"
                mirrored.append(assignment)
            else:
                mirrored.insert(first, assignment)
        if mirrored == lines:
            return False
        _write_env_lines(env_path, mirrored, preserve_mode=exists)
    return True
