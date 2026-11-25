"""Identity headers on model requests to Office Agent: which agent, session and installation sent them.

Only ``model.officeagent.us`` gets them: they name a person's install, so no other provider may see them. The
values must stay the ones Office Agent command telemetry reports for the same work (``source.agent``,
``source.session_id``, ``source.instance_id``), so OA can join a session's model usage with its commands; they are
read from the same helpers. They ride in HTTP headers beside the bearer token; the request body never changes.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

from utils import base_url_host_matches

logger = logging.getLogger(__name__)

OFFICE_AGENT_HOST = "model.officeagent.us"
AGENT_HEADER = "X-OA-Agent"
SESSION_HEADER = "X-OA-Session-Id"
INSTANCE_HEADER = "X-OA-Instance-Id"

# The session whose turn or compaction is running. The compactors make their side calls with a runtime of their own
# that holds no session id, so those requests know their exact session only from here. A ContextVar: concurrent
# sessions never see each other's, and the compaction worker thread inherits the caller's.
_running_session: ContextVar[Optional[str]] = ContextVar("office_agent_running_session", default=None)


@contextmanager
def running_session(session_id: Optional[str]) -> Iterator[None]:
    """Name ``session_id`` on requests sent inside the block that do not bring a session id themselves."""
    token = _running_session.set(session_id or None)
    try:
        yield
    finally:
        _running_session.reset(token)


def _sendable(value: str) -> bool:
    """A header value the HTTP client accepts: it raises on anything else, which would fail the model request."""
    return bool(value) and value.isascii() and value.isprintable() and value == value.strip()


def _identity(session_id: Optional[str]) -> dict[str, str]:
    from hermes_cli.observability.command_telemetry import instance_id
    from hermes_cli.observability.command_telemetry_events import OPTIONAL_MAX_CHARS, bounded_id
    from hermes_cli.profiles import current_profile_name

    agent = current_profile_name("default") or ""
    values = {
        AGENT_HEADER: agent if len(agent) <= OPTIONAL_MAX_CHARS else "",
        SESSION_HEADER: bounded_id(session_id or _running_session.get(), fallback=""),
        INSTANCE_HEADER: instance_id(),
    }
    return {header: value for header, value in values.items() if _sendable(value)}


def office_agent_headers(base_url: Optional[str], session_id: Optional[str] = None) -> dict[str, str]:
    """The ``X-OA-*`` headers for a request to Office Agent, ``{}`` for any other host.

    A value that is missing or unfit for a header is left out; building them never fails the request.
    """
    if not base_url_host_matches(base_url or "", OFFICE_AGENT_HOST):
        return {}
    try:
        return _identity(session_id)
    except Exception:
        logger.warning("Office Agent headers could not be built; sending the request without them", exc_info=True)
        return {}
