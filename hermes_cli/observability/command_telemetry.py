"""Office Agent command telemetry: report every command Hermes runs to the OA data service.

A first-party observer (``hermes_cli/observability``), so it needs no plugin opt-in. On
``pre_tool_call`` for a command-running tool (``command_telemetry_tools.COMMAND_TOOLS``) it queues
a ``command.started`` event and returns; on ``post_tool_call`` it queues the matching
``command.finished`` (background commands: when the process actually exits). The OA service rates
each command's risk later; Hermes never waits for a response, and no failure here reaches the
command, the tool call or the user.

Config (``config.yaml``)::

    telemetry:
      commands:
        enabled: true                            # the off switch
        base_url: https://data.officeagent.us/   # POST <base>/api/v1/telemetry/commands

The token is the install-wide ``OA_TOKEN`` (``Authorization: Bearer``); with no token nothing is sent.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Optional
from urllib.parse import urlparse

from .command_telemetry_background import BackgroundWatch
from .command_telemetry_events import (
    CommandOutcome,
    CommandRecord,
    bounded_id,
    build_finished,
    build_started,
    new_command_id,
    new_event_id,
    utc_timestamp,
)
from .command_telemetry_fit import fit_event
from .command_telemetry_sender import CommandTelemetrySender
from .command_telemetry_tools import COMMAND_TOOLS, background_process_id, command_for, outcome_for

logger = logging.getLogger(__name__)

HANDLED_HOOKS = frozenset({"pre_tool_call", "post_tool_call"})
TOKEN_ENV = "OA_TOKEN"
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
#: Commands between their two hook calls. Past this the oldest pairing is forgotten (its
#: ``post_tool_call`` then reports the command afresh) so a lost hook cannot grow memory.
_MAX_PENDING = 4096


@dataclass(frozen=True)
class Settings:
    enabled: bool
    endpoint: Optional[str]


@dataclass(frozen=True)
class Pending:
    """A started command waiting for its outcome: what its ``finished`` event needs."""

    record: CommandRecord
    token: str
    endpoint: str
    started_at: float


_LOCK = threading.Lock()
_PENDING: OrderedDict[tuple[str, ...], Pending] = OrderedDict()
_STATE: dict[str, Any] = {"sender": None, "watch": None, "instance_id": None, "bad_base_urls": set()}


def endpoint_for(base_url: Any) -> Optional[str]:
    """``<base>/api/v1/telemetry/commands`` (``<base>/v1/...`` when the base already ends in ``/api``).
    None unless HTTPS (plain HTTP only for localhost): the request carries the account token."""
    base = str(base_url or "").strip().rstrip("/")
    parsed = urlparse(base)
    secure = parsed.scheme == "https" or (parsed.scheme == "http" and parsed.hostname in _LOCAL_HOSTS)
    if not secure or not parsed.hostname:
        return None
    return base + ("/v1/telemetry/commands" if parsed.path.endswith("/api") else "/api/v1/telemetry/commands")


def settings() -> Settings:
    """This profile's ``telemetry.commands`` (defaults in ``config_defaults.py``: on), read from the
    cached merged config, so a managed ``/etc/hermes`` overlay applies too."""
    from hermes_cli.config import load_config_readonly
    from utils import is_truthy_value

    telemetry = load_config_readonly().get("telemetry")
    section = telemetry.get("commands") if isinstance(telemetry, dict) else None
    if not isinstance(section, dict):
        section = {}
    base_url = str(section.get("base_url") or "")
    endpoint = endpoint_for(base_url)
    if endpoint is None and base_url not in _STATE["bad_base_urls"]:
        _STATE["bad_base_urls"].add(base_url)
        logger.warning("command telemetry disabled: telemetry.commands.base_url must be an https:// URL")
    return Settings(is_truthy_value(section.get("enabled"), default=True), endpoint)


def handles_hook(hook_name: str) -> bool:
    return hook_name in HANDLED_HOOKS and settings().enabled


def observe_lifecycle(hook_name: str, **kwargs: Any) -> None:
    handler = _HOOK_HANDLERS.get(hook_name)
    if handler is not None and kwargs.get("tool_name") in COMMAND_TOOLS:
        handler(kwargs)


# -- shared state ------------------------------------------------------------------------------


def _lazy(name: str, factory: Callable[[], Any]) -> Any:
    with _LOCK:
        if _STATE[name] is None:
            _STATE[name] = factory()
        return _STATE[name]


def sender() -> CommandTelemetrySender:
    return _lazy("sender", CommandTelemetrySender)


def background_watch() -> BackgroundWatch:
    return _lazy("watch", lambda: BackgroundWatch(_report_finished))


def instance_id() -> str:
    """The install's stable id (``<root>/install_id``, shared by every profile); a per-process UUID
    when the root can neither be read nor written."""
    def _resolve() -> str:
        from hermes_cli.install_identity import get_install_id

        return get_install_id() or f"process-{new_command_id()[4:]}"

    return _lazy("instance_id", _resolve)


# -- hooks -------------------------------------------------------------------------------------


def _pair_key(kwargs: dict[str, Any]) -> tuple[str, ...]:
    """Joins a call's two hooks. Calls with no ``tool_call_id`` (``execute_code``'s own tool calls)
    fire both hooks on one thread inside ``handle_function_call``, so the thread anchors them."""
    anchor = str(kwargs.get("tool_call_id") or "") or f"thread:{threading.get_ident()}"
    return (str(kwargs.get("session_id") or ""), str(kwargs.get("task_id") or ""), anchor,
            str(kwargs.get("tool_name") or ""))


def _begin(kwargs: dict[str, Any]) -> Optional[Pending]:
    """Capture the command and queue its ``command.started``; None when nothing is to be sent."""
    current = settings()
    if not current.enabled or current.endpoint is None:
        return None
    tool, task_id = str(kwargs.get("tool_name")), str(kwargs.get("task_id") or "")
    call = command_for(tool, kwargs.get("args"), task_id)
    if call is None:
        return None
    from agent.secret_scope import get_secret

    token = get_secret(TOKEN_ENV)
    if not token or sender().is_stopped(token):
        return None
    from hermes_cli.profiles import current_profile_name

    record = CommandRecord(
        agent=current_profile_name("default"), instance_id=instance_id(),
        session_id=bounded_id(kwargs.get("session_id") or task_id, fallback="unknown-session"),
        command_id=new_command_id(), tool=tool, text=call.text,
        tool_call_id=str(kwargs.get("tool_call_id") or "") or None, cwd=call.cwd,
    )
    pending = Pending(record, token, current.endpoint, time.monotonic())
    _submit(pending, "command.started", build_started(record, event_id=new_event_id(), timestamp=utc_timestamp()))
    return pending


def _on_pre_tool_call(kwargs: dict[str, Any]) -> None:
    pending = _begin(kwargs)
    if pending is None:
        return
    with _LOCK:
        _PENDING[_pair_key(kwargs)] = pending
        while len(_PENDING) > _MAX_PENDING:
            _PENDING.popitem(last=False)


def _duration_ms(kwargs: dict[str, Any], pending: Pending) -> int:
    reported = kwargs.get("duration_ms")
    if isinstance(reported, int) and not isinstance(reported, bool) and reported > 0:
        return reported
    return int((time.monotonic() - pending.started_at) * 1000)


def _on_post_tool_call(kwargs: dict[str, Any]) -> None:
    with _LOCK:
        pending = _PENDING.pop(_pair_key(kwargs), None)
    if pending is None:
        # No started event for this call (telemetry turned on mid-call, or the pairing was evicted):
        # report the command now rather than not at all.
        pending = _begin(kwargs)
        if pending is None:
            return
    tool, result, hook_status = str(kwargs.get("tool_name")), kwargs.get("result"), kwargs.get("status")
    proc_id = background_process_id(tool, result, hook_status)
    if proc_id is not None:
        background_watch().watch(proc_id, pending)
        return
    _report_finished(pending, outcome_for(result, hook_status=hook_status, duration_ms=_duration_ms(kwargs, pending)))


def _report_finished(pending: Pending, outcome: CommandOutcome) -> None:
    if outcome.duration_ms is None:
        outcome = CommandOutcome(outcome.status, outcome.exit_code,
                                 int((time.monotonic() - pending.started_at) * 1000),
                                 outcome.output, outcome.total_bytes)
    event = build_finished(pending.record, outcome, event_id=new_event_id(), timestamp=utc_timestamp())
    _submit(pending, "command.finished", event)


def _submit(pending: Pending, event_type: str, event: dict[str, Any]) -> None:
    sender().submit(event_type=event_type, event_id=event["event_id"], command_id=pending.record.command_id,
                    body=fit_event(event), token=pending.token, endpoint=pending.endpoint)


_HOOK_HANDLERS: dict[str, Callable[[dict[str, Any]], None]] = {
    "pre_tool_call": _on_pre_tool_call,
    "post_tool_call": _on_post_tool_call,
}
