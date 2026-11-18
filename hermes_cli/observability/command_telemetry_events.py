"""Event builder for Office Agent command telemetry.

Turns one command Hermes runs into the two OA events, ``command.started`` and
``command.finished``, with exactly the fields the OA ingest contract names (anything else on
Hermes's side never reaches the body). Identity: the triple ``source.instance_id`` +
``source.session_id`` + ``command.id`` is identical in a command's two events and never reused,
because ``command.id`` is a fresh UUID4 per command; each event gets its own UUID4 ``event_id``,
which a retry reuses because the body is built once. ``source.agent`` is the name of the profile
the command ran in (``default`` for the root), so OA can tell an install's agents apart.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Optional

SCHEMA_VERSION = "1.0"
STARTED = "command.started"
FINISHED = "command.finished"

#: Contract caps on the identity fields. Hermes's own ids are far shorter; a longer one is hashed
#: (stably, so both events of the command still carry the same value) instead of shortened.
_ID_MAX = 256
#: Optional ``source``/``command`` strings over this length are left out rather than sent.
OPTIONAL_MAX_CHARS = 256
#: An output preview never needs more characters than the body has bytes.
PREVIEW_CAP_CHARS = 4096


def new_event_id() -> str:
    return f"evt_{uuid.uuid4().hex}"


def new_command_id() -> str:
    return f"cmd_{uuid.uuid4().hex}"


def utc_timestamp(when: Optional[datetime] = None) -> str:
    moment = (when or datetime.now(UTC)).astimezone(UTC)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


def clean_text(value: Any) -> str:
    """``value`` as a ``str`` that encodes to valid UTF-8 (lone surrogates become ``?``)."""
    return str(value).encode("utf-8", "replace").decode("utf-8")


def bounded_id(value: Any, *, fallback: str) -> str:
    text = clean_text(value) if value else fallback
    if len(text) <= _ID_MAX:
        return text
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CommandRecord:
    """One command, captured just before it runs. ``agent`` is the profile it runs in, captured here
    because a background command's ``finished`` event is sent from outside that profile's scope."""

    agent: str
    instance_id: str
    session_id: str
    command_id: str
    tool: str
    text: str
    tool_call_id: Optional[str] = None
    cwd: Optional[str] = None


@dataclass(frozen=True)
class CommandOutcome:
    """How a command ended. ``exit_code`` is ``None`` when the process gave none (killed, timed out,
    never started); ``output`` is the start of the combined output and ``total_bytes`` the size of the
    full output it was taken from."""

    status: str
    exit_code: Optional[int] = None
    duration_ms: Optional[int] = None
    output: Optional[str] = None
    total_bytes: Optional[int] = None


def _optional(target: dict[str, Any], key: str, value: Optional[str]) -> None:
    if value and len(value) <= OPTIONAL_MAX_CHARS:
        target[key] = value


def _source(record: CommandRecord) -> dict[str, Any]:
    source: dict[str, Any] = {}
    _optional(source, "agent", record.agent)
    source.update(instance_id=record.instance_id, session_id=record.session_id)
    _optional(source, "tool_call_id", record.tool_call_id)
    return source


def _envelope(event_type: str, event_id: str, timestamp: str, record: CommandRecord) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "event_id": event_id, "event_type": event_type,
            "timestamp": timestamp, "source": _source(record)}


def build_started(record: CommandRecord, *, event_id: str, timestamp: str) -> dict[str, Any]:
    event = _envelope(STARTED, event_id, timestamp, record)
    command: dict[str, Any] = {"id": record.command_id}
    _optional(command, "tool", record.tool)
    command["text"] = record.text
    if record.cwd:
        command["cwd"] = record.cwd
    event["command"] = command
    return event


def _output_block(outcome: CommandOutcome) -> Optional[dict[str, Any]]:
    if outcome.output is None:
        return None
    preview = outcome.output[:PREVIEW_CAP_CHARS]
    preview_bytes = len(preview.encode("utf-8"))
    total = max(outcome.total_bytes or 0, preview_bytes)
    return {"preview": preview, "total_bytes": total, "truncated": preview_bytes < total}


def build_finished(record: CommandRecord, outcome: CommandOutcome, *, event_id: str,
                   timestamp: str) -> dict[str, Any]:
    event = _envelope(FINISHED, event_id, timestamp, record)
    event["command"] = {"id": record.command_id}
    execution: dict[str, Any] = {"status": outcome.status, "exit_code": outcome.exit_code}
    if outcome.duration_ms is not None:
        execution["duration_ms"] = max(0, int(outcome.duration_ms))
    if (output := _output_block(outcome)) is not None:
        execution["output"] = output
    event["execution"] = execution
    return event
