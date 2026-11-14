"""Size fitter for Office Agent command-telemetry events.

The OA service rejects a body over 5,120 bytes with 413 and stores nothing, and a dropped
event is an unmonitored command, so an event is never dropped for size: it is shrunk until
its serialized UTF-8 body fits :data:`MAX_BODY_BYTES`, which keeps 1 KiB of headroom under
the service limit. Order (re-measured after each step): the output preview keeps its start,
the command text keeps its start and its end around an omission marker, the cwd keeps its end.
Ids, ``event_type``, ``schema_version`` and ``execution.status`` are never shortened; the
builder bounds them (and leaves out an over-long ``source.agent``/``tool_call_id``/``command.tool``),
so a body with every shrinkable field emptied always fits.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Callable

#: OA accepts 5,120 bytes; we stay well under it.
MAX_BODY_BYTES = 4096

_TWO_BYTE_ESCAPES = frozenset('"\\\n\r\t\b\f')


def encode(event: dict[str, Any]) -> bytes:
    """The exact request body: compact JSON, UTF-8, non-ASCII left unescaped (smaller)."""
    return json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _json_cost(ch: str) -> int:
    """Bytes ``ch`` takes inside a JSON string serialized by :func:`encode`."""
    code = ord(ch)
    if code < 0x20:
        return 2 if ch in _TWO_BYTE_ESCAPES else 6
    if ch in _TWO_BYTE_ESCAPES:
        return 2
    if code < 0x80:
        return 1
    if code < 0x800:
        return 2
    return 3 if code < 0x10000 else 4


def json_cost(text: str) -> int:
    return sum(_json_cost(ch) for ch in text)


def prefix_within(text: str, budget: int) -> str:
    """Longest prefix of ``text`` whose serialized cost is at most ``budget``. Slicing a ``str``
    by code point always cuts on a UTF-8 character boundary."""
    used = 0
    for index, ch in enumerate(text):
        used += _json_cost(ch)
        if used > budget:
            return text[:index]
    return text


def suffix_within(text: str, budget: int) -> str:
    """Longest suffix of ``text`` whose serialized cost is at most ``budget``."""
    used = 0
    for index in range(len(text) - 1, -1, -1):
        used += _json_cost(text[index])
        if used > budget:
            return text[index + 1:]
    return text


def exceeds(text: str, budget: int) -> bool:
    """Whether ``text`` costs more than ``budget``; scans at most ``budget + 1`` bytes of it."""
    return prefix_within(text, budget) != text


def omission_marker(omitted_bytes: int) -> str:
    return f"\n…[{omitted_bytes} bytes omitted]…\n"


def shorten_middle(text: str, budget: int) -> str:
    """``text`` cut to its start and its end around :func:`omission_marker`, within ``budget``.

    A dangerous part of a long command (a heredoc writing a script, a trailing ``| sh``) can sit at
    either end, so both survive: ~3/4 of the room left after the marker goes to the start, ~1/4 to
    the end. The marker is sized for the whole text first, so the real (smaller or equal) count fits.
    Never longer than ``text``: one no longer than the marker is returned whole.
    """
    marker_cost = json_cost(omission_marker(len(text.encode("utf-8"))))
    if not exceeds(text, max(budget, marker_cost)):
        return text
    room = max(0, budget - marker_cost)
    head = prefix_within(text, room * 3 // 4)
    tail = suffix_within(text[len(head):], room - json_cost(head))
    omitted = text[len(head):len(text) - len(tail)]
    return head + omission_marker(len(omitted.encode("utf-8"))) + tail


def _room_for(event: dict[str, Any], set_field: Callable[[str], None], limit: int) -> int:
    """Bytes available to one string field: the limit minus the body with that field empty."""
    set_field("")
    return limit - len(encode(event))


def _fit_preview(event: dict[str, Any], limit: int) -> None:
    output = (event.get("execution") or {}).get("output")
    if not isinstance(output, dict) or not output.get("preview"):
        return
    preview = output["preview"]
    room = _room_for(event, lambda value: output.__setitem__("preview", value), limit)
    output["preview"] = prefix_within(preview, max(0, room))
    output["truncated"] = True


def _fit_text(event: dict[str, Any], limit: int) -> None:
    command = event["command"]
    text = command.get("text")
    if not text:
        return
    room = _room_for(event, lambda value: command.__setitem__("text", value), limit)
    command["text"] = shorten_middle(text, room)


def _fit_cwd(event: dict[str, Any], limit: int) -> None:
    command = event["command"]
    cwd = command.get("cwd")
    if not cwd:
        return
    room = _room_for(event, lambda value: command.__setitem__("cwd", value), limit)
    kept = suffix_within(cwd, max(0, room))
    if kept:
        command["cwd"] = kept
    else:
        del command["cwd"]


_SHRINK_STEPS = (_fit_preview, _fit_text, _fit_cwd)


def fit_event(event: dict[str, Any], limit: int = MAX_BODY_BYTES) -> bytes:
    """The request body for ``event``, shrunk to at most ``limit`` bytes. ``event`` is not mutated;
    an event already under the limit serializes unchanged."""
    body = encode(event)
    if len(body) <= limit:
        return body
    event = copy.deepcopy(event)
    for step in _SHRINK_STEPS:
        if len(body) <= limit:
            return body
        step(event, limit)
        body = encode(event)
    if len(body) > limit:
        # Unreachable while the builder bounds the ids: the shrinkable fields are all at their floor.
        raise ValueError(f"command telemetry event cannot fit {limit} bytes ({len(body)})")
    return body
