"""Which Hermes tools run commands, and how to read a command and its outcome from a tool call.

One table (:data:`COMMAND_TOOLS`) names every tool that executes something on the customer's
machine: ``terminal`` (shell), ``execute_code`` (a Python script, whose own tool calls are
reported separately when they reach ``terminal``), ``browser_exec`` (Python driving the browser
CLI) and ``process_manage`` ``write``/``submit`` (input typed into a running shell or REPL).
Command text and output previews are redacted with Hermes's own secret masker before they leave
the machine.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .command_telemetry_events import PREVIEW_CAP_CHARS, CommandOutcome, clean_text

logger = logging.getLogger(__name__)

_TIMED_OUT_MARKER = "[Command timed out after"
_INTERRUPTED_MARKER = "[Command interrupted]"

#: ``post_tool_call`` statuses that say how the call ended regardless of its result text.
_HOOK_STATUS = {"blocked": "blocked", "cancelled": "cancelled", "timeout": "timed_out"}

#: A result's own ``status`` word, mapped onto the OA vocabulary (unknown words pass through).
_RESULT_STATUS = {
    "success": "completed", "ok": "completed", "timeout": "timed_out", "interrupted": "cancelled",
    "error": "error", "blocked": "blocked", "pending_approval": "pending_approval", "disabled": "disabled",
}


@dataclass(frozen=True)
class CommandCall:
    text: str
    cwd: Optional[str] = None


def _redact(text: str) -> str:
    from agent.redact import redact_sensitive_text

    return redact_sensitive_text(text, force=True)


def _string(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value.strip() else None


def _terminal_cwd(args: dict[str, Any], task_id: str) -> Optional[str]:
    """Best-effort: explicit ``workdir`` > this session's recorded cwd > the configured default.
    Mirrors ``tools.terminal_tool``'s precedence; container remapping is not reproduced."""
    if workdir := _string(args.get("workdir")):
        return workdir
    try:
        from tools.approval import get_current_session_key
        from tools.terminal_tool import _get_env_config, get_session_cwd

        return get_session_cwd(get_current_session_key(default="") or task_id) or _string(_get_env_config().get("cwd"))
    except Exception:  # a broken TERMINAL_* value must not cost the event, only its optional cwd
        logger.debug("command telemetry: could not resolve terminal cwd", exc_info=True)
        return None


def _terminal(args: dict[str, Any], task_id: str) -> Optional[CommandCall]:
    text = _string(args.get("command"))
    return CommandCall(text, _terminal_cwd(args, task_id)) if text else None


def _code(args: dict[str, Any], _task_id: str) -> Optional[CommandCall]:
    text = _string(args.get("code"))
    return CommandCall(text) if text else None


def _process_input(args: dict[str, Any], _task_id: str) -> Optional[CommandCall]:
    if args.get("action") not in ("write", "submit"):
        return None
    text = _string(args.get("data"))
    return CommandCall(text) if text else None


#: tool name -> reads the command from the call's arguments (None: this call runs no command).
COMMAND_TOOLS: dict[str, Callable[[dict[str, Any], str], Optional[CommandCall]]] = {
    "terminal": _terminal,
    "execute_code": _code,
    "browser_exec": _code,
    "process_manage": _process_input,
}


def command_for(tool_name: str, args: Any, task_id: str = "") -> Optional[CommandCall]:
    """The command a call to ``tool_name`` runs, redacted and UTF-8 clean, or None."""
    reader = COMMAND_TOOLS.get(tool_name)
    if reader is None or not isinstance(args, dict):
        return None
    call = reader(args, task_id or "")
    if call is None:
        return None
    return CommandCall(clean_text(_redact(call.text)), clean_text(call.cwd) if call.cwd else None)


def _result_dict(result: Any) -> Optional[dict[str, Any]]:
    if isinstance(result, dict):
        return result
    if not isinstance(result, str):
        return None
    try:
        parsed = json.loads(result)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def background_process_id(tool_name: str, result: Any, hook_status: Optional[str] = None) -> Optional[str]:
    """The process-registry id when a terminal call left its command running in the background
    (``background=true``, a timeout promoted to background, a foreground command yielded)."""
    if tool_name != "terminal" or hook_status in _HOOK_STATUS:
        return None
    data = _result_dict(result) or {}
    proc_id = data.get("session_id")
    return proc_id if isinstance(proc_id, str) and proc_id else None


def _exit_status(code: int, data: dict[str, Any], output: str) -> tuple[str, Optional[int]]:
    if code == -1 and data.get("error"):
        return "error", None
    if code == 124 and _TIMED_OUT_MARKER in output:
        return "timed_out", None
    if code == 130 and _INTERRUPTED_MARKER in output:
        return "cancelled", None
    if code < 0:
        return "killed", None
    return ("completed" if code == 0 else "failed"), code


def _status_and_exit(data: dict[str, Any], output: str) -> tuple[str, Optional[int]]:
    word = data.get("status")
    if isinstance(word, str) and word:
        return _RESULT_STATUS.get(word, word[:64]), None
    code = data.get("exit_code")
    if isinstance(code, int) and not isinstance(code, bool):
        return _exit_status(code, data, output)
    return ("error" if data.get("error") else "completed"), None


def _full_output_bytes(data: dict[str, Any], output: str) -> int:
    """Size of the full output: Hermes's spill file when the capture window overflowed, else the
    output itself (``output_total_chars`` is the fallback when the spill file is gone)."""
    path = data.get("full_output_path")
    if isinstance(path, str) and path:
        try:
            return os.path.getsize(path)
        except OSError:
            pass
    total_chars = data.get("output_total_chars")
    shown = len(output.encode("utf-8"))
    return max(shown, total_chars) if isinstance(total_chars, int) else shown


def outcome_for(result: Any, *, hook_status: Optional[str] = None,
                duration_ms: Optional[int] = None) -> CommandOutcome:
    """How a finished (not backgrounded) command call ended, from its tool result."""
    if hook_status in _HOOK_STATUS:
        return CommandOutcome(_HOOK_STATUS[hook_status], None, duration_ms)
    data = _result_dict(result)
    if data is None:
        return CommandOutcome("error", None, duration_ms)
    raw_output = data.get("output")
    if not isinstance(raw_output, str) or not raw_output:
        return CommandOutcome(*_status_and_exit(data, ""), duration_ms)
    status, exit_code = _status_and_exit(data, raw_output)
    preview = clean_text(_redact(raw_output[:PREVIEW_CAP_CHARS]))
    return CommandOutcome(status, exit_code, duration_ms, preview, _full_output_bytes(data, raw_output))
