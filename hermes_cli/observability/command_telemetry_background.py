"""``command.finished`` for terminal commands that outlive their tool call.

``terminal(background=true)``, a timeout promoted to the background and a foreground command
yielded to the background all return while the command keeps running. The tool result is not how
the command ended, so the command is watched in Hermes's process registry and its
``command.finished`` is sent when the registry records the exit (or loses the process).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional

from .command_telemetry_events import PREVIEW_CAP_CHARS, CommandOutcome, clean_text

logger = logging.getLogger(__name__)

POLL_SECONDS = 2.0
#: Commands watched at once; past this the oldest is reported as ``lost`` rather than leaked.
MAX_WATCHED = 1000

_REASON_STATUS = {"killed": "killed", "lost": "lost", "failed_start": "failed_to_start"}


def _lookup(proc_id: str) -> Any:
    from tools.process_registry import process_registry

    return process_registry.get(proc_id)


def _redact_output(text: str, command: str) -> str:
    from agent.redact import redact_terminal_output
    from tools.ansi_strip import strip_ansi

    return redact_terminal_output(strip_ansi(text), command, force=True)


def outcome_from_session(session: Any) -> CommandOutcome:
    """How a background process ended, from its registry record."""
    reason = getattr(session, "completion_reason", "exited") or "exited"
    code = getattr(session, "exit_code", None)
    if reason in _REASON_STATUS:
        status, exit_code = _REASON_STATUS[reason], None
    elif not isinstance(code, int):
        status, exit_code = "unknown", None
    elif code < 0:
        status, exit_code = "killed", None
    else:
        status, exit_code = ("completed" if code == 0 else "failed"), code
    started, exited = getattr(session, "started_at", 0.0) or 0.0, getattr(session, "exited_at", 0.0) or 0.0
    duration_ms = int((exited - started) * 1000) if started and exited >= started else None
    buffer = getattr(session, "output_buffer", "") or ""
    if not buffer:
        return CommandOutcome(status, exit_code, duration_ms)
    # The registry keeps a rolling tail; past it, total_output_chars counts what scrolled away.
    scrolled = max(0, int(getattr(session, "total_output_chars", 0) or 0) - len(buffer))
    total = len(buffer.encode("utf-8")) + scrolled
    preview = clean_text(_redact_output(buffer[:PREVIEW_CAP_CHARS], getattr(session, "command", "") or ""))
    return CommandOutcome(status, exit_code, duration_ms, preview, total)


class BackgroundWatch:
    """Polls the process registry for watched commands; one daemon thread while any are watched."""

    def __init__(self, on_exit: Callable[[Any, CommandOutcome], None], *,
                 lookup: Callable[[str], Any] = _lookup, poll_seconds: float = POLL_SECONDS,
                 start_thread: bool = True):
        self._on_exit, self._lookup, self._poll = on_exit, lookup, poll_seconds
        self._start_thread = start_thread
        self._lock = threading.Lock()
        self._watched: dict[str, tuple[Any, float]] = {}
        self._thread: Optional[threading.Thread] = None

    def watch(self, proc_id: str, pending: Any) -> None:
        evicted = None
        with self._lock:
            self._watched[proc_id] = (pending, time.monotonic())
            if len(self._watched) > MAX_WATCHED:
                oldest = min(self._watched, key=lambda key: self._watched[key][1])
                evicted = self._watched.pop(oldest)[0]
            if self._start_thread and (self._thread is None or not self._thread.is_alive()):
                # Process-wide watcher over the process-wide registry: each watched command already
                # carries its owning profile's token and endpoint, so no profile scope is needed.
                thread = threading.Thread(target=self._run, name="oa-command-telemetry-bg", daemon=True)  # health: allow HX012 -- process-wide; each event carries its own profile scope
                self._thread = thread
                self._thread.start()
        if evicted is not None:
            self._on_exit(evicted, CommandOutcome("lost"))

    def watching(self) -> int:
        with self._lock:
            return len(self._watched)

    def poll_once(self) -> int:
        """Report every watched command that has ended; returns how many are still running."""
        with self._lock:
            watched = list(self._watched.items())
        for proc_id, (pending, _since) in watched:
            session = self._lookup(proc_id)
            if session is not None and not getattr(session, "exited", False):
                continue
            with self._lock:
                if self._watched.pop(proc_id, None) is None:
                    continue
            self._on_exit(pending, CommandOutcome("lost") if session is None else outcome_from_session(session))
        return self.watching()

    def _run(self) -> None:
        while True:
            time.sleep(self._poll)
            try:
                remaining = self.poll_once()
            except Exception:
                logger.warning("command telemetry background watch failed", exc_info=True)
                continue
            with self._lock:
                if remaining == 0 and not self._watched:
                    self._thread = None
                    return
