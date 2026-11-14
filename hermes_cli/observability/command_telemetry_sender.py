"""Background sender for Office Agent command telemetry.

Monitoring, not prevention: nothing here ever runs on the command's thread or reaches the tool
call. Events wait in memory (at most :data:`MAX_WAITING`; when full the oldest waiting event is
dropped), at most :data:`MAX_IN_FLIGHT` requests run at once, and a failure worth retrying is
re-sent with the SAME body (so the same ``event_id``) on the 1-2-4-8-16-32 s then 60 s backoff
until 15 minutes after its first attempt. Events still waiting when Hermes stops are lost, which
the OA contract accepts. Logs name the event type, ``event_id``, ``command.id``, the HTTP status
and OA's ``error`` message only: never the command text, its output, the body or the token.
"""

from __future__ import annotations

import enum
import hashlib
import http.client
import itertools
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 10
MAX_IN_FLIGHT = 4
MAX_WAITING = 1000
BACKOFF_SECONDS = (1, 2, 4, 8, 16, 32)
BACKOFF_CAP_SECONDS = 60
GIVE_UP_AFTER_SECONDS = 15 * 60

#: OA's 403 for an account that does not have command telemetry. Matched case-insensitively.
ORG_ONLY_ERROR = "Only an organization account or its members can send command telemetry"

_USER_AGENT = "hermes-agent-command-telemetry/1"


@dataclass(frozen=True)
class Response:
    status: int
    error: str = ""
    rate_limit_reset: Optional[str] = None


def post(endpoint: str, body: bytes, token: str, *, timeout: float = REQUEST_TIMEOUT_SECONDS) -> Response:
    """POST one event. Raises on transport failure (timeout, DNS, refused, reset); never on an HTTP status."""
    request = urllib.request.Request(endpoint, data=body, method="POST", headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {token}", "User-Agent": _USER_AGENT,
    })
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return Response(response.status)
    except urllib.error.HTTPError as exc:
        return Response(exc.code, _error_message(exc.read(2048) if exc.fp else b""),
                        exc.headers.get("X-RateLimit-Reset") if exc.headers else None)


def _error_message(raw: bytes) -> str:
    try:
        data = json.loads(raw.decode("utf-8", "replace"))
    except ValueError:
        return ""
    error = data.get("error") if isinstance(data, dict) else None
    return error[:300] if isinstance(error, str) else ""


class Action(enum.Enum):
    DONE = "done"
    RETRY = "retry"
    DROP = "drop"
    BUG = "bug"
    STOP_TOKEN = "stop_token"


#: Statuses that are final without a retry; ``BUG`` ones mean Hermes built a bad event.
_FINAL = {400: Action.BUG, 401: Action.DROP, 402: Action.DROP, 403: Action.DROP, 413: Action.BUG}


def classify(response: Optional[Response]) -> Action:
    """What to do with an attempt's result (``None`` = timeout or network error)."""
    if response is None or response.status == 429 or response.status >= 500:
        return Action.RETRY
    if 200 <= response.status < 300:
        return Action.DONE
    if response.status == 403 and ORG_ONLY_ERROR.lower() in response.error.lower():
        return Action.STOP_TOKEN
    return _FINAL.get(response.status, Action.DROP)


def backoff_delay(failed_attempts: int) -> float:
    """Seconds to wait after the ``failed_attempts``-th failure (1-based)."""
    index = failed_attempts - 1
    return float(BACKOFF_SECONDS[index] if index < len(BACKOFF_SECONDS) else BACKOFF_CAP_SECONDS)


def _rate_limit_delay(response: Optional[Response], wall: float) -> Optional[float]:
    """Seconds until ``X-RateLimit-Reset`` (epoch seconds), or None when absent or unreadable."""
    if response is None or response.status != 429 or not response.rate_limit_reset:
        return None
    try:
        reset = float(response.rate_limit_reset)
    except ValueError:
        return None
    return max(0.0, reset - wall)


def token_key(token: str) -> str:
    """The in-memory key a token is remembered by (never the token itself)."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass
class Outgoing:
    event_type: str
    event_id: str
    command_id: str
    body: bytes
    token: str
    endpoint: str
    seq: int = 0
    due: float = 0.0
    first_attempt: Optional[float] = None
    failures: int = 0
    token_key: str = field(default="", repr=False)


PostFn = Callable[[str, bytes, str], Response]


class CommandTelemetrySender:
    """In-memory queue + :data:`MAX_IN_FLIGHT` daemon workers. ``start_workers=False`` leaves the
    queue to :meth:`pump`, which tests drive with a fake clock."""

    def __init__(self, *, post_fn: PostFn = post, clock: Callable[[], float] = time.monotonic,
                 wall: Callable[[], float] = time.time, workers: int = MAX_IN_FLIGHT,
                 max_waiting: int = MAX_WAITING, start_workers: bool = True):
        self._post, self._clock, self._wall = post_fn, clock, wall
        self._workers, self._max_waiting, self._start_workers = workers, max_waiting, start_workers
        self._cond = threading.Condition()
        self._waiting: dict[int, Outgoing] = {}
        self._seq = itertools.count(1)
        self._threads: list[threading.Thread] = []
        self._stopped_tokens: set[str] = set()
        self._closed = False

    # -- producer side (hook threads) ------------------------------------------------------------

    def is_stopped(self, token: str) -> bool:
        return token_key(token) in self._stopped_tokens

    def submit(self, *, event_type: str, event_id: str, command_id: str, body: bytes, token: str,
               endpoint: str) -> None:
        """Queue one event; returns at once."""
        key = token_key(token)
        if key in self._stopped_tokens:
            return
        item = Outgoing(event_type, event_id, command_id, body, token, endpoint, token_key=key)
        with self._cond:
            item.seq, item.due = next(self._seq), self._clock()
            self._enqueue_locked(item)
            self._ensure_workers()

    def _enqueue_locked(self, item: Outgoing) -> None:
        """Caller holds the lock. Over :data:`MAX_WAITING`, the oldest waiting event goes."""
        if self._closed:
            return
        self._waiting[item.seq] = item
        if len(self._waiting) > self._max_waiting:
            oldest = self._waiting.pop(min(self._waiting))
            logger.warning("command telemetry queue full: dropped %s %s (command %s)",
                           oldest.event_type, oldest.event_id, oldest.command_id)
        self._cond.notify()

    def _ensure_workers(self) -> None:
        if not self._start_workers or self._threads:
            return
        # One process-wide sender serves every profile: each event carries the token, endpoint and
        # body captured in its owning profile's scope, so a worker reads no profile state and must
        # not inherit whichever profile happened to start it.
        for index in range(self._workers):
            name = f"oa-command-telemetry-{index}"
            thread = threading.Thread(target=self._run, name=name, daemon=True)  # health: allow HX012 -- process-wide; each event carries its own profile scope
            thread.start()
            self._threads.append(thread)

    # -- consumer side ---------------------------------------------------------------------------

    def waiting(self) -> int:
        with self._cond:
            return len(self._waiting)

    def next_due(self) -> Optional[float]:
        with self._cond:
            return min((item.due for item in self._waiting.values()), default=None)

    def _take_ready(self) -> Optional[Outgoing]:
        """Caller holds the lock. The ready item that is due first (oldest on a tie), if any."""
        now = self._clock()
        ready = [item for item in self._waiting.values() if item.due <= now]
        if not ready:
            return None
        item = min(ready, key=lambda candidate: (candidate.due, candidate.seq))
        del self._waiting[item.seq]
        return item

    def _run(self) -> None:
        while True:
            with self._cond:
                item = self._take_ready()
                while item is None and not self._closed:
                    due = min((queued.due for queued in self._waiting.values()), default=None)
                    self._cond.wait(None if due is None else max(0.0, due - self._clock()))
                    item = self._take_ready()
                if item is None:
                    return
            try:
                self._attempt(item)
            except Exception as exc:  # one bad event must not kill a worker (and with four, telemetry)
                logger.error("command telemetry bug: %s %s (command %s) failed with %s",
                             item.event_type, item.event_id, item.command_id, type(exc).__name__)

    def pump(self) -> int:
        """Send every item due now, synchronously, on the calling thread; returns how many were sent."""
        sent = 0
        while True:
            with self._cond:
                item = self._take_ready()
            if item is None:
                return sent
            self._attempt(item)
            sent += 1

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._waiting.clear()
            self._cond.notify_all()

    # -- one attempt -----------------------------------------------------------------------------

    def _attempt(self, item: Outgoing) -> None:
        if item.token_key in self._stopped_tokens:
            return
        if item.first_attempt is None:
            item.first_attempt = self._clock()
        try:
            response: Optional[Response] = self._post(item.endpoint, item.body, item.token)
        except (OSError, http.client.HTTPException) as exc:
            logger.debug("command telemetry %s %s: %s", item.event_type, item.event_id, type(exc).__name__)
            response = None
        self._settle(item, response)

    def _settle(self, item: Outgoing, response: Optional[Response]) -> None:
        action = classify(response)
        status = response.status if response is not None else "network error"
        error = response.error if response is not None else ""
        if action is Action.RETRY:
            self._retry(item, response)
        elif action is Action.STOP_TOKEN:
            self._stop_token(item, error)
        elif action is Action.BUG:
            logger.error("command telemetry bug: OA rejected %s %s (command %s) with %s: %s",
                         item.event_type, item.event_id, item.command_id, status, error)
        elif action is Action.DROP:
            logger.warning("command telemetry: OA refused %s %s (command %s) with %s: %s",
                           item.event_type, item.event_id, item.command_id, status, error)

    def _retry(self, item: Outgoing, response: Optional[Response]) -> None:
        item.failures += 1
        delay = _rate_limit_delay(response, self._wall())
        if delay is None:
            delay = backoff_delay(item.failures)
        now = self._clock()
        if now + delay - (item.first_attempt or now) > GIVE_UP_AFTER_SECONDS:
            logger.warning("command telemetry: gave up on %s %s (command %s) after %d attempts",
                           item.event_type, item.event_id, item.command_id, item.failures)
            return
        item.due = now + delay
        with self._cond:
            self._enqueue_locked(item)

    def _stop_token(self, item: Outgoing, error: str) -> None:
        with self._cond:
            first = item.token_key not in self._stopped_tokens
            self._stopped_tokens.add(item.token_key)
            for seq in [seq for seq, queued in self._waiting.items() if queued.token_key == item.token_key]:
                del self._waiting[seq]
        if first:
            logger.warning("command telemetry stopped for this Office Agent token until Hermes restarts: %s", error)

