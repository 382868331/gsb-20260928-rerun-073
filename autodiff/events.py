"""External-effect events and durable training event logs.

A checkpointed segment is executed once during the forward pass and *again*
during backward() (and once more for every repeated backward).  Random ops
are safe to replay because the RNG state is restored, but an arbitrary
external operation (network call, counter increment, file write, ...) is
not: replaying it would produce the effect a second time.

record_external_event makes such effects explicit:

* inside a recomputation interval it never calls the user function again --
  the result recorded on the first forward is returned, and an internal
  replay entry is appended so user-visible events stay distinguishable from
  framework-internal recomputation;
* an event that cannot be replayed safely (unknown id, no event log, or an
  unresolved event) is rejected *before* any effect can be produced;
* with a durable EventLog, confirmed events survive process restart and are
  never submitted twice, even after a crash between "submitted" and
  "committed".

The journal is append-only.  A failure leaves the submitted entry in place
without any erasure, so an effect that really happened can never be rolled
back silently; the caller must reconcile it explicitly.
"""
from __future__ import annotations

import contextvars
import json
import os
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np


class ExternalEventError(RuntimeError):
    """Base class for external-event handling errors."""


class ReplayExternalEventError(ExternalEventError):
    """Raised during recomputation before any effect could be reproduced."""


class PendingExternalEventError(ExternalEventError):
    """An event is submitted without a matching committed record.

    Its external effect may or may not have happened.  The journal is left
    untouched; inspect the external system and call
    EventLog.resolve_pending() once the outcome is known.
    """

    def __init__(self, event_id: str):
        super().__init__(
            f"external event {event_id!r} is submitted but not committed; "
            "its effect may have happened, so it cannot be retried or rolled "
            "back automatically -- reconcile it with EventLog.resolve_pending()"
        )
        self.event_id = event_id


_LOG_STACK: contextvars.ContextVar[tuple["EventLog", ...]] = contextvars.ContextVar(
    "event_log_stack", default=()
)
_REPLAY: contextvars.ContextVar[bool] = contextvars.ContextVar("event_replay", default=False)


def current_log() -> "EventLog | None":
    stack = _LOG_STACK.get()
    return stack[-1] if stack else None


def is_replaying() -> bool:
    """True while a checkpoint segment is being recomputed in backward."""
    return _REPLAY.get()


class use_event_log:
    """Context manager making *log* the active event log."""

    def __init__(self, log: "EventLog"):
        self._log = log
        self._token = None

    def __enter__(self) -> "EventLog":
        self._token = _LOG_STACK.set(_LOG_STACK.get() + (self._log,))
        return self._log

    def __exit__(self, *exc) -> None:
        _LOG_STACK.reset(self._token)


class _replay_scope:
    """Internal context marking checkpoint recomputation."""

    def __init__(self, log: "EventLog | None"):
        self._log = log
        self._replay_token = None
        self._log_token = None

    def __enter__(self):
        self._replay_token = _REPLAY.set(True)
        if self._log is not None:
            self._log_token = _LOG_STACK.set(_LOG_STACK.get() + (self._log,))
        return self

    def __exit__(self, *exc):
        if self._log_token is not None:
            _LOG_STACK.reset(self._log_token)
        _REPLAY.reset(self._replay_token)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return value


def _safe_jsonable(value: Any) -> Any:
    value = _to_jsonable(value)
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        value = {"__unserializable_repr__": repr(value)}
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EventLog:
    """Append-only, durable JSONL journal of external training events.

    Each user-visible event has two records:

        {"type": "event", "id": ..., "status": "submitted"}
        {"type": "event", "id": ..., "status": "committed", "result": ...}

    Checkpoint recomputation appends {"type": "replay", ...} records, which
    never carry new effects.
    """

    def __init__(self, path: str | os.PathLike | None = None):
        self.path = None if path is None else os.fspath(path)
        # event id -> latest record for that event
        self._events: dict[str, dict] = {}
        self._replays: list[dict] = []
        if self.path is not None and os.path.exists(self.path):
            self._load()

    # ------------------------------------------------------------ journal io
    def _load(self) -> None:
        pending: dict[str, dict] = {}
        with open(self.path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ExternalEventError(
                        f"corrupt event journal {self.path!r} at line {line_no}: {exc}"
                    ) from exc
                kind = entry.get("type")
                if kind == "event":
                    event_id = entry["id"]
                    status = entry["status"]
                    if status == "submitted":
                        pending[event_id] = entry
                    elif status == "committed":
                        pending.pop(event_id, None)
                        self._events[event_id] = entry
                    else:
                        raise ExternalEventError(
                            f"unknown event status {status!r} in journal line {line_no}"
                        )
                elif kind == "replay":
                    self._replays.append(entry)
                else:
                    raise ExternalEventError(
                        f"unknown journal entry type {kind!r} in line {line_no}"
                    )
        if pending:
            event_id = next(iter(pending))
            raise PendingExternalEventError(event_id)

    def _append(self, entry: dict) -> None:
        entry = dict(entry)
        entry.setdefault("ts", _utc_now())
        line = json.dumps(entry, sort_keys=True, ensure_ascii=False)
        if self.path is None:
            # In-memory log: still authoritative for the live process.
            self._apply(entry)
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._apply(entry)

    def _apply(self, entry: dict) -> None:
        if entry["type"] == "event":
            self._events[entry["id"]] = entry
        elif entry["type"] == "replay":
            self._replays.append(entry)

    # ------------------------------------------------------------- inspection
    def events(self) -> list[dict]:
        """User-visible external events in submission order."""
        return [dict(e) for e in self._events.values()]

    def replays(self) -> list[dict]:
        """Internal recomputation records (never new external effects)."""
        return [dict(e) for e in self._replays]

    def result(self, event_id: str) -> Any:
        entry = self._events[event_id]
        if entry["status"] != "committed":
            raise PendingExternalEventError(event_id)
        return entry.get("result")

    def is_committed(self, event_id: str) -> bool:
        return self._events.get(event_id, {}).get("status") == "committed"

    # ------------------------------------------------------------ public api
    def record(self, event_id: str, fn: Callable[[], Any]) -> Any:
        return record_external_event(event_id, fn, log=self)

    def record_replay(self, event_id: str) -> Any:
        if not self.is_committed(event_id):
            if event_id in self._events:
                raise PendingExternalEventError(event_id)
            raise ReplayExternalEventError(
                f"external event {event_id!r} has no committed effect on this "
                "recomputation interval; it cannot be replayed safely. Register "
                "it with record_external_event() during the original forward "
                "pass using a durable EventLog"
            )
        self._append({"type": "replay", "event_id": event_id})
        return self.result(event_id)

    def resolve_pending(self, event_id: str, result: Any = None) -> Any:
        """Mark a submitted-but-not-committed event as externally confirmed.

        Only call this after checking the external system.  Nothing is ever
        deleted from the journal.
        """
        entry = self._events.get(event_id)
        if entry is None:
            raise KeyError(f"no event with id {event_id!r}")
        if entry["status"] == "committed":
            return entry.get("result")
        self._append(
            {"type": "event", "id": event_id, "status": "committed",
             "result": _safe_jsonable(result), "reconciled": True}
        )
        return self.result(event_id)


def record_external_event(
    event_id: str,
    fn: Callable[[], Any],
    *,
    log: "EventLog | None" = None,
) -> Any:
    """Run an external effect exactly once and make it checkpoint-replayable.

    event_id must be stable across the original execution and any replay or
    process restart (e.g. "step-7-batch-sample").  fn performs the effect
    and returns whatever the model needs (numbers, arrays, dicts).

    During checkpoint recomputation fn is never called; the committed result
    is returned.  Without an EventLog, replay is impossible and raises
    ReplayExternalEventError before any effect.
    """
    if not isinstance(event_id, str) or not event_id:
        raise ValueError("event_id must be a non-empty string")
    log = log if log is not None else current_log()

    if is_replaying():
        if log is None:
            raise ReplayExternalEventError(
                f"external event {event_id!r} was encountered while "
                "recomputing a checkpoint interval without an EventLog; the "
                "original effect cannot be reproduced without risking a "
                "duplicate. Use an EventLog + record_external_event() for any "
                "external operation inside a checkpointed function"
            )
        return log.record_replay(event_id)

    if log is None:
        # No persistence requested: behaves as a normal call; a later replay
        # of an enclosing checkpoint rejects it (see the replay branch above).
        return fn()

    entry = log._events.get(event_id)
    if entry is not None:
        if entry["status"] == "committed":
            # Reopened journal / retried step: the effect already happened.
            return entry.get("result")
        raise PendingExternalEventError(event_id)

    # Durable "intent" record *before* the effect: a crash here means the
    # caller must reconcile instead of blindly resubmitting.
    log._append({"type": "event", "id": event_id, "status": "submitted"})
    result = fn()
    log._append(
        {"type": "event", "id": event_id, "status": "committed",
         "result": _safe_jsonable(result)}
    )
    return result
