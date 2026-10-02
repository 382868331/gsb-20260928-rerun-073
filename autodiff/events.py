"""Durable and checkpoint-local event tracking."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class _Segment:
    phase: str
    events: dict[str, dict[str, Any]] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)


_SEGMENT_STACK: list[_Segment] = []
_LOG_STACK: list["TrainingEventLog"] = []


class EventExistsError(RuntimeError):
    """Raised when a committed event id is submitted with different data."""


class UnsupportedExternalEventError(RuntimeError):
    """Raised before an effect when it cannot be replayed safely."""


class StepEffectError(RuntimeError):
    """An effect's apply callback raised; carries the step effect report.

    The report classifies every planned effect of the step as applied
    (the effect definitely happened), unconfirmed (the apply callback
    raised, so the effect may or may not have happened) or pending
    (never attempted).  Already-applied effects are durable facts and
    are never rolled back.
    """

    def __init__(self, event_id: str, reason: BaseException, report: dict):
        self.event_id = event_id
        self.reason = reason
        self.report = report
        super().__init__(
            f"effect {event_id!r} failed: {reason!r}; "
            f"applied={report['applied']} unconfirmed={report['unconfirmed']} "
            f"pending={report['pending']}"
        )


class UnconfirmedEffectError(RuntimeError):
    """A step reached an effect whose outcome is still unconfirmed.

    The user must call TrainingEventLog.verify() with a conclusion
    before the step can resume from this position.
    """

    def __init__(self, event_id: str, report: dict):
        self.event_id = event_id
        self.report = report
        super().__init__(
            f"effect {event_id!r} is unconfirmed; call log.verify({event_id!r}, "
            "occurred=...) before resuming the step"
        )


def _json_default(value: Any):
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"event payload is not JSON serializable: {type(value)!r}")


class TrainingEventLog:
    """Append-only JSONL log of user-visible training events."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._events = self._load_unlocked()

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._events.values())

    def __contains__(self, event_id: str) -> bool:
        return event_id in self._events

    def __len__(self) -> int:
        return len(self._events)

    def get(self, event_id: str) -> dict[str, Any]:
        return self._events[event_id]

    def record(self, event_id: str, payload: Any = None, *, kind: str = "user") -> dict[str, Any]:
        event_id = str(event_id)
        event = {"id": event_id, "kind": str(kind), "payload": payload, "phase": "user_visible"}
        return self._append(event)

    def _append(self, event: dict[str, Any]) -> dict[str, Any]:
        event_id = event["id"]
        if event_id in self._events:
            old = self._events[event_id]
            if old != event:
                if event.get("kind") in ("user", "effect", "verification", "step"):
                    raise EventExistsError(
                        f"training event {event_id!r} is already committed; "
                        "it cannot be submitted again"
                    )
            else:
                return old
        encoded = json.dumps(event, ensure_ascii=False, sort_keys=True, default=_json_default)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(encoded + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._events[event_id] = event
        return event

    def run(
        self,
        event_id: str,
        apply: Callable[[], Any],
        replay: Callable[[Any], Any] | None = None,
    ) -> Any:
        event_id = str(event_id)
        if event_id in self._events:
            payload = self._events[event_id]["payload"]
            return replay(payload) if replay is not None else payload
        if replay is None:
            raise UnsupportedExternalEventError(
                "external checkpoint events require replay(payload); without a pure "
                "reconstruction, recomputation could repeat the effect"
            )
        payload = apply()
        self.record(event_id, payload)
        return replay(payload)

    def rollback(self, reason: str | None = None) -> None:
        """Acknowledge failed training without deleting durable effects."""
        return None

    # -- step-scoped dependent effects -------------------------------------

    def step(self, step_id: str, plan: list[str] | None = None) -> "TrainingStep":
        """Open a training step that groups interdependent external effects."""
        return TrainingStep(self, str(step_id), plan)

    def _effect_attempts(self, step_id: str, name: str) -> list[dict[str, Any]]:
        base = f"effect/{step_id}/{name}"
        attempts = [
            e for e in self._events.values()
            if e.get("kind") == "effect" and (e["id"] == base or e["id"].startswith(base + "#"))
        ]
        attempts.sort(key=lambda e: e.get("attempt", 1))
        return attempts

    def _verification(self, attempt_id: str) -> dict[str, Any] | None:
        record = self._events.get(f"verify/{attempt_id}")
        return record if record is not None and record.get("kind") == "verification" else None

    def _attempt_state(self, attempt: dict[str, Any]) -> tuple[str, Any]:
        """Resolve an attempt to (state, payload) using any verification."""
        verification = self._verification(attempt["id"])
        if verification is not None:
            conclusion = verification["payload"]
            if conclusion["occurred"]:
                payload = attempt.get("payload")
                if payload is None:
                    payload = conclusion.get("payload")
                return "applied", payload
            return "aborted", None
        if attempt.get("status") == "applied":
            return "applied", attempt.get("payload")
        return "unconfirmed", None

    def verify(
        self,
        attempt_id: str,
        occurred: bool,
        payload: Any = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        """Record a user verification conclusion for an unconfirmed effect.

        Idempotent: re-submitting the same conclusion returns the existing
        record and never triggers another effect.  A conflicting conclusion
        raises EventExistsError.  Only a conclusion of occurred=False
        allows the step to re-apply the effect on resume.
        """
        attempt_id = str(attempt_id)
        attempt = self._events.get(attempt_id)
        if attempt is None or attempt.get("kind") != "effect":
            raise KeyError(f"unknown effect attempt {attempt_id!r}")
        conclusion = {
            "event": attempt_id,
            "occurred": bool(occurred),
            "payload": payload,
            "note": note,
        }
        if occurred and attempt.get("payload") is None and payload is None:
            raise ValueError(
                "verifying occurred=True requires the observed payload, "
                "because the failed apply did not return one"
            )
        record_id = f"verify/{attempt_id}"
        existing = self._events.get(record_id)
        if existing is not None:
            if existing["payload"] != conclusion:
                raise EventExistsError(
                    f"effect {attempt_id!r} already has a conflicting verification"
                )
            return existing
        return self.record(record_id, conclusion, kind="verification")

    def step_report(self, step_id: str, plan: list[str] | None = None) -> dict[str, Any]:
        """Classify a step's effects as applied / unconfirmed / pending."""
        step_id = str(step_id)
        names: list[str] = []
        for event in self._events.values():
            if event.get("kind") == "effect" and event.get("step") == step_id:
                if event["name"] not in names:
                    names.append(event["name"])
        planned = list(plan) if plan is not None else []
        for name in names:
            if name not in planned:
                planned.append(name)
        applied: list[str] = []
        unconfirmed: list[str] = []
        pending: list[str] = []
        for name in planned:
            attempts = self._effect_attempts(step_id, name)
            if not attempts:
                pending.append(name)
                continue
            state, _ = self._attempt_state(attempts[-1])
            if state == "applied":
                applied.append(name)
            elif state == "unconfirmed":
                unconfirmed.append(name)
            else:
                pending.append(name)
        return {
            "step": step_id,
            "applied": applied,
            "unconfirmed": unconfirmed,
            "pending": pending,
        }

    def record_internal(
        self, event_id: str, payload: Any, *, kind: str = "internal"
    ) -> dict[str, Any]:
        return self.record(event_id, payload, kind=kind)

    def record_rng_state(self, event_id: str | None = None) -> dict[str, Any]:
        from .core import get_rng_state
        sequence = [e for e in self._events.values() if e.get("kind") == "internal_rng_state"]
        index = len(sequence) if event_id is None else event_id
        snapshot_id = f"rng-state:{index}"
        event = self.record_internal(snapshot_id, get_rng_state(), kind="internal_rng_state")
        self.record_internal("rng-state:latest", snapshot_id, kind="internal_rng_pointer")
        return event

    def restore_rng_state(self, event_id: str = "rng-state:latest") -> None:
        from .core import set_rng_state
        pointer = self.get(event_id)["payload"]
        set_rng_state(self.get(pointer)["payload"])

    def user_events(self) -> tuple[dict[str, Any], ...]:
        return tuple(e for e in self.events if e.get("kind") == "user")

    def internal_events(self) -> tuple[dict[str, Any], ...]:
        return tuple(e for e in self.events if e.get("kind") != "user")

    @contextmanager
    def use(self):
        _LOG_STACK.append(self)
        try:
            yield self
        finally:
            _LOG_STACK.pop()

    def _load_unlocked(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        events: dict[str, dict[str, Any]] = {}
        with self.path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                event = json.loads(line)
                event_id = event["id"]
                if (
                    event_id in events
                    and event.get("kind") in ("user", "effect", "verification", "step")
                    and events[event_id] != event
                ):
                    raise EventExistsError(
                        f"conflicting duplicate event {event_id!r} at line {line_no}"
                    )
                events[event_id] = event
        return events


@contextmanager
def _segment(phase: str, events: dict[str, dict[str, Any]] | None = None):
    segment = _Segment(phase, events if events is not None else {})
    _SEGMENT_STACK.append(segment)
    try:
        yield segment
    finally:
        _SEGMENT_STACK.pop()


def _random_event(kind: str, event_id: str, shape: tuple[int, ...], draw: Callable[[], Any]) -> Any:
    if not _SEGMENT_STACK:
        return draw()
    segment = _SEGMENT_STACK[-1]
    if segment.phase == "recompute":
        index = segment.counters.get(kind, 0)
        segment.counters[kind] = index + 1
        return segment.events[f"{kind}:{index}:{event_id}"]["payload"]
    value = draw()
    index = segment.counters.get(kind, 0)
    segment.counters[kind] = index + 1
    record_id = f"{kind}:{index}:{event_id}"
    segment.events[record_id] = {
        "id": record_id,
        "kind": kind,
        "phase": "user_visible",
        "payload": value,
    }
    return value


def external_event(
    event_id: str,
    apply: Callable[[], Any],
    replay: Callable[[Any], Any] | None = None,
) -> Any:
    """Perform an external effect once and replay its value without effects."""
    event_id = str(event_id)
    log = _LOG_STACK[-1] if _LOG_STACK else None
    segment = _SEGMENT_STACK[-1] if _SEGMENT_STACK else None

    if replay is None:
        raise UnsupportedExternalEventError(
            "external operations in recomputable training require a pure replay callback; "
            "effects whose result cannot be reconstructed are rejected before they run"
        )

    if segment is not None and segment.phase == "recompute":
        if log is not None and event_id in log:
            payload = log.get(event_id)["payload"]
        else:
            payload = segment.events[event_id]["payload"]
        return replay(payload)

    if log is not None:
        result = log.run(event_id, apply, replay)
        payload = log.get(event_id)["payload"]
    else:
        payload = apply()
        result = replay(payload)

    if segment is not None:
        segment.events[event_id] = {
            "id": event_id,
            "kind": "user",
            "phase": "user_visible",
            "payload": payload,
        }
    return result


event = external_event
EventLog = TrainingEventLog


class TrainingStep:
    """One training step grouping interdependent external effects.

    Effects applied inside the step are durable facts: a failure never
    rolls them back.  The step is recorded as completed only when the
    body finishes without error and every attempted effect is applied;
    otherwise an interrupted record (with the applied / unconfirmed /
    pending classification) is appended, so a partially-executed step is
    never saved as a normal completed step.  Re-running the step resumes
    from the first unresolved effect: applied effects replay from the
    log, unconfirmed ones require log.verify() first, and only effects
    verified as not-occurred are re-applied (as a new, linked attempt).
    """

    def __init__(self, log: TrainingEventLog, step_id: str, plan: list[str] | None = None):
        self.log = log
        self.step_id = step_id
        self.plan = list(plan) if plan is not None else None
        self.completed = self._completion() is not None
        self._finished = False

    def _completion(self) -> dict[str, Any] | None:
        record = self.log._events.get(f"step/{self.step_id}")
        if record is not None and record.get("payload", {}).get("status") == "completed":
            return record
        return None

    def report(self) -> dict[str, Any]:
        return self.log.step_report(self.step_id, self.plan)

    def effect(
        self,
        name: str,
        apply: Callable[[], Any],
        replay: Callable[[Any], Any] | None = None,
    ) -> Any:
        """Run one named effect of this step, resuming from durable records."""
        if self.completed:
            raise EventExistsError(
                f"step {self.step_id!r} is already completed; its effects and "
                "parameters belong to a finished training round"
            )
        if replay is None:
            raise UnsupportedExternalEventError(
                "step effects require a pure replay(payload) callback so "
                "recomputation and restarts never repeat the effect"
            )
        name = str(name)
        attempts = self.log._effect_attempts(self.step_id, name)
        if attempts:
            state, payload = self.log._attempt_state(attempts[-1])
            if state == "applied":
                return replay(payload)
            if state == "unconfirmed":
                raise UnconfirmedEffectError(attempts[-1]["id"], self.report())
        try:
            payload = apply()
        except Exception as exc:
            failed = self._record_attempt(name, len(attempts) + 1, "unconfirmed", None, exc)
            report = self.report()
            report["reason"] = repr(exc)
            raise StepEffectError(failed["id"], exc, report) from exc
        attempt = self._record_attempt(name, len(attempts) + 1, "applied", payload, None)
        return replay(attempt["payload"])

    def _record_attempt(
        self,
        name: str,
        number: int,
        status: str,
        payload: Any,
        error: BaseException | None,
    ) -> dict[str, Any]:
        base = f"effect/{self.step_id}/{name}"
        attempt_id = base if number == 1 else f"{base}#{number}"
        event = {
            "id": attempt_id,
            "kind": "effect",
            "phase": "user_visible",
            "step": self.step_id,
            "name": name,
            "attempt": number,
            "status": status,
            "payload": payload,
            "error": repr(error) if error is not None else None,
        }
        return self.log._append(event)

    def __enter__(self) -> "TrainingStep":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.completed:
            return False
        report = self.report()
        if exc is not None or report["unconfirmed"]:
            if exc is not None:
                report["reason"] = repr(exc)
            self.log._append({
                "id": f"step/{self.step_id}/interrupted/{len(self._interruptions()) + 1}",
                "kind": "step",
                "phase": "user_visible",
                "payload": {"status": "interrupted", "step": self.step_id, "report": report},
            })
            return False
        self.log._append({
            "id": f"step/{self.step_id}",
            "kind": "step",
            "phase": "user_visible",
            "payload": {
                "status": "completed",
                "step": self.step_id,
                "effects": report["applied"],
            },
        })
        self.completed = True
        return False

    def _interruptions(self) -> list[dict[str, Any]]:
        prefix = f"step/{self.step_id}/interrupted/"
        return [
            e for e in self.log._events.values()
            if e.get("kind") == "step" and e["id"].startswith(prefix)
        ]
