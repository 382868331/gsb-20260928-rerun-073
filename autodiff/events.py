"""Durable and checkpoint-local event tracking."""
from __future__ import annotations

import json
import os
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable


@dataclass
class _Segment:
    phase: str
    events: dict[str, dict[str, Any]] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)


_SEGMENT_STACK: list[_Segment] = []
_LOG_STACK: list["TrainingEventLog"] = []
_STEP_STACK: list["TrainingStep"] = []


class EventExistsError(RuntimeError):
    """Raised when a committed event id is submitted with different data."""


class UnsupportedExternalEventError(RuntimeError):
    """Raised before an effect when it cannot be replayed safely."""


class TrainingStepError(RuntimeError):
    """Base class for resumable training-step failures."""


class ExternalEffectUncertain(TrainingStepError):
    """Raise from apply() when a request may have reached the external system."""


class ExternalEffectNotExecuted(TrainingStepError):
    """Raise from apply() when the external system definitely did not act."""


class UnconfirmedEffectError(TrainingStepError):
    """Raised when resume reaches an effect whose outcome is unknown."""


class IncompleteTrainingStepError(TrainingStepError):
    """Raised when completion is requested before computation is finished."""


def _json_default(value: Any):
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"event payload is not JSON serializable: {type(value)!r}")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_json_default)


def _digest(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _model_snapshot(model: Any) -> dict[str, Any]:
    params = list(model.parameters())
    import numpy as np
    return {
        "parameter_count": len(params),
        "parameters": [np.asarray(p.data).tolist() for p in params],
        "gradients": [None if p.grad is None else np.asarray(p.grad.data).tolist() for p in params],
    }


class TrainingEventLog:
    """Append-only JSONL log of user-visible training events."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._history: list[dict[str, Any]] = []
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
        if event_id in self._events:
            old = self._events[event_id]
            if kind == "user":
                if old != event:
                    raise EventExistsError(
                        f"training event {event_id!r} is already committed; "
                        "it cannot be submitted again"
                    )
                return old
        return self._append_event(event, overwrite=kind != "user")

    def _append_event(self, event: dict[str, Any], *, overwrite: bool = False) -> dict[str, Any]:
        event = deepcopy(event)
        event_id = event["id"]
        if event_id in self._events and not overwrite:
            raise EventExistsError(f"training event {event_id!r} is already committed")
        encoded = json.dumps(event, ensure_ascii=False, sort_keys=True, default=_json_default)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(encoded + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._history.append(deepcopy(event))
        self._events[event_id] = event
        return event

    def append(
        self,
        event_id: str,
        payload: Any = None,
        *,
        kind: str,
        phase: str = "user_visible",
        overwrite: bool = False,
        **extra: Any,
    ) -> dict[str, Any]:
        event = {"id": str(event_id), "kind": str(kind), "payload": payload, "phase": str(phase)}
        event.update(extra)
        return self._append_event(event, overwrite=overwrite)

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

    def begin_step(self, step_id: str | int, effects: list[str] | tuple[str, ...]) -> "TrainingStep":
        return TrainingStep(self, step_id, effects)

    def resume_step(self, step_id: str | int) -> "TrainingStep":
        return TrainingStep.resume(self, step_id)

    def step(self, step_id: str | int, effects: list[str] | tuple[str, ...]) -> "TrainingStep":
        step_key = TrainingStep._step_key(step_id)
        if step_key in self._events:
            return TrainingStep.resume(self, step_id)
        return TrainingStep(self, step_id, effects)

    def step_state(self, step_id: str | int) -> str:
        record = self.get(TrainingStep._step_key(step_id))
        return str(record["payload"]["state"])

    def step_failure(self, step_id: str | int) -> dict[str, Any] | None:
        payload = self.get(TrainingStep._step_key(step_id))["payload"]
        return payload.get("failure") if payload.get("state") == "failed" else None

    def effects(self, step_id: str | int) -> dict[str, dict[str, Any]]:
        prefix = f"{TrainingStep._step_key(step_id)}:effect:"
        names = list(self.get(TrainingStep._step_key(step_id))["payload"]["effects"])
        records = {key[len(prefix):]: deepcopy(event["payload"]) for key, event in self._events.items() if key.startswith(prefix)}
        for name in names:
            records.setdefault(name, {"step": str(step_id), "name": name, "status": "not_executed", "payload": None})
        return records

    def effect_statuses(self, step_id: str | int) -> dict[str, str]:
        return {name: payload["status"] for name, payload in self.effects(step_id).items()}

    def effect(self, step_id: str | int, effect_name: str) -> dict[str, Any]:
        return deepcopy(self.get(TrainingStep._effect_key(step_id, effect_name))["payload"])

    def verify_effect(self, step_id: str | int, effect_name: str, occurred: bool, payload: Any = None, *, note: str = ""):
        return self.resume_step(step_id).verify_effect(effect_name, occurred, payload, note=note)

    def is_step_complete(self, step_id: str | int) -> bool:
        try:
            return self.step_state(step_id) == "complete"
        except KeyError:
            return False

    def audit_events(self, step_id: str | int) -> tuple[dict[str, Any], ...]:
        step_key = TrainingStep._step_key(step_id)
        prefix = f"{step_key}:"
        return tuple(deepcopy(e) for e in self._history if e["id"] == step_key or e["id"].startswith(prefix))

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
                    and event.get("kind") == "user"
                    and events[event_id] != event
                ):
                    raise EventExistsError(
                        f"conflicting duplicate event {event_id!r} at line {line_no}"
                    )
                self._history.append(event)
                events[event_id] = event
        return events


class TrainingStep:
    """Durable state machine for one training round and its external effects."""

    def __init__(self, log: TrainingEventLog, step_id: str | int, effects: list[str] | tuple[str, ...]):
        self.log = log
        self.step_id = str(step_id)
        self.effect_names = [str(name) for name in effects]
        if len(self.effect_names) != len(set(self.effect_names)):
            raise ValueError("effect names within a training step must be unique")
        key = self._step_key(self.step_id)
        if key in log:
            raise EventExistsError(f"training step {self.step_id!r} already exists; use resume_step")
        from .core import get_rng_state
        self._write_state("started", effects=self.effect_names, rng_state=get_rng_state())

    @staticmethod
    def _step_key(step_id: str | int) -> str:
        return f"training-step:{step_id}"

    @staticmethod
    def _effect_key(step_id: str | int, effect_name: str) -> str:
        return f"{TrainingStep._step_key(step_id)}:effect:{effect_name}"

    @classmethod
    def resume(cls, log: TrainingEventLog, step_id: str | int) -> "TrainingStep":
        step_id = str(step_id)
        payload = log.get(cls._step_key(step_id))["payload"]
        self = cls.__new__(cls)
        self.log = log
        self.step_id = step_id
        self.effect_names = list(payload["effects"])
        if payload.get("rng_state") is not None:
            from .core import set_rng_state
            set_rng_state(payload["rng_state"])
        return self

    def __enter__(self):
        _STEP_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        if _STEP_STACK and _STEP_STACK[-1] is self:
            _STEP_STACK.pop()
        if exc_type is None or self.state == "complete":
            return False
        self.fail(exc)
        return False

    @property
    def state(self) -> str:
        return str(self.log.get(self._step_key(self.step_id))["payload"]["state"])

    def _write_state(self, state: str, **extra: Any) -> dict[str, Any]:
        payload = {"state": state, "step": self.step_id, "effects": self.effect_names}
        payload.update(extra)
        return self.log.append(
            self._step_key(self.step_id), payload,
            kind="training_step", phase="training", overwrite=True,
        )

    def _effect_record(self, effect_name: str) -> dict[str, Any]:
        name = str(effect_name)
        return self.log.get(self._effect_key(self.step_id, name))["payload"]

    def effect_status(self, effect_name: str) -> str:
        return str(self._effect_record(effect_name)["status"])

    def effect_payload(self, effect_name: str) -> Any:
        record = self._effect_record(effect_name)
        if record["status"] != "occurred":
            raise UnconfirmedEffectError(f"effect {effect_name!r} is {record['status']}, not occurred")
        return deepcopy(record["payload"])

    def statuses(self) -> dict[str, str]:
        return {name: self.effect_status(name) if self._has_effect(name) else "not_executed" for name in self.effect_names}

    def _has_effect(self, name: str) -> bool:
        return self._effect_key(self.step_id, name) in self.log

    def _assert_can_run(self, effect_name: str) -> None:
        if self.state == "complete":
            raise EventExistsError(f"training step {self.step_id!r} is complete")
        name = str(effect_name)
        if name not in self.effect_names:
            raise KeyError(f"effect {name!r} is not declared for step {self.step_id!r}")
        index = self.effect_names.index(name)
        for prior in self.effect_names[:index]:
            if not self._has_effect(prior) or self._effect_record(prior)["status"] != "occurred":
                status = self.effect_status(prior) if self._has_effect(prior) else "not_executed"
                raise UnconfirmedEffectError(
                    f"cannot run {name!r}: earlier effect {prior!r} is {status}"
                )

    def run_effect(
        self,
        effect_name: str,
        apply: Callable[[], Any] | None = None,
        replay: Callable[[Any], Any] | None = None,
    ) -> Any:
        name = str(effect_name)
        key = self._effect_key(self.step_id, name)
        if key in self.log:
            record = self.log.get(key)["payload"]
            status = record["status"]
            if status == "occurred":
                return replay(record["payload"]) if replay is not None else deepcopy(record["payload"])
            if status == "not_occurred":
                pass
            else:
                raise UnconfirmedEffectError(
                    f"effect {name!r} is unconfirmed; call verify_effect before resuming"
                )
        else:
            self._assert_can_run(name)
            if apply is None:
                raise ValueError("apply is required for an effect that has not occurred")
        try:
            payload = apply()
        except ExternalEffectUncertain as exc:
            self._write_effect(name, "unconfirmed", reason=str(exc))
            self.fail(exc)
            raise
        except ExternalEffectNotExecuted as exc:
            self._write_effect(name, "not_occurred", reason=str(exc))
            self.fail(exc)
            raise
        except Exception as exc:
            self._write_effect(name, "unconfirmed", reason=repr(exc))
            self.fail(exc)
            raise
        self._write_effect(name, "occurred", payload)
        return replay(payload) if replay is not None else deepcopy(payload)

    def _write_effect(self, name: str, status: str, payload: Any = None, reason: str | None = None) -> dict[str, Any]:
        record = {
            "step": self.step_id,
            "name": name,
            "index": self.effect_names.index(name),
            "status": status,
            "payload": payload,
            "reason": reason,
        }
        return self.log.append(
            self._effect_key(self.step_id, name), record,
            kind="training_effect", phase="training", overwrite=True,
        )

    def verify_effect(self, effect_name: str, occurred: bool, payload: Any = None, *, note: str = "") -> dict[str, Any]:
        name = str(effect_name)
        if name not in self.effect_names:
            raise KeyError(f"effect {name!r} is not declared for step {self.step_id!r}")
        key = self._effect_key(self.step_id, name)
        if key in self.log:
            current = self.log.get(key)["payload"]
            status = current["status"]
            if status == "occurred":
                if not occurred or (payload is not None and payload != current["payload"]):
                    raise EventExistsError(f"effect {name!r} was already verified as occurred")
                payload = current["payload"]
            elif status == "not_occurred":
                if occurred:
                    raise EventExistsError(f"effect {name!r} was already verified as not occurred")
                self._append_verification(name, False, None, note)
                return self.log.get(key)
            elif status == "unconfirmed":
                self._write_effect(name, "occurred" if occurred else "not_occurred", payload if occurred else None)
            else:
                raise RuntimeError(f"unknown effect status {status!r}")
        elif occurred:
            self._assert_can_run(name)
            self._write_effect(name, "occurred", payload)
        else:
            raise UnconfirmedEffectError(f"effect {name!r} has no attempted or uncertain record to verify")
        self._append_verification(name, occurred, payload if occurred else None, note)
        if self.state == "failed":
            self._write_state("resumable", resumed_from=name)
        return self.log.get(key)

    def _append_verification(self, name: str, occurred: bool, payload: Any, note: str) -> dict[str, Any]:
        sequence = [e for e in self.log.events if e.get("kind") == "training_verification" and e["payload"].get("step") == self.step_id]
        verification_id = f"{self._step_key(self.step_id)}:verification:{len(sequence)}"
        return self.log.append(
            verification_id,
            {"step": self.step_id, "name": name, "occurred": bool(occurred), "payload": deepcopy(payload), "note": note},
            kind="training_verification", phase="training",
        )

    def fail(self, reason: BaseException | str | None = None) -> dict[str, Any]:
        if self.state in {"complete", "failed"}:
            return self.log.get(self._step_key(self.step_id))
        statuses = self.statuses()
        uncertain = [name for name, status in statuses.items() if status == "unconfirmed"]
        reason_text = repr(reason) if isinstance(reason, BaseException) else (str(reason) if reason is not None else None)
        return self._write_state("failed", statuses=statuses, uncertain=uncertain, failure={"reason": reason_text})

    def complete(
        self,
        model: Any,
        *,
        gradients_computed: bool = True,
        optimizer: Any | None = None,
        loss: Any | None = None,
    ) -> dict[str, Any]:
        statuses = self.statuses()
        if any(status != "occurred" for status in statuses.values()):
            exc = IncompleteTrainingStepError(f"cannot complete step {self.step_id!r}: {statuses}")
            self.fail(exc)
            raise exc
        params = list(model.parameters())
        if not gradients_computed or not any(p.grad is not None for p in params):
            exc = IncompleteTrainingStepError("gradients for this training round are not complete")
            self.fail(exc)
            raise exc
        before = _model_snapshot(model)
        loss_value = None if loss is None else float(loss.data if hasattr(loss, "data") else loss)
        grad_digest = _digest(before["gradients"])
        if optimizer is not None:
            optimizer.step()
        after = _model_snapshot(model)
        effect_digest = _digest({name: self.effect_payload(name) for name in self.effect_names})
        from .core import get_rng_state
        completion = {
            "loss": loss_value,
            "effect_digest": effect_digest,
            "model_before_optimizer": before,
            "model_after_optimizer": after,
            "gradient_digest": grad_digest,
            "rng_state": get_rng_state(),
        }
        self._write_state("complete", statuses=statuses, completion=completion)
        return self.log.get(self._step_key(self.step_id))


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

    if _STEP_STACK:
        step = _STEP_STACK[-1]
        if segment is not None and segment.phase == "recompute" and replay is None:
            raise UnsupportedExternalEventError(
                "external operations in recomputable training require a pure replay callback; "
                "effects whose result cannot be reconstructed are rejected before they run"
            )
        return step.run_effect(event_id, apply, replay)

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
