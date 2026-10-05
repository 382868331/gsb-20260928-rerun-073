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


class CompensationError(TrainingStepError):
    """Raised when a compensation arrangement is inconsistent with the facts."""


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

    # ----------------------------------------------------- compensation
    @staticmethod
    def _comp_key(step_id: str | int, effect_name: str) -> str:
        return f"{TrainingStep._effect_key(step_id, effect_name)}:compensation"

    def _compensation_record(self, step_id: str | int, effect_name: str) -> dict[str, Any]:
        key = self._comp_key(step_id, effect_name)
        if key not in self._events:
            raise KeyError(
                f"no compensation arrangement for effect {effect_name!r} of step {step_id!r}; "
                "call decide_compensation first"
            )
        record = deepcopy(self._events[key]["payload"])
        record = self._normalize_compensation_record(record)
        return record

    @staticmethod
    def _normalize_compensation_record(record: dict[str, Any]) -> dict[str, Any]:
        record.setdefault("quantity_pending", 0.0)
        record.setdefault("total_quantity_approved", record.get("quantity_allowed", 0.0))
        record.setdefault("quantity_withdrawn", 0.0)
        record.setdefault("attempts", [])
        if "recipients" not in record:
            record["recipients"] = {
                "default": {
                    "quantity_allowed": float(record.get("quantity_allowed", 0.0)),
                    "quantity_compensated": float(record.get("quantity_compensated", 0.0)),
                    "quantity_pending": float(record.get("quantity_pending", 0.0)),
                    "quantity_withdrawn": 0.0,
                }
            }
        for recipient in record["recipients"].values():
            recipient.setdefault("quantity_allowed", 0.0)
            recipient.setdefault("quantity_compensated", 0.0)
            recipient.setdefault("quantity_pending", 0.0)
            recipient.setdefault("quantity_withdrawn", 0.0)
        record["quantity_allowed"] = sum(
            float(item["quantity_allowed"]) for item in record["recipients"].values()
        )
        record["quantity_compensated"] = sum(
            float(item["quantity_compensated"]) for item in record["recipients"].values()
        )
        record["quantity_pending"] = sum(
            float(item["quantity_pending"]) for item in record["recipients"].values()
        )
        record["total_quantity_allocated"] = float(record["quantity_allowed"])
        record["quantity_withdrawn"] = sum(
            float(item.get("quantity_withdrawn", 0.0)) for item in record["recipients"].values()
        )
        return record

    def _write_compensation(self, record: dict[str, Any]) -> dict[str, Any]:
        return self.append(
            self._comp_key(record["step"], record["name"]), record,
            kind="compensation", phase="training", overwrite=True,
        )

    def _append_compensation_audit(self, record: dict[str, Any], action: str, note: str, extra: dict[str, Any]) -> None:
        key = self._comp_key(record["step"], record["name"])
        sequence = [e for e in self._history if e["id"].startswith(f"{key}:{action}:")]
        payload = {
            "step": record["step"],
            "name": record["name"],
            "decision": record["decision"],
            "status": record["status"],
            "quantity_allowed": record["quantity_allowed"],
            "quantity_compensated": record["quantity_compensated"],
            "quantity_pending": record.get("quantity_pending", 0.0),
            "total_quantity_approved": record.get("total_quantity_approved", record["quantity_allowed"]),
            "total_quantity_allocated": record.get("total_quantity_allocated", record["quantity_allowed"]),
            "quantity_withdrawn": record.get("quantity_withdrawn", 0.0),
            "note": note,
        }
        payload.update(extra)
        self.append(
            f"{key}:{action}:{len(sequence)}", payload,
            kind=f"compensation_{action}", phase="training",
        )

    def decide_compensation(
        self,
        step_id: str | int,
        effect_name: str,
        decision: str,
        *,
        quantity: float | None = None,
        allocations: dict[str, float] | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        """Record the human arrangement for an external fact of a step.

        decision is one of "retain" (keep the registration as-is),
        "compensate" (apply an offsetting effect, up to quantity), or
        "pending" (still awaiting verification). The original effect
        record is never rewritten: an occurred effect stays occurred.
        """
        step_id = str(step_id)
        name = str(effect_name)
        if decision not in ("retain", "compensate", "pending"):
            raise ValueError(f"unknown compensation decision {decision!r}")
        effect_status = self._effect_status_with_inheritance(step_id, name)
        if effect_status is None:
            raise KeyError(f"effect {name!r} is not declared for step {step_id!r}")
        if decision != "pending" and effect_status != "occurred":
            raise UnconfirmedEffectError(
                f"effect {name!r} of step {step_id!r} is {effect_status}; "
                "only effects confirmed as occurred can be retained or compensated"
            )
        key = self._comp_key(step_id, name)
        if key in self._events:
            record = deepcopy(self._events[key]["payload"])
            record = self._normalize_compensation_record(record)
            executed = record["quantity_compensated"] > 0.0
            if executed and record["decision"] != decision:
                raise EventExistsError(
                    f"compensation for effect {name!r} already executed; the decision is locked"
                )
            record["decision"] = decision
            if quantity is not None:
                total = float(record.get("total_quantity_approved", record["quantity_allowed"]))
                if float(quantity) != total:
                    raise CompensationError(
                        "the total compensation approval cannot change after approval; "
                        "use reallocate_compensation to move unused recipient quota"
                    )
                recipient = record["recipients"]["default"]
                committed = recipient["quantity_compensated"] + recipient["quantity_pending"]
                if float(quantity) < committed:
                    raise CompensationError(
                        "allowed quantity cannot be below the already compensated "
                        "or still unconfirmed amount"
                    )
                record["total_quantity_approved"] = float(quantity)
                recipient["quantity_allowed"] = float(quantity)
        else:
            inherited = self._inherited_compensated_quantity(step_id, name)
            allowed = 1.0 if quantity is None else float(quantity)
            allocation_map = {"default": allowed}
            if allocations is not None:
                allocation_map = {str(recipient_id): float(value) for recipient_id, value in allocations.items()}
                if any(value < 0.0 for value in allocation_map.values()):
                    raise CompensationError("recipient allocations cannot be negative")
                allocation_total = sum(allocation_map.values())
                if quantity is None:
                    allowed = allocation_total
                elif abs(allocation_total - float(quantity)) > 1e-9:
                    raise CompensationError("recipient allocations must sum to the total approved quantity")
            record = {
                "step": step_id,
                "name": name,
                "target": TrainingStep._effect_key(step_id, name),
                "decision": decision,
                "status": "occurred" if inherited > 0 else "not_started",
                "quantity_allowed": allowed,
                "quantity_compensated": inherited,
                "quantity_pending": 0.0,
                "total_quantity_approved": allowed,
                "total_quantity_allocated": allowed,
                "quantity_withdrawn": 0.0,
                "recipients": {
                    recipient_id: {
                        "quantity_allowed": value,
                        "quantity_compensated": inherited if recipient_id == "default" else 0.0,
                        "quantity_pending": 0.0,
                        "quantity_withdrawn": 0.0,
                    }
                    for recipient_id, value in allocation_map.items()
                },
                "attempts": [],
                "payload": None,
                "reason": None,
            }
        self._write_compensation(record)
        self._append_compensation_audit(record, "decision", note, {})
        return deepcopy(record)

    def compensation(self, step_id: str | int, effect_name: str) -> dict[str, Any]:
        """Current compensation record: target, decision, scope and progress."""
        return self._compensation_record(step_id, effect_name)

    def _effect_status_with_inheritance(self, step_id: str | int, effect_name: str) -> str | None:
        step_id = str(step_id)
        name = str(effect_name)
        key = TrainingStep._effect_key(step_id, name)
        if key in self._events:
            return str(self._events[key]["payload"]["status"])
        for event in self._events.values():
            if event.get("kind") == "inherited_effect":
                payload = event["payload"]
                if payload.get("step") == step_id and payload.get("name") == name:
                    return str(payload.get("status"))
        return None

    def _inherited_compensated_quantity(self, step_id: str | int, effect_name: str) -> float:
        step_id = str(step_id)
        name = str(effect_name)
        total = 0.0
        for event in self._events.values():
            if event.get("kind") == "inherited_compensation":
                payload = event["payload"]
                if payload.get("step") == step_id and payload.get("name") == name:
                    total += float(payload.get("quantity_compensated", 0.0))
        return total

    def compensations(self, step_id: str | int) -> dict[str, dict[str, Any]]:
        prefix = f"{TrainingStep._step_key(step_id)}:effect:"
        out: dict[str, dict[str, Any]] = {}
        for key, event in self._events.items():
            if event.get("kind") == "compensation" and key.startswith(prefix):
                out[event["payload"]["name"]] = deepcopy(event["payload"])
        return out

    def run_compensation(
        self,
        step_id: str | int,
        effect_name: str,
        apply: Callable[[], Any] | None = None,
        replay: Callable[[Any], Any] | None = None,
        *,
        quantity: float | None = None,
        recipient_id: str = "default",
        attempt_id: str | None = None,
    ) -> Any:
        """Execute the compensating effect once, within the allowed scope.

        Idempotent: an already confirmed compensation is replayed, never
        re-applied; quantities are clamped to the remaining allowed scope so
        duplicate requests cannot double-offset the original fact.
        """
        step_id = str(step_id)
        name = str(effect_name)
        recipient_id = str(recipient_id)
        record = self._compensation_record(step_id, name)
        if record["decision"] != "compensate":
            raise CompensationError(
                f"effect {name!r} is decided as {record['decision']!r}, not 'compensate'"
            )
        effect_status = self._effect_status_with_inheritance(step_id, name)
        if effect_status != "occurred":
            raise UnconfirmedEffectError(
                f"effect {name!r} of step {step_id!r} is {effect_status}; without evidence "
                "that the original effect occurred no compensation may execute"
            )
        recipient = record["recipients"].get(recipient_id)
        if recipient is None:
            raise CompensationError(f"compensation recipient {recipient_id!r} has no allocation")
        if recipient.get("quantity_withdrawn", 0.0) > 0 and recipient["quantity_compensated"] + recipient["quantity_pending"] <= 0:
            raise CompensationError(f"recipient {recipient_id!r} allocation was withdrawn")
        remaining = recipient["quantity_allowed"] - recipient["quantity_compensated"] - recipient["quantity_pending"]
        recipient_has_pending = recipient["quantity_pending"] > 0.0
        if recipient_has_pending:
            raise UnconfirmedEffectError(
                f"recipient {recipient_id!r} has unconfirmed compensation; resolve its receipt before retrying"
            )
        if quantity is not None and float(quantity) > remaining:
            raise CompensationError(
                f"requested compensation quantity {float(quantity)} exceeds the remaining "
                f"approved balance {remaining} for effect {name!r}; the approval boundary "
                "cannot be met by trimming only the internal record"
            )
        if record["status"] == "occurred" and remaining <= 0:
            return replay(record["payload"]) if replay is not None else deepcopy(record["payload"])
        if remaining <= 0:
            if replay is not None and record["payload"] is not None:
                return replay(record["payload"])
            return deepcopy(record["payload"])
        if apply is None:
            raise ValueError("apply is required for a compensation that has not completed")
        qty = remaining if quantity is None else float(quantity)
        if qty <= 0:
            raise CompensationError("compensation quantity must be positive")
        attempt_id = str(attempt_id) if attempt_id is not None else f"{recipient_id}:{len(record['attempts']) + 1}"
        if any(item["attempt_id"] == attempt_id for item in record["attempts"]):
            previous = next(item for item in record["attempts"] if item["attempt_id"] == attempt_id)
            if previous["recipient_id"] != recipient_id or float(previous.get("quantity", 0.0)) != qty:
                raise EventExistsError("compensation attempt id was already used with different data")
            if previous["status"] == "occurred":
                return replay(previous["payload"]) if replay is not None else deepcopy(previous["payload"])
            raise UnconfirmedEffectError(f"compensation attempt {attempt_id!r} is still unconfirmed")
        try:
            payload = apply()
        except ExternalEffectUncertain as exc:
            recipient["quantity_pending"] += qty
            record = self._normalize_compensation_record(record)
            record["status"] = "unconfirmed"
            record["reason"] = str(exc)
            record["attempts"].append({
                "attempt_id": attempt_id,
                "recipient_id": recipient_id,
                "quantity": qty,
                "status": "unconfirmed",
                "payload": None,
                "reason": str(exc),
            })
            self._write_compensation(record)
            self._append_compensation_audit(
                record, "attempt", str(exc),
                {"quantity": qty, "outcome": "uncertain", "recipient_id": recipient_id, "attempt_id": attempt_id},
            )
            raise
        except ExternalEffectNotExecuted as exc:
            record = self._normalize_compensation_record(record)
            record["status"] = "unconfirmed" if record["quantity_pending"] > 0.0 else "not_occurred"
            record["reason"] = str(exc)
            record["attempts"].append({
                "attempt_id": attempt_id,
                "recipient_id": recipient_id,
                "quantity": qty,
                "status": "not_occurred",
                "payload": None,
                "reason": str(exc),
            })
            self._write_compensation(record)
            self._append_compensation_audit(
                record, "attempt", str(exc),
                {"quantity": qty, "outcome": "not_occurred", "recipient_id": recipient_id, "attempt_id": attempt_id},
            )
            raise
        except Exception as exc:
            recipient["quantity_pending"] += qty
            record = self._normalize_compensation_record(record)
            record["status"] = "unconfirmed"
            record["reason"] = repr(exc)
            record["attempts"].append({
                "attempt_id": attempt_id,
                "recipient_id": recipient_id,
                "quantity": qty,
                "status": "unconfirmed",
                "payload": None,
                "reason": repr(exc),
            })
            self._write_compensation(record)
            self._append_compensation_audit(
                record, "attempt", repr(exc),
                {"quantity": qty, "outcome": "uncertain", "recipient_id": recipient_id, "attempt_id": attempt_id},
            )
            raise
        recipient["quantity_compensated"] += qty
        record = self._normalize_compensation_record(record)
        record["status"] = "unconfirmed" if record["quantity_pending"] > 0.0 else "occurred"
        record["payload"] = payload if recipient_id == "default" else record.get("payload")
        record["reason"] = None
        record["attempts"].append({
            "attempt_id": attempt_id,
            "recipient_id": recipient_id,
            "quantity": qty,
            "status": "occurred",
            "payload": payload,
            "reason": None,
        })
        self._write_compensation(record)
        self._append_compensation_audit(
            record, "receipt", "",
            {"quantity": qty, "payload": deepcopy(payload), "recipient_id": recipient_id, "attempt_id": attempt_id},
        )
        return replay(payload) if replay is not None else deepcopy(payload)

    def verify_compensation(
        self,
        step_id: str | int,
        effect_name: str,
        occurred: bool,
        payload: Any = None,
        *,
        quantity: float | None = None,
        recipient_id: str = "default",
        attempt_id: str | None = None,
        receipt_id: str | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        """Resolve an unconfirmed compensation from a (possibly late) receipt.

        Confirming an already confirmed compensation only appends an audit
        record; the compensated quantity never increases twice.
        """
        step_id = str(step_id)
        name = str(effect_name)
        recipient_id = str(recipient_id)
        record = self._compensation_record(step_id, name)
        recipient = record["recipients"].get(recipient_id)
        if recipient is None:
            raise CompensationError(f"compensation recipient {recipient_id!r} has no allocation")
        verification_prefix = f"{self._comp_key(step_id, name)}:verification:"
        next_receipt_number = len([e for e in self._history if e["id"].startswith(verification_prefix)]) + 1
        receipt_id = str(receipt_id) if receipt_id is not None else f"receipt:{next_receipt_number}"
        known_receipt_attempts = [item for item in record["attempts"] if item.get("receipt_id") == receipt_id]
        if known_receipt_attempts:
            known_recipient = known_receipt_attempts[-1]["recipient_id"]
            if known_recipient != recipient_id:
                raise CompensationError(
                    f"receipt {receipt_id!r} belongs to {known_recipient!r}; it cannot be charged to {recipient_id!r}"
                )
            self._append_compensation_audit(
                record, "verification", note,
                {"duplicate": True, "recipient_id": recipient_id, "attempt_id": attempt_id, "receipt_id": receipt_id},
            )
            return deepcopy(record)
        if attempt_id is None and recipient["quantity_pending"] <= 0.0:
            if not occurred:
                raise EventExistsError(
                    f"compensation for recipient {recipient_id!r} was already confirmed; it cannot be revoked"
                )
            self._append_compensation_audit(
                record, "verification", note,
                {"duplicate": True, "recipient_id": recipient_id, "receipt_id": receipt_id},
            )
            return deepcopy(record)
        if attempt_id is None:
            pending_attempts = [
                item for item in record["attempts"]
                if item["recipient_id"] == recipient_id and item["status"] == "unconfirmed"
            ]
            if len(pending_attempts) != 1:
                raise CompensationError("attempt_id is required when zero or multiple attempts are pending")
            attempt = pending_attempts[0]
        else:
            attempt_id = str(attempt_id)
            matches = [item for item in record["attempts"] if item["attempt_id"] == attempt_id]
            if not matches:
                raise KeyError(f"compensation attempt {attempt_id!r} not found")
            attempt = matches[-1]
            if attempt["recipient_id"] != recipient_id:
                raise CompensationError(
                    f"attempt {attempt_id!r} belongs to {attempt['recipient_id']!r}, not {recipient_id!r}"
                )
        if attempt["status"] == "occurred":
            if not occurred:
                raise EventExistsError(
                    f"compensation attempt {attempt['attempt_id']!r} was already confirmed; it cannot be revoked"
                )
            self._append_compensation_audit(record, "verification", note, {"duplicate": True})
            return deepcopy(record)
        if attempt["status"] != "unconfirmed":
            raise UnconfirmedEffectError(
                f"compensation attempt {attempt['attempt_id']!r} is {attempt['status']}; nothing to verify"
            )
        pending = float(attempt.get("quantity_pending", attempt["quantity"]))
        if occurred:
            qty = pending if quantity is None else float(quantity)
            if qty <= 0.0 or qty > pending:
                raise CompensationError(
                    f"receipt confirms {qty} but only {pending} of attempt {attempt['attempt_id']!r} is pending"
                )
            if qty > recipient["quantity_allowed"] - recipient["quantity_compensated"]:
                raise CompensationError(
                    "confirmed quantity exceeds the remaining approved balance; "
                    "the approval boundary cannot be met by trimming only the internal record"
                )
            recipient["quantity_pending"] -= qty
            recipient["quantity_compensated"] += qty
            remainder = pending - qty
            attempt["quantity_pending"] = remainder
            attempt["status"] = "occurred" if remainder <= 0.0 else "unconfirmed"
            attempt["payload"] = payload
            attempt["reason"] = None
        else:
            recipient["quantity_pending"] -= pending
            attempt["quantity_pending"] = 0.0
            attempt["status"] = "not_occurred"
            attempt["payload"] = None
            qty = 0.0
        record = self._normalize_compensation_record(record)
        record["status"] = "unconfirmed" if record["quantity_pending"] > 0.0 else "occurred"
        record["reason"] = None if record["status"] == "occurred" else "one or more attempts are unconfirmed"
        attempt["receipt_id"] = receipt_id
        self._write_compensation(record)
        self._append_compensation_audit(
            record, "verification", note,
            {
                "occurred": bool(occurred),
                "quantity": qty,
                "recipient_id": recipient_id,
                "attempt_id": attempt["attempt_id"],
                "receipt_id": receipt_id,
            },
        )
        return deepcopy(record)

    def compensation_allowance(self, step_id: str | int, effect_name: str) -> dict[str, Any]:
        """Reconcile total approval against each recipient's separate quota."""
        record = self._compensation_record(step_id, effect_name)
        recipients = record["recipients"]
        recipient_report = {}
        for recipient_id, recipient in recipients.items():
            committed = recipient["quantity_compensated"] + recipient["quantity_pending"]
            recipient_report[recipient_id] = {
                "quantity_allowed": recipient["quantity_allowed"],
                "quantity_compensated": recipient["quantity_compensated"],
                "quantity_pending": recipient["quantity_pending"],
                "quantity_available": recipient["quantity_allowed"] - committed,
                "quantity_withdrawn": recipient.get("quantity_withdrawn", 0.0),
            }
        allocated = sum(item["quantity_allowed"] for item in recipients.values())
        withdrawn = sum(item.get("quantity_withdrawn", 0.0) for item in recipients.values())
        confirmed = sum(item["quantity_compensated"] for item in recipients.values())
        pending = sum(item["quantity_pending"] for item in recipients.values())
        return {
            "total_quantity_approved": record["total_quantity_approved"],
            "total_quantity_allocated": allocated,
            "total_quantity_withdrawn": withdrawn,
            "total_quantity_compensated": confirmed,
            "total_quantity_pending": pending,
            "total_quantity_available": allocated - confirmed - pending,
            "recipients": recipient_report,
        }

    def compensation_reallocation_options(
        self,
        step_id: str | int,
        effect_name: str,
        source_recipient_id: str,
        *,
        quantity: float | None = None,
        target_recipient_ids: list[str] | tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Return non-mutating retain / withdraw / partial-transfer options."""
        allowance = self.compensation_allowance(step_id, effect_name)
        source_recipient_id = str(source_recipient_id)
        source = allowance["recipients"].get(source_recipient_id)
        if source is None:
            raise CompensationError(f"source recipient {source_recipient_id!r} has no allocation")
        available = float(source["quantity_available"])
        qty = available if quantity is None else float(quantity)
        if qty <= 0.0 or qty > available:
            raise CompensationError(
                f"reallocation quantity {qty} must be within unused source quota {available}; "
                "confirmed and pending quota is not movable"
            )
        def projected(withdraw: bool = False, target: str | None = None) -> dict[str, Any]:
            recipients = deepcopy(allowance["recipients"])
            recipients[source_recipient_id]["quantity_allowed"] -= qty
            recipients[source_recipient_id]["quantity_available"] -= qty
            if withdraw:
                recipients[source_recipient_id]["quantity_withdrawn"] += qty
            else:
                target = str(target)
                recipients.setdefault(target, {
                    "quantity_allowed": 0.0,
                    "quantity_compensated": 0.0,
                    "quantity_pending": 0.0,
                    "quantity_available": 0.0,
                    "quantity_withdrawn": 0.0,
                })
                recipients[target]["quantity_allowed"] += qty
                recipients[target]["quantity_available"] += qty
            allocated = sum(item["quantity_allowed"] for item in recipients.values())
            withdrawn = sum(item["quantity_withdrawn"] for item in recipients.values())
            return {
                "quantity": qty,
                "total_quantity_allocated": allocated,
                "total_quantity_withdrawn": withdrawn,
                "recipients": recipients,
            }
        return {
            "retain": projected(target=source_recipient_id),
            "withdraw": projected(withdraw=True),
            "transfers": {
                str(target): projected(target=target)
                for target in target_recipient_ids
            },
        }

    def reallocate_compensation(
        self,
        step_id: str | int,
        effect_name: str,
        source_recipient_id: str,
        quantity: float,
        target_recipient_id: str | None = None,
        *,
        note: str = "",
    ) -> dict[str, Any]:
        """Move or withdraw only explicitly identified unused recipient quota."""
        step_id = str(step_id)
        name = str(effect_name)
        source_recipient_id = str(source_recipient_id)
        qty = float(quantity)
        record = self._compensation_record(step_id, name)
        source = record["recipients"].get(source_recipient_id)
        if source is None:
            raise CompensationError(f"source recipient {source_recipient_id!r} has no allocation")
        available = source["quantity_allowed"] - source["quantity_compensated"] - source["quantity_pending"]
        if qty <= 0.0 or qty > available:
            raise CompensationError(
                f"reallocation quantity {qty} must be within unused source quota {available}; "
                "confirmed and pending quota is not movable"
            )
        action = "withdraw" if target_recipient_id is None else "transfer"
        target_recipient_id = None if target_recipient_id is None else str(target_recipient_id)
        source["quantity_allowed"] -= qty
        if action == "withdraw":
            source["quantity_withdrawn"] = source.get("quantity_withdrawn", 0.0) + qty
        else:
            destination = record["recipients"].setdefault(target_recipient_id, {
                "quantity_allowed": 0.0,
                "quantity_compensated": 0.0,
                "quantity_pending": 0.0,
                "quantity_withdrawn": 0.0,
            })
            destination["quantity_allowed"] += qty
        record = self._normalize_compensation_record(record)
        self._write_compensation(record)
        self._append_compensation_audit(
            record, "reallocation", note,
            {
                "action": action,
                "quantity": qty,
                "source_recipient_id": source_recipient_id,
                "target_recipient_id": target_recipient_id,
            },
        )
        return deepcopy(record)

    # ----------------------------------------------------- resolution
    def effect_resolution(self, step_id: str | int, effect_name: str) -> dict[str, Any]:
        """Whether one effect is reconciled with the external facts."""
        step_id = str(step_id)
        name = str(effect_name)
        status = self.effect_statuses(step_id).get(name)
        if status is None:
            raise KeyError(f"effect {name!r} is not declared for step {step_id!r}")
        if status in ("not_occurred", "not_executed"):
            return {"effect": name, "status": status, "resolution": "resolved", "detail": status}
        if status == "unconfirmed":
            return {"effect": name, "status": status, "resolution": "conflict", "detail": "unconfirmed"}
        key = self._comp_key(step_id, name)
        record = self._normalize_compensation_record(deepcopy(self._events[key]["payload"])) if key in self._events else None
        if record is None or record["decision"] == "pending":
            return {"effect": name, "status": status, "resolution": "conflict", "detail": "awaiting_decision"}
        if record["decision"] == "retain":
            return {"effect": name, "status": status, "resolution": "resolved", "detail": "retained"}
        if record["status"] == "unconfirmed":
            detail = "compensation_unconfirmed"
        elif record["quantity_compensated"] >= record["quantity_allowed"]:
            detail = "compensated"
        elif record["quantity_compensated"] > 0:
            detail = "partially_compensated"
        else:
            detail = "compensation_pending"
        resolution = "resolved" if detail == "compensated" else "conflict"
        return {"effect": name, "status": status, "resolution": resolution, "detail": detail}

    def continuation(self, step_id: str | int) -> dict[str, Any]:
        """Which declared effects may proceed and which still conflict.

        Effects are mutually dependent in declaration order: an effect is
        continuable only when it and every earlier effect are resolved.
        """
        step_id = str(step_id)
        names = list(self.get(TrainingStep._step_key(step_id))["payload"]["effects"])
        effects: dict[str, dict[str, Any]] = {}
        continuable: list[str] = []
        conflicting: list[str] = []
        blocked = False
        for name in names:
            info = self.effect_resolution(step_id, name)
            ok = info["resolution"] == "resolved" and not blocked
            info["continuable"] = ok
            effects[name] = info
            if ok:
                continuable.append(name)
            else:
                conflicting.append(name)
                blocked = True
        return {
            "step": step_id,
            "effects": effects,
            "continuable": continuable,
            "conflicting": conflicting,
            "step_continuable": not conflicting,
        }

    # ----------------------------------------------------- branching
    def create_branch(
        self,
        branch_id: str,
        from_step: str | int,
        path: str | os.PathLike[str],
    ) -> "TrainingEventLog":
        """Fork a new training branch from a completed compute checkpoint.

        The new branch inherits the external facts that really happened
        (occurred effects and occurred compensations) plus the model and RNG
        state of the checkpoint. Compensation decisions that are not yet
        confirmed on this branch are not carried over.
        """
        branch_id = str(branch_id)
        record = self.get(TrainingStep._step_key(from_step))["payload"]
        if record["state"] != "complete":
            raise IncompleteTrainingStepError(
                f"step {from_step!r} is {record['state']}; branches start from confirmed checkpoints"
            )
        completion = record["completion"]
        branch = TrainingEventLog(path)
        if len(branch):
            raise EventExistsError(f"branch target {str(path)!r} is not an empty log")
        branch.append(
            f"branch:{branch_id}",
            {
                "branch": branch_id,
                "from_step": str(from_step),
                "source_log": str(self.path),
                "model": deepcopy(completion["model_after_optimizer"]),
                "rng_state": deepcopy(completion["rng_state"]),
                "gradient_digest": completion["gradient_digest"],
                "loss": completion["loss"],
            },
            kind="training_branch", phase="training",
        )
        for key, event in self._events.items():
            kind = event.get("kind")
            status = event["payload"].get("status") if isinstance(event.get("payload"), dict) else None
            if kind == "training_effect" and status == "occurred":
                branch.append(
                    f"branch:{branch_id}:fact:{key}", deepcopy(event["payload"]),
                    kind="inherited_effect", phase="training",
                )
            elif kind == "compensation" and status == "occurred":
                branch.append(
                    f"branch:{branch_id}:fact:{key}", deepcopy(event["payload"]),
                    kind="inherited_compensation", phase="training",
                )
        return branch

    def branch_record(self) -> dict[str, Any]:
        for event in self._events.values():
            if event.get("kind") == "training_branch":
                return deepcopy(event["payload"])
        raise KeyError("this log is not a branch created by create_branch")

    def inherited_effects(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            deepcopy(e["payload"]) for e in self._events.values()
            if e.get("kind") in ("inherited_effect", "inherited_compensation")
        )

    def restore_branch(self, model: Any, optimizer: Any | None = None) -> dict[str, Any]:
        """Load the branch checkpoint into model (and reset optimizer).

        Only parameter data, gradients and optimizer state are touched; no
        optimizer update is applied, so processing compensation on either
        branch can never replay a parameter update.
        """
        import numpy as np
        info = self.branch_record()
        params = list(model.parameters())
        snapshot = info["model"]["parameters"]
        if len(params) != len(snapshot):
            raise ValueError("model parameter count does not match the branch checkpoint")
        for param, values in zip(params, snapshot):
            param.data = np.asarray(values, dtype=float)
            param.grad = None
        if optimizer is not None:
            reset = getattr(optimizer, "reset_state", None)
            if reset is not None:
                reset()
        from .core import set_rng_state
        set_rng_state(info["rng_state"])
        return info

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
