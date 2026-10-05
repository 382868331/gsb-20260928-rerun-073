"""Durable and checkpoint-local event tracking."""
from __future__ import annotations

import json
import os
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from threading import RLock
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


class _CompensationReplay(Exception):
    """Internal: no allowance remains, replay the stored receipt."""

    def __init__(self, payload: Any, replay: Callable[[Any], Any] | None):
        self.payload = payload
        self.replay = replay


def _quantity_exceeds(requested: float, available: float) -> bool:
    """Strict comparison with tolerance for floating-point quantity ledgers."""
    return float(requested) > float(available) + 1e-9


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
        self._lock = RLock()
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
        with self._lock:
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
    DEFAULT_COMPENSATION_TARGET = "default"

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
        return self._normalize_compensation_record(deepcopy(self._events[key]["payload"]))

    def _normalize_compensation_record(self, record: dict[str, Any]) -> dict[str, Any]:
        record.setdefault("revision", 0)
        record.setdefault("quantity_pending", 0.0)
        allocations = record.get("allocations")
        if allocations is None:
            # records written before multiple payees existed: their single
            # scope becomes the default payee so every quantity is preserved
            allocation = {
                "target": self.DEFAULT_COMPENSATION_TARGET,
                "quantity_allowed": float(record.get("quantity_allowed", 0.0)),
                "quantity_compensated": float(record.get("quantity_compensated", 0.0)),
                "quantity_pending": float(record.get("quantity_pending", 0.0)),
                "status": record.get("status", "not_started"),
                "payload": record.get("payload"),
                "reason": record.get("reason"),
                "attempts": [],
            }
            legacy_allocations = [allocation] if (
                allocation["quantity_allowed"] > 0
                or allocation["quantity_compensated"] > 0
                or allocation["quantity_pending"] > 0
            ) else []
            record["allocations"] = legacy_allocations
        for allocation in record.get("allocations", []):
            attempts = allocation.setdefault("attempts", [])
            if allocation.get("quantity_pending", 0.0) and not attempts:
                # record predates per-attempt reservations
                attempts.append({
                    "id": f"legacy:{allocation['target']}",
                    "quantity": float(allocation["quantity_pending"]),
                    "status": "unconfirmed",
                    "payload": None,
                })
        self._refresh_compensation_totals(record)
        return record

    @staticmethod
    def _allocation_index(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {str(a["target"]): a for a in record.get("allocations", [])}

    def _refresh_compensation_totals(self, record: dict[str, Any]) -> dict[str, Any]:
        allocations = record.setdefault("allocations", [])
        compensated = sum(float(a.get("quantity_compensated", 0.0)) for a in allocations)
        pending = sum(
            float(t.get("quantity", 0.0))
            for a in allocations for t in a.get("attempts", [])
            if t.get("status") == "unconfirmed"
        )
        allocated = sum(float(a.get("quantity_allowed", 0.0)) for a in allocations)
        allowed = float(record.get("quantity_allowed", allocated))
        if _quantity_exceeds(allocated, allowed):
            # the per-payee view must never promise more than the total approval
            raise CompensationError(
                f"allocated quantity {allocated} exceeds the approved quantity {allowed} "
                f"for effect {record.get('name')!r}"
            )
        record["quantity_allowed"] = allowed
        record["quantity_compensated"] = compensated
        record["quantity_pending"] = pending
        record["quantity_allocated"] = allocated
        record["quantity_unallocated"] = allowed - allocated
        statuses: set[str] = set()
        for allocation in allocations:
            attempt_statuses = {str(t.get("status")) for t in allocation.get("attempts", [])}
            if "unconfirmed" in attempt_statuses:
                allocation["status"] = "unconfirmed"
            elif "not_occurred" in attempt_statuses and "occurred" not in attempt_statuses:
                allocation["status"] = "not_occurred"
            elif float(allocation.get("quantity_compensated", 0.0)) > 0:
                allocation["status"] = "occurred"
            else:
                allocation["status"] = "not_started"
            statuses.add(allocation["status"])
            allocation["quantity_pending"] = sum(
                float(t.get("quantity", 0.0)) for t in allocation.get("attempts", [])
                if t.get("status") == "unconfirmed"
            )
        if "unconfirmed" in statuses:
            status = "unconfirmed"
        elif compensated > 0:
            status = "occurred"
        elif statuses == {"not_occurred"}:
            status = "not_occurred"
        elif statuses:
            status = "not_occurred" if statuses <= {"not_occurred", "not_started"} and "not_occurred" in statuses else "not_started"
        else:
            status = "not_started"
        record["status"] = status
        for allocation in allocations:
            allocation["quantity_remaining"] = (
                float(allocation["quantity_allowed"])
                - float(allocation.get("quantity_compensated", 0.0))
                - float(allocation.get("quantity_pending", 0.0))
            )
        return record

    def _write_compensation(self, record: dict[str, Any]) -> dict[str, Any]:
        return self._write_compensation_cas(record, None)

    def _write_compensation_cas(self, record: dict[str, Any], expected_revision: int | None) -> dict[str, Any]:
        """Persist the ledger state, refusing lost updates from parallel workers."""
        key = self._comp_key(record["step"], record["name"])
        current = self._events.get(key)
        current_revision = int(self._normalize_compensation_record(deepcopy(current["payload"]))["revision"]) if current else 0
        if expected_revision is not None:
            if current_revision != expected_revision:
                raise CompensationError(
                    f"compensation ledger for effect {record['name']!r} changed concurrently; "
                    "re-read the payee allowance and retry the settlement"
                )
        self._refresh_compensation_totals(record)
        record["revision"] = current_revision + 1
        written = self.append(
            key, record, kind="compensation", phase="training", overwrite=True,
        )
        return written

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
            "quantity_allocated": record.get("quantity_allocated", 0.0),
            "quantity_unallocated": record.get("quantity_unallocated", 0.0),
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
            record = self._compensation_record(step_id, name)
            if record["status"] == "occurred" and record["decision"] != decision:
                raise EventExistsError(
                    f"compensation for effect {name!r} already executed; the decision is locked"
                )
            record["decision"] = decision
            if quantity is not None:
                committed = record["quantity_compensated"] + record["quantity_pending"]
                if float(quantity) + 1e-9 < committed:
                    raise CompensationError(
                        "allowed quantity cannot be below the already compensated "
                        "or still unconfirmed amount"
                    )
                record["quantity_allowed"] = float(quantity)
                # already granted per-payee allowances are commitments; only
                # the still-unallocated part may shrink without an explicit
                # withdrawal/reallocation
                if _quantity_exceeds(record["quantity_allocated"], float(quantity)):
                    raise CompensationError(
                        "allowed quantity cannot be below the amount already "
                        "allocated to independent payees; withdraw or transfer "
                        "those allowances explicitly first"
                    )
        else:
            inherited_by_target = self._inherited_compensation_by_target(step_id, name)
            inherited = sum(inherited_by_target.values())
            allowed = 1.0 if quantity is None else float(quantity)
            allocations: list[dict[str, Any]] = []
            for inherited_target, inherited_quantity in inherited_by_target.items():
                allocation = self._new_allocation(inherited_target, inherited_quantity)
                allocation["quantity_compensated"] = inherited_quantity
                allocation["quantity_remaining"] = 0.0
                allocation["status"] = "occurred"
                allocations.append(allocation)
            # the inherited payee's allowance is topped up to the approved
            # scope, so the same payee can receive the remaining balance
            if allocations and len(allocations) == 1:
                allocations[0]["quantity_allowed"] = allowed
                allocated_total = allowed
            else:
                allocated_total = inherited
            record = {
                "step": step_id,
                "name": name,
                "target": TrainingStep._effect_key(step_id, name),
                "decision": decision,
                "status": "occurred" if inherited > 0 else "not_started",
                "quantity_allowed": allowed,
                "quantity_compensated": inherited,
                "quantity_pending": 0.0,
                "quantity_allocated": allocated_total,
                "quantity_unallocated": allowed - allocated_total,
                "allocations": allocations,
                "payload": None,
                "reason": None,
            }
        self._write_compensation(record)
        self._append_compensation_audit(record, "decision", note, {})
        return deepcopy(record)

    def _new_allocation(self, target: str, quantity: float) -> dict[str, Any]:
        return {
            "target": str(target),
            "quantity_allowed": float(quantity),
            "quantity_compensated": 0.0,
            "quantity_pending": 0.0,
            "quantity_remaining": float(quantity),
            "status": "not_started",
            "payload": None,
            "reason": None,
            "attempts": [],
        }

    def allocate_compensation(
        self,
        step_id: str | int,
        effect_name: str,
        target: str,
        quantity: float,
        *,
        note: str = "",
    ) -> dict[str, Any]:
        """Grant part of the total approval to one independent external payee.

        Only the still-unallocated part of the approval can be granted.
        Granting more to an existing payee tops up its allowance; it never
        creates a second quota for the same payee.
        """
        target = str(target)
        quantity = float(quantity)
        if quantity < 0:
            raise CompensationError("allocated quantity must be non-negative")
        with self._lock:
            record = self._compensation_record(step_id, effect_name)
            if record["decision"] != "compensate":
                raise CompensationError(
                    f"effect {effect_name!r} is decided as {record['decision']!r}; "
                    "only compensating effects can allocate payee allowances"
                )
            allocations = self._allocation_index(record)
            if target not in allocations and _quantity_exceeds(quantity, record["quantity_unallocated"]):
                raise CompensationError(
                    f"cannot allocate {quantity} to payee {target!r}: only "
                    f"{record['quantity_unallocated']} of the total approval is unallocated"
                )
            if target in allocations and _quantity_exceeds(quantity, record["quantity_unallocated"]):
                raise CompensationError(
                    f"cannot top up payee {target!r} by {quantity}: only "
                    f"{record['quantity_unallocated']} of the total approval is unallocated"
                )
            if target in allocations:
                allocations[target]["quantity_allowed"] += quantity
            elif quantity > 0:
                record["allocations"].append(self._new_allocation(target, quantity))
            self._write_compensation(record)
            self._append_compensation_audit(
                record, "allocation", note, {"target": target, "quantity": quantity}
            )
            return deepcopy(record)

    def _releasable_allowance(self, allocation: dict[str, Any]) -> float:
        return (
            float(allocation["quantity_allowed"])
            - float(allocation["quantity_compensated"])
            - float(allocation["quantity_pending"])
        )

    def withdraw_compensation_allocation(
        self,
        step_id: str | int,
        effect_name: str,
        target: str,
        quantity: float,
        *,
        note: str = "",
    ) -> dict[str, Any]:
        """Pull an unexecuted payee allowance back to the unallocated pool.

        Confirmed and still-unconfirmed (in-flight) quantities can never be
        withdrawn: a sent-but-unacknowledged action keeps reserving the
        original payee's quota until its late receipt is resolved.
        """
        target = str(target)
        quantity = float(quantity)
        if quantity < 0:
            raise CompensationError("withdrawn quantity must be non-negative")
        with self._lock:
            record = self._compensation_record(step_id, effect_name)
            allocation = self._allocation_index(record).get(target)
            if allocation is None:
                raise KeyError(f"payee {target!r} has no allocation for effect {effect_name!r}")
            available = self._releasable_allowance(allocation)
            if _quantity_exceeds(quantity, available):
                raise CompensationError(
                    f"cannot withdraw {quantity} from payee {target!r}: only {available} is "
                    "unexecuted; confirmed and in-flight quantities stay reserved"
                )
            allocation["quantity_allowed"] -= quantity
            self._write_compensation(record)
            self._append_compensation_audit(
                record, "withdrawal", note, {"target": target, "quantity": quantity}
            )
            return deepcopy(record)

    def reallocate_compensation(
        self,
        step_id: str | int,
        effect_name: str,
        source_target: str,
        destination_target: str,
        quantity: float,
        *,
        note: str = "",
    ) -> dict[str, Any]:
        """Move an unexecuted allowance from one payee to another payee.

        The transfer is an explicit allocation change. In-flight, still
        unconfirmed quantities are not movable, so a late receipt for the
        original payee can never be charged to the new owner.
        """
        source_target = str(source_target)
        destination_target = str(destination_target)
        quantity = float(quantity)
        if quantity < 0:
            raise CompensationError("transferred quantity must be non-negative")
        if source_target == destination_target:
            raise CompensationError("source and destination payees must differ")
        with self._lock:
            record = self._compensation_record(step_id, effect_name)
            allocations = self._allocation_index(record)
            source = allocations.get(source_target)
            if source is None:
                raise KeyError(
                    f"payee {source_target!r} has no allocation for effect {effect_name!r}"
                )
            available = self._releasable_allowance(source)
            if _quantity_exceeds(quantity, available):
                raise CompensationError(
                    f"cannot transfer {quantity} from payee {source_target!r}: only {available} "
                    "is unexecuted; in-flight quantities stay reserved against the original payee"
                )
            source["quantity_allowed"] -= quantity
            if destination_target in allocations:
                allocations[destination_target]["quantity_allowed"] += quantity
            else:
                record["allocations"].append(self._new_allocation(destination_target, quantity))
            self._write_compensation(record)
            self._append_compensation_audit(
                record, "reallocation", note,
                {"source": source_target, "destination": destination_target, "quantity": quantity},
            )
            return deepcopy(record)

    def compensation_reallocation_options(
        self,
        step_id: str | int,
        effect_name: str,
        source_target: str,
    ) -> dict[str, Any]:
        """Compare keep / withdraw / transfer plans without any external action.

        Every figure describes a hypothetical plan; calling this method does
        not change any allowance and sends nothing to an external system.
        """
        source_target = str(source_target)
        with self._lock:
            record = self._compensation_record(step_id, effect_name)
            allocations = self._allocation_index(record)
            source = allocations.get(source_target)
            if source is None:
                raise KeyError(
                    f"payee {source_target!r} has no allocation for effect {effect_name!r}"
                )
            releasable = self._releasable_allowance(source)
            plans: list[dict[str, Any]] = []
            for destination, allocation in allocations.items():
                if destination == source_target:
                    continue
                plan = {
                    "target": destination,
                    "quantity_transferable": releasable,
                    source_target: source["quantity_allowed"] - releasable,
                    destination: allocation["quantity_allowed"] + releasable,
                }
                plans.append(plan)
            return {
                "step": str(step_id),
                "name": str(effect_name),
                "keep": {"target": source_target, "quantity_available": releasable},
                "withdraw": {
                    "target": source_target,
                    "quantity_released": releasable,
                    "quantity_unallocated_after": record["quantity_unallocated"] + releasable,
                },
                "transfer": plans,
            }

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

    def _inherited_compensation_by_target(self, step_id: str | int, effect_name: str) -> dict[str, float]:
        step_id = str(step_id)
        name = str(effect_name)
        totals: dict[str, float] = {}
        for event in self._events.values():
            if event.get("kind") == "inherited_compensation":
                payload = event["payload"]
                if payload.get("step") == step_id and payload.get("name") == name:
                    target = str(payload.get("target", self.DEFAULT_COMPENSATION_TARGET))
                    totals[target] = totals.get(target, 0.0) + float(payload.get("quantity_compensated", 0.0))
        return totals

    def _inherited_compensated_quantity(self, step_id: str | int, effect_name: str) -> float:
        return sum(self._inherited_compensation_by_target(step_id, effect_name).values())

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
        target: str | None = None,
        replay_only: bool = False,
    ) -> Any:
        """Execute the compensating effect once, within the allowed scope.

        The request is charged against one independent payee's allowance.
        Idempotent: an already completed payee request is replayed, never
        re-applied; other payees' unused allowances are never touched.
        """
        step_id = str(step_id)
        name = str(effect_name)
        target_name = self.DEFAULT_COMPENSATION_TARGET if target is None else str(target)
        try:
            attempt_id, qty = self._reserve_compensation(
                step_id, name, target_name, None if replay_only else quantity,
                apply is None, replay,
            )
        except _CompensationReplay as done:
            return done.replay(done.payload) if done.replay is not None else deepcopy(done.payload)
        try:
            payload = apply()
        except ExternalEffectUncertain as exc:
            self._settle_attempt(step_id, name, target_name, attempt_id, qty, "uncertain", reason=str(exc))
            raise
        except ExternalEffectNotExecuted as exc:
            self._settle_attempt(step_id, name, target_name, attempt_id, qty, "not_occurred", reason=str(exc))
            raise
        except Exception as exc:
            self._settle_attempt(step_id, name, target_name, attempt_id, qty, "uncertain", reason=repr(exc))
            raise
        self._settle_attempt(step_id, name, target_name, attempt_id, qty, "occurred", payload=payload)
        return replay(payload) if replay is not None else deepcopy(payload)

    def _reserve_compensation(
        self,
        step_id: str,
        name: str,
        target_name: str,
        quantity: float | None,
        apply_missing: bool,
        replay: Callable[[Any], Any] | None,
    ) -> tuple[str, float]:
        import time
        for wait_attempt in range(200):
            wait = False
            with self._lock:
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
                allocations = self._allocation_index(record)
                allocation = allocations.get(target_name)
                if allocation is None and target_name == self.DEFAULT_COMPENSATION_TARGET and not record["allocations"]:
                    allocation = self._new_allocation(target_name, record["quantity_allowed"])
                    record["allocations"].append(allocation)
                if allocation is None:
                    raise CompensationError(
                        f"payee {target_name!r} has no allowance for effect {name!r}; "
                        "call allocate_compensation first"
                    )
                remaining = self._releasable_allowance(allocation)
                qty = remaining if quantity is None else float(quantity)
                in_flight = any(t["status"] == "unconfirmed" for t in allocation.get("attempts", []))
                if remaining <= 0 and in_flight and quantity is not None and wait_attempt < 199:
                    # peers are applying externally: wait for their receipts,
                    # then re-read instead of assuming success or failure
                    wait = True
                elif remaining <= 0 and in_flight:
                    raise UnconfirmedEffectError(
                        f"compensation for payee {target_name!r} is unconfirmed and no balance "
                        "remains unreserved; settle the late receipt first"
                    )
                elif remaining <= 0 and quantity is not None and float(quantity) > 0:
                    raise CompensationError(
                        f"payee {target_name!r} has no remaining approved balance; an explicit "
                        "new request cannot be satisfied, use a replay call"
                    )
                elif remaining <= 0:
                    raise _CompensationReplay(allocation["payload"], replay)
                else:
                    if apply_missing:
                        raise ValueError("apply is required for a compensation that has not completed")
                    if qty <= 0:
                        raise CompensationError("compensation quantity must be positive")
                    if in_flight and _quantity_exceeds(qty, remaining):
                        raise UnconfirmedEffectError(
                            f"compensation for payee {target_name!r} has an in-flight attempt; "
                            "call verify_compensation with the late receipt before requesting "
                            "more than the still-unreserved balance"
                        )
                    if not in_flight and _quantity_exceeds(qty, remaining):
                        raise CompensationError(
                            f"requested compensation quantity {float(quantity)} exceeds the remaining "
                            f"approved balance {remaining} for payee {target_name!r}; other payees' "
                            "allowances cannot be spent without an explicit reallocation"
                        )
                    attempt_id = (
                        f"{step_id}:{name}:{target_name}:attempt:{len(allocation['attempts'])}"
                    )
                    allocation["attempts"].append(
                        {"id": attempt_id, "quantity": qty, "status": "unconfirmed", "payload": None}
                    )
                    revision = int(record.get("revision", 0))
                    try:
                        self._write_compensation_cas(record, revision)
                    except CompensationError:
                        continue
                    self.append(
                        f"compensation-reservation:{attempt_id}",
                        {"step": step_id, "name": name, "target": target_name,
                         "quantity": qty, "attempt_id": attempt_id},
                        kind="compensation_reservation", phase="training",
                    )
                    return attempt_id, qty
            if wait:
                time.sleep(0.01)
        raise UnconfirmedEffectError(
            f"timed out waiting for in-flight compensation of payee {target_name!r}"
        )

    def _settle_attempt(
        self,
        step_id: str,
        name: str,
        target_name: str,
        attempt_id: str,
        qty: float,
        outcome: str,
        *,
        payload: Any = None,
        reason: str = "",
    ) -> dict[str, Any]:
        while True:
            with self._lock:
                record = self._compensation_record(step_id, name)
                allocation = self._allocation_index(record)[target_name]
                attempts = allocation["attempts"]
                attempt = next((t for t in attempts if t["id"] == attempt_id), None)
                if attempt is None or attempt["status"] != "unconfirmed":
                    return deepcopy(record)
                revision = int(record.get("revision", 0))
                if outcome == "occurred":
                    attempt["status"] = "occurred"
                    attempt["payload"] = payload
                    allocation["quantity_compensated"] += qty
                    allocation["payload"] = payload
                    allocation["reason"] = None
                elif outcome == "not_occurred":
                    attempt["status"] = "not_occurred"
                    attempt["payload"] = None
                    allocation["reason"] = reason or None
                else:
                    allocation["reason"] = reason or None
                try:
                    self._write_compensation_cas(record, revision)
                except CompensationError:
                    continue
                audit_outcome = outcome if outcome != "occurred" else "receipt"
                self._append_compensation_audit(
                    record, "attempt" if outcome != "occurred" else "receipt", reason,
                    {"target": target_name, "quantity": qty, "outcome": outcome,
                     "attempt_id": attempt_id, "payload": deepcopy(payload)},
                )
                return deepcopy(record)

    def verify_compensation(
        self,
        step_id: str | int,
        effect_name: str,
        occurred: bool,
        payload: Any = None,
        *,
        quantity: float | None = None,
        target: str | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        """Resolve an unconfirmed compensation from a (possibly late) receipt.

        The receipt is always settled against the original payee's quota. A
        duplicate or late receipt only appends an audit record; it is never
        charged against a payee that received a reallocated allowance.
        """
        step_id = str(step_id)
        name = str(effect_name)
        target_name = self.DEFAULT_COMPENSATION_TARGET if target is None else str(target)
        with self._lock:
            record = self._compensation_record(step_id, name)
            allocations = self._allocation_index(record)
            if target_name not in allocations:
                raise KeyError(
                    f"payee {target_name!r} has no allocation for effect {name!r}"
                )
            allocation = allocations[target_name]
            if allocation["status"] == "occurred" and allocation["quantity_pending"] <= 0:
                if not occurred:
                    raise EventExistsError(
                        f"compensation for payee {target_name!r} was already confirmed; "
                        "it cannot be revoked"
                    )
                self._append_compensation_audit(
                    record, "verification", note,
                    {"target": target_name, "duplicate": True},
                )
                return deepcopy(record)
            if allocation["status"] != "unconfirmed":
                raise UnconfirmedEffectError(
                    f"compensation for payee {target_name!r} is {allocation['status']}; "
                    "nothing to verify"
                )
            pending_attempts = [t for t in allocation["attempts"] if t["status"] == "unconfirmed"]
            attempt = pending_attempts[-1]
            attempt_quantity = float(attempt["quantity"])
            qty = 0.0
            if occurred:
                qty = attempt_quantity if quantity is None else float(quantity)
                if _quantity_exceeds(qty, attempt_quantity):
                    raise CompensationError(
                        f"receipt confirms {qty} but the in-flight attempt for payee "
                        f"{target_name!r} only reserved {attempt_quantity}; confirming more "
                        "than was attempted would "
                        "consume the allowance twice"
                    )
                if _quantity_exceeds(qty, allocation["quantity_allowed"] - allocation["quantity_compensated"]):
                    raise CompensationError(
                        "confirmed quantity exceeds the payee's remaining allowance; the "
                        "approval boundary cannot be met by trimming only the internal record"
                    )
                allocation["quantity_compensated"] += qty
                if qty + 1e-9 >= attempt_quantity:
                    attempt["status"] = "occurred"
                    attempt["payload"] = payload
                else:
                    # partial confirmation: split the reservation so the
                    # confirmed slice and the still-open slice never overlap
                    attempt["quantity"] = attempt_quantity - qty
                    allocation["attempts"].append(
                        {"id": f"{attempt['id']}:confirmed", "quantity": qty,
                         "status": "occurred", "payload": payload}
                    )
                allocation["payload"] = payload
                allocation["reason"] = None
            else:
                attempt["status"] = "not_occurred"
                attempt["payload"] = None
                allocation["reason"] = note or None
            self._write_compensation(record)
            self._append_compensation_audit(
                record, "verification", note,
                {"target": target_name, "occurred": bool(occurred), "quantity": qty if occurred else 0.0},
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
        record = deepcopy(self._events[key]["payload"]) if key in self._events else None
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
            elif kind == "compensation":
                payload = self._normalize_compensation_record(deepcopy(event["payload"]))
                for index, allocation in enumerate(payload.get("allocations", [])):
                    confirmed = float(allocation.get("quantity_compensated", 0.0))
                    if confirmed <= 0:
                        # unconfirmed or not-yet-executed payee actions are
                        # decisions on this branch, not inherited facts
                        continue
                    fact = {
                        "step": payload["step"],
                        "name": payload["name"],
                        "target": allocation["target"],
                        "decision": payload["decision"],
                        "quantity_allowed": confirmed,
                        "quantity_compensated": confirmed,
                        "quantity_pending": 0.0,
                        "status": "occurred",
                        "payload": allocation.get("payload"),
                        "reason": None,
                    }
                    branch.append(
                        f"branch:{branch_id}:fact:{key}:{allocation['target']}:{index}", fact,
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
