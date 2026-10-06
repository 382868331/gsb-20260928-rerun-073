"""Batched adjustment of compensation allowance across recipients.

Covers:
* whole-plan validation: cyclic transfers and several transfers sharing one
  source are checked against net per-recipient unused quota, not item-by-item;
* batched execution with partial receipts: each adjustment keeps its actual
  effect and its still-unconfirmed in-transit range;
* late corrections as independent reconciliation decisions that never rewrite
  the original approved quantity; duplicate receipts never double-apply;
* cancellation of unsent adjustments while sent ones await their receipt;
* persistence across reopen and total-quota conservation through duplicate
  receipts, failure branches and cross-recipient partial success.
"""
import pytest

import autodiff as ad
from autodiff.events import CompensationError

from .test_compensation import run_completed_step


def make_log(path, allocations=None, quantity=10.0):
    log = ad.TrainingEventLog(path)
    run_completed_step(log, 0, effects=["register"])
    log.decide_compensation(
        0, "register", "compensate", quantity=quantity, allocations=allocations,
    )
    return log


def conservation(log, step=0, name="register"):
    allowance = log.compensation_allowance(step, name)
    total = (
        allowance["total_quantity_allocated"]
        + allowance["total_quantity_withdrawn"]
        + allowance["total_quantity_in_transit"]
    )
    assert total == pytest.approx(allowance["total_quantity_approved"])
    return allowance


# ------------------------------------------------------- whole-plan validation
def test_plan_rejects_shared_quota_overcommit_but_allows_cycle(tmp_path):
    log = make_log(tmp_path / "log.jsonl", {"a": 6.0, "b": 4.0})
    # two transfers sharing the same source quota: each looks legal alone
    with pytest.raises(CompensationError, match="as a whole"):
        log.plan_compensation_adjustment(0, "register", [
            {"source": "a", "target": "b", "quantity": 4.0},
            {"source": "a", "target": "c", "quantity": 4.0},
        ])
    # nothing was committed by the rejected plan
    assert log.compensation_adjustment_plans(0, "register") == {}
    # a cycle nets to zero per recipient and is legal as a whole
    plan = log.plan_compensation_adjustment(0, "register", [
        {"source": "a", "target": "b", "quantity": 3.0, "adjustment_id": "a-to-b"},
        {"source": "b", "target": "a", "quantity": 3.0, "adjustment_id": "b-to-a"},
    ], plan_id="cycle")
    assert plan["status"] == "planned"
    assert conservation(log)["total_quantity_in_transit"] == 0.0


def test_plan_rejects_unknown_recipient_and_bad_quantities(tmp_path):
    log = make_log(tmp_path / "log.jsonl", {"a": 6.0, "b": 4.0})
    with pytest.raises(CompensationError):
        log.plan_compensation_adjustment(0, "register", [
            {"source": "a", "target": "ghost", "quantity": 1.0},
            {"source": "ghost", "target": "a", "quantity": 1.0},
        ])
    with pytest.raises(CompensationError):
        log.plan_compensation_adjustment(0, "register", [
            {"source": "a", "target": "b", "quantity": 0.0},
        ])
    with pytest.raises(CompensationError):
        log.plan_compensation_adjustment(0, "register", [
            {"source": "a", "target": "a", "quantity": 1.0},
        ])
    with pytest.raises(CompensationError):
        log.plan_compensation_adjustment(0, "register", [])
    with pytest.raises(KeyError):
        log.plan_compensation_adjustment(0, "undeclared", [
            {"source": "a", "target": "b", "quantity": 1.0},
        ])


def test_pending_compensation_quota_is_not_treated_as_spare(tmp_path):
    log = make_log(tmp_path / "log.jsonl", {"a": 6.0, "b": 4.0})
    with pytest.raises(ad.ExternalEffectUncertain):
        log.run_compensation(
            0, "register",
            lambda: (_ for _ in ()).throw(ad.ExternalEffectUncertain("lost")),
            quantity=5.0, recipient_id="a", attempt_id="a-pending",
        )
    # 5 of a's 6 are unconfirmed: only 1 is truly unused and movable
    with pytest.raises(CompensationError):
        log.plan_compensation_adjustment(0, "register", [
            {"source": "a", "target": "b", "quantity": 2.0},
        ])
    plan = log.plan_compensation_adjustment(0, "register", [
        {"source": "a", "target": "b", "quantity": 1.0},
    ])
    assert plan["status"] == "planned"


# ------------------------------------------------------- batched execution
def test_partial_receipts_keep_effect_and_unconfirmed_range(tmp_path):
    path = tmp_path / "log.jsonl"
    log = make_log(path, {"a": 6.0, "b": 4.0})
    log.plan_compensation_adjustment(0, "register", [
        {"source": "a", "target": "b", "quantity": 4.0, "adjustment_id": "move-4"},
        {"source": "b", "target": None, "quantity": 2.0, "adjustment_id": "drop-2"},
    ], plan_id="batch")
    log.run_compensation_adjustment(0, "register", "batch", "move-4")
    log.run_compensation_adjustment(0, "register", "batch", "drop-2")
    # sending twice is rejected while the receipt is outstanding
    with pytest.raises(ad.UnconfirmedEffectError):
        log.run_compensation_adjustment(0, "register", "batch", "move-4")
    allowance = conservation(log)
    assert allowance["total_quantity_in_transit"] == 6.0
    assert allowance["total_quantity_allocated"] == 4.0
    # only part of the first receipt arrives: the rest stays in transit
    plan = log.verify_compensation_adjustment(
        0, "register", "batch", "move-4", True,
        quantity=1.5, receipt_id="r-1", note="partial confirmation",
    )
    item = next(i for i in plan["adjustments"] if i["adjustment_id"] == "move-4")
    assert item["status"] == "sent"
    assert item["confirmed_quantity"] == 1.5
    assert item["in_transit"] == 2.5
    allowance = conservation(log)
    assert allowance["recipients"]["b"]["quantity_allowed"] == 3.5
    assert allowance["total_quantity_in_transit"] == 4.5
    # the remainder arrives late as an independent reconciliation decision
    log.verify_compensation_adjustment(
        0, "register", "batch", "move-4", True,
        receipt_id="r-2", note="late remainder",
    )
    plan = log.compensation_adjustment_plan(0, "register", "batch")
    item = next(i for i in plan["adjustments"] if i["adjustment_id"] == "move-4")
    assert item["status"] == "confirmed"
    assert conservation(log)["total_quantity_in_transit"] == 2.0
    # the original approval was never rewritten
    assert log.compensation(0, "register")["total_quantity_approved"] == 10.0


def test_late_correction_is_independent_and_never_rewrites_approval(tmp_path):
    log = make_log(tmp_path / "log.jsonl", {"a": 6.0, "b": 4.0})
    log.plan_compensation_adjustment(0, "register", [
        {"source": "a", "target": "b", "quantity": 3.0, "adjustment_id": "move"},
    ], plan_id="p")
    log.run_compensation_adjustment(0, "register", "p", "move")
    log.verify_compensation_adjustment(
        0, "register", "p", "move", True, quantity=1.0, receipt_id="first",
    )
    # a late correction for the same in-transit range is a new decision
    log.verify_compensation_adjustment(
        0, "register", "p", "move", True, quantity=2.0, receipt_id="correction",
        note="late correction of the remaining range",
    )
    record = log.compensation(0, "register")
    assert record["total_quantity_approved"] == 10.0
    assert log.compensation_allowance(0, "register")["recipients"]["b"]["quantity_allowed"] == 7.0
    audits = [
        e for e in log.audit_events(0)
        if e["kind"] == "compensation_adjustment_verify"
    ]
    assert [e["payload"]["receipt_id"] for e in audits] == ["first", "correction"]
    assert audits[-1]["payload"]["note"] == "late correction of the remaining range"
    # settled adjustments cannot be rewritten by another receipt
    with pytest.raises(CompensationError):
        log.verify_compensation_adjustment(
            0, "register", "p", "move", True, receipt_id="too-late",
        )


def test_duplicate_receipts_and_cross_adjustment_receipts(tmp_path):
    log = make_log(tmp_path / "log.jsonl", {"a": 6.0, "b": 4.0})
    log.plan_compensation_adjustment(0, "register", [
        {"source": "a", "target": "b", "quantity": 2.0, "adjustment_id": "one"},
        {"source": "a", "target": None, "quantity": 1.0, "adjustment_id": "two"},
    ], plan_id="p")
    log.run_compensation_adjustment(0, "register", "p", "one")
    log.run_compensation_adjustment(0, "register", "p", "two")
    log.verify_compensation_adjustment(0, "register", "p", "one", True, receipt_id="rcpt")
    # a duplicate receipt only appends an audit record, nothing moves twice
    plan = log.verify_compensation_adjustment(0, "register", "p", "one", True, receipt_id="rcpt")
    assert conservation(log)["recipients"]["b"]["quantity_allowed"] == 6.0
    duplicates = [
        e for e in log.audit_events(0)
        if e["kind"] == "compensation_adjustment_verify" and e["payload"].get("duplicate")
    ]
    assert len(duplicates) == 1
    # the same receipt cannot be charged to another adjustment
    with pytest.raises(CompensationError):
        log.verify_compensation_adjustment(0, "register", "p", "two", True, receipt_id="rcpt")
    assert plan["status"] == "in_progress"


def test_cancel_only_unsent_and_failure_branch_restores_source(tmp_path):
    log = make_log(tmp_path / "log.jsonl", {"a": 6.0, "b": 4.0})
    log.plan_compensation_adjustment(0, "register", [
        {"source": "a", "target": "b", "quantity": 2.0, "adjustment_id": "send-me"},
        {"source": "a", "target": "b", "quantity": 1.0, "adjustment_id": "cancel-me"},
        {"source": "b", "target": "a", "quantity": 3.0, "adjustment_id": "fail-me"},
    ], plan_id="p")
    log.run_compensation_adjustment(0, "register", "p", "send-me")
    log.run_compensation_adjustment(0, "register", "p", "fail-me")
    # sent adjustments cannot be cancelled; they wait for a receipt
    with pytest.raises(ad.UnconfirmedEffectError):
        log.cancel_compensation_adjustment(0, "register", "p", "send-me")
    plan = log.cancel_compensation_adjustment(
        0, "register", "p", "cancel-me", note="no longer needed",
    )
    item = next(i for i in plan["adjustments"] if i["adjustment_id"] == "cancel-me")
    assert item["status"] == "cancelled"
    with pytest.raises(CompensationError):
        log.cancel_compensation_adjustment(0, "register", "p", "cancel-me")
    # a failed receipt returns the in-transit quantity to its source
    log.verify_compensation_adjustment(
        0, "register", "p", "fail-me", False,
        receipt_id="fail-rcpt", note="transfer rejected",
    )
    allowance = conservation(log)
    assert allowance["recipients"]["b"]["quantity_allowed"] == 4.0
    assert allowance["total_quantity_in_transit"] == 2.0
    plan = log.compensation_adjustment_plan(0, "register", "p")
    assert plan["status"] == "in_progress"
    log.verify_compensation_adjustment(0, "register", "p", "send-me", True, receipt_id="ok")
    assert log.compensation_adjustment_plan(0, "register", "p")["status"] == "settled"
    assert conservation(log)["total_quantity_in_transit"] == 0.0


# ------------------------------------------------------- reopen and conservation
def test_reopen_continues_batched_execution_with_conservation(tmp_path):
    path = tmp_path / "log.jsonl"
    log = make_log(path, {"a": 5.0, "b": 3.0, "c": 2.0})
    log.plan_compensation_adjustment(0, "register", [
        {"source": "a", "target": "b", "quantity": 2.0, "adjustment_id": "a-b"},
        {"source": "b", "target": "c", "quantity": 1.0, "adjustment_id": "b-c"},
        {"source": "c", "target": None, "quantity": 1.0, "adjustment_id": "c-drop"},
    ], plan_id="batch")
    log.run_compensation_adjustment(0, "register", "batch", "a-b")
    log.run_compensation_adjustment(0, "register", "batch", "b-c")

    reopened = ad.TrainingEventLog(path)
    allowance = conservation(reopened)
    assert allowance["total_quantity_approved"] == 10.0
    assert allowance["total_quantity_in_transit"] == 3.0
    # unsent adjustments can still be sent or cancelled after reopen
    with pytest.raises(UnconfirmedEffectError := ad.UnconfirmedEffectError):
        reopened.run_compensation_adjustment(0, "register", "batch", "a-b")
    reopened.run_compensation_adjustment(0, "register", "batch", "c-drop")
    # cross-recipient partial success: confirm parts, fail one branch
    reopened.verify_compensation_adjustment(
        0, "register", "batch", "a-b", True, quantity=1.0, receipt_id="ab-1",
    )
    reopened.verify_compensation_adjustment(
        0, "register", "batch", "b-c", False, receipt_id="bc-fail",
    )
    reopened.verify_compensation_adjustment(
        0, "register", "batch", "c-drop", True, receipt_id="cd-1",
    )
    allowance = conservation(reopened)
    assert allowance["recipients"]["b"]["quantity_allowed"] == pytest.approx(4.0)
    assert allowance["recipients"]["c"]["quantity_allowed"] == pytest.approx(1.0)
    assert allowance["total_quantity_withdrawn"] == pytest.approx(1.0)
    assert allowance["total_quantity_in_transit"] == pytest.approx(1.0)
    # duplicate receipt after reopen still does not double-apply
    reopened.verify_compensation_adjustment(
        0, "register", "batch", "a-b", True, quantity=1.0, receipt_id="ab-1",
    )
    assert conservation(reopened)["recipients"]["b"]["quantity_allowed"] == pytest.approx(4.0)

    reopened_again = ad.TrainingEventLog(path)
    reopened_again.verify_compensation_adjustment(
        0, "register", "batch", "a-b", True, receipt_id="ab-2",
    )
    plan = reopened_again.compensation_adjustment_plan(0, "register", "batch")
    assert plan["status"] == "settled"
    final = conservation(reopened_again)
    assert final["total_quantity_approved"] == 10.0
    assert final["total_quantity_allocated"] == pytest.approx(9.0)
    assert final["total_quantity_withdrawn"] == pytest.approx(1.0)
    assert final["total_quantity_in_transit"] == 0.0
