"""Compensation arrangements and checkpoint branches for external effects.

Covers:
* compensation decisions (retain / compensate / pending) for occurred
  effects without rewriting the original fact;
* idempotent compensation execution, scoped quantities, late receipts and
  duplicate verification never double-offsetting;
* no compensation without evidence that the original effect occurred;
* continuation reports for interdependent, partially compensated effects;
* branching from a confirmed checkpoint: inherited facts, fresh optimizer
  state, no replayed parameter updates;
* recovery after interruption: original facts, human decisions and
  compensation outcomes stay traceable; plain steps stay compatible.
"""
import numpy as np
import pytest

import autodiff as ad
from autodiff import nn, optim, tensor
from autodiff.events import CompensationError


def make_model(seed=0):
    return nn.Sequential(nn.Linear(4, 8, seed=seed), nn.ReLU(), nn.Linear(8, 2, seed=seed + 1))


def run_completed_step(log, step, effects=("reserve", "submit"), payloads=None, model=None, seed=0):
    model = model if model is not None else make_model(seed)
    opt = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    x = tensor(np.random.default_rng(1).standard_normal((3, 4)))
    payloads = payloads or {}
    with log.use(), log.begin_step(step, effects) as tx:
        for name in effects:
            tx.run_effect(name, lambda n=name: payloads.get(n, {"effect": n}))
        loss = ad.sum(model(x) ** 2)
        loss.backward()
        tx.complete(model, optimizer=opt, loss=loss)
    return model, opt


# ------------------------------------------------------- decisions
def test_decision_requires_occurred_effect(tmp_path):
    log = ad.TrainingEventLog(tmp_path / "log.jsonl")
    with log.use(), log.begin_step(0, ["reserve", "submit"]) as tx:
        tx.run_effect("reserve", lambda: {"id": 1})
        with pytest.raises(ad.ExternalEffectUncertain):
            tx.run_effect("submit", lambda: (_ for _ in ()).throw(ad.ExternalEffectUncertain("maybe")))
    # submit is unconfirmed: no compensation decision may execute for it
    with pytest.raises(ad.UnconfirmedEffectError):
        log.decide_compensation(0, "submit", "compensate")
    with pytest.raises(ad.UnconfirmedEffectError):
        log.decide_compensation(0, "submit", "retain")
    # pending is always allowed; occurred effects accept all decisions
    log.decide_compensation(0, "submit", "pending")
    rec = log.decide_compensation(0, "reserve", "retain", note="keep the registration")
    assert rec["decision"] == "retain"
    # the original fact is never rewritten
    assert log.effect_statuses(0)["reserve"] == "occurred"
    assert log.effect_statuses(0)["submit"] == "unconfirmed"


def test_compensation_never_runs_without_evidence(tmp_path):
    log = ad.TrainingEventLog(tmp_path / "log.jsonl")
    with log.use(), log.begin_step(0, ["reserve"]) as tx:
        with pytest.raises(ad.ExternalEffectNotExecuted):
            tx.run_effect("reserve", lambda: (_ for _ in ()).throw(ad.ExternalEffectNotExecuted("no")))
    calls = []
    with pytest.raises(ad.UnconfirmedEffectError):
        log.decide_compensation(0, "reserve", "compensate")
    with pytest.raises(KeyError):
        log.run_compensation(0, "reserve", lambda: calls.append(1) or {"ok": True})
    assert calls == []


# ------------------------------------------------------- execution
def test_run_compensation_is_idempotent_and_scoped(tmp_path):
    log = ad.TrainingEventLog(tmp_path / "log.jsonl")
    run_completed_step(log, 0, effects=["register"])
    log.decide_compensation(0, "register", "compensate", quantity=3.0)

    calls = []
    out = log.run_compensation(0, "register", lambda: calls.append(1) or {"offset": 1}, quantity=2.0)
    assert out == {"offset": 1}
    rec = log.compensation(0, "register")
    assert rec["quantity_compensated"] == 2.0
    assert rec["quantity_allowed"] == 3.0
    assert rec["target"].endswith(":effect:register")

    # a request beyond the remaining approved balance is refused before any
    # external action: the ledger may not be trimmed to fake compliance
    with pytest.raises(ad.CompensationError):
        log.run_compensation(0, "register", lambda: calls.append(1) or {"offset": 2}, quantity=5.0)
    assert calls == [1]
    assert log.compensation(0, "register")["quantity_compensated"] == 2.0
    # requesting exactly the remaining balance executes and reconciles
    log.run_compensation(0, "register", lambda: calls.append(1) or {"offset": 2}, quantity=1.0)
    assert log.compensation(0, "register")["quantity_compensated"] == 3.0
    # fully compensated: repeated applications replay, never re-apply
    again = log.run_compensation(0, "register", lambda: calls.append(1) or {"offset": 3})
    assert again == {"offset": 2}
    assert calls == [1, 1]
    # decision is locked once the compensation executed
    with pytest.raises(ad.EventExistsError):
        log.decide_compensation(0, "register", "retain")


def test_late_receipt_and_duplicate_verify_no_double_count(tmp_path):
    log = ad.TrainingEventLog(tmp_path / "log.jsonl")
    run_completed_step(log, 0, effects=["register"])
    log.decide_compensation(0, "register", "compensate", quantity=2.0)

    with pytest.raises(ad.ExternalEffectUncertain):
        log.run_compensation(0, "register", lambda: (_ for _ in ()).throw(ad.ExternalEffectUncertain("receipt lost")))
    # unconfirmed: retry is refused until the receipt is resolved
    with pytest.raises(ad.UnconfirmedEffectError):
        log.run_compensation(0, "register", lambda: {"offset": 1})
    # late receipt arrives and confirms
    rec = log.verify_compensation(0, "register", True, {"offset": "late"}, note="late receipt")
    assert rec["quantity_compensated"] == 2.0
    # a duplicate late receipt is audited but does not compensate twice
    rec = log.verify_compensation(0, "register", True, {"offset": "late"}, note="duplicate")
    assert rec["quantity_compensated"] == 2.0
    with pytest.raises(ad.EventExistsError):
        log.verify_compensation(0, "register", False)
    kinds = [e["kind"] for e in log.audit_events(0)]
    assert kinds.count("compensation_verification") == 2


def test_reverify_original_after_partial_compensation(tmp_path):
    log = ad.TrainingEventLog(tmp_path / "log.jsonl")
    run_completed_step(log, 0, effects=["register"], payloads={"register": {"qty": 5}})
    log.decide_compensation(0, "register", "compensate", quantity=5.0)
    log.run_compensation(0, "register", lambda: {"offset": 2}, quantity=2.0)
    # re-verifying the original occurred fact must not disturb compensation
    log.verify_effect(0, "register", True, {"qty": 5}, note="double-checked")
    rec = log.compensation(0, "register")
    assert rec["quantity_compensated"] == 2.0
    assert log.effect_statuses(0)["register"] == "occurred"
    assert log.effect_resolution(0, "register")["detail"] == "partially_compensated"


# ------------------------------------------------------- continuation
def test_continuation_with_interdependent_effects(tmp_path):
    log = ad.TrainingEventLog(tmp_path / "log.jsonl")
    run_completed_step(log, 0, effects=["reserve", "submit", "publish"])

    report = log.continuation(0)
    assert report["conflicting"] == ["reserve", "submit", "publish"]
    assert report["continuable"] == []

    log.decide_compensation(0, "reserve", "retain")
    log.decide_compensation(0, "submit", "compensate", quantity=2.0)
    log.run_compensation(0, "submit", lambda: {"offset": 1}, quantity=1.0)
    report = log.continuation(0)
    # reserve is reconciled; submit is only partially compensated and
    # everything depending on it stays in conflict with the external facts
    assert report["continuable"] == ["reserve"]
    assert report["conflicting"] == ["submit", "publish"]
    assert report["effects"]["submit"]["detail"] == "partially_compensated"
    assert not report["step_continuable"]

    log.run_compensation(0, "submit", lambda: {"offset": 1})
    log.decide_compensation(0, "publish", "retain")
    report = log.continuation(0)
    assert report["continuable"] == ["reserve", "submit", "publish"]
    assert report["step_continuable"]


# ------------------------------------------------------- branching
def test_branch_inherits_facts_not_unconfirmed_decisions(tmp_path):
    log = ad.TrainingEventLog(tmp_path / "main.jsonl")
    model, opt = run_completed_step(log, 0, effects=["register"])
    log.decide_compensation(0, "register", "compensate", quantity=2.0)
    log.run_compensation(0, "register", lambda: {"offset": 1}, quantity=1.0)
    # a second, still unconfirmed compensation decision on the main branch
    log.decide_compensation(0, "register", "compensate", quantity=2.0)

    branch = log.create_branch("retry", 0, tmp_path / "branch.jsonl")
    info = branch.branch_record()
    assert info["from_step"] == "0"
    inherited = branch.inherited_effects()
    names = {e.get("name") for e in inherited}
    assert "register" in names
    # the occurred compensation is a real fact and is inherited
    comps = [e for e in inherited if e.get("quantity_compensated")]
    assert comps and comps[0]["quantity_compensated"] == 1.0
    # the unconfirmed decision itself is not inherited as a branch decision
    with pytest.raises(KeyError):
        branch.compensation(0, "register")

    # model parameters correspond to the checkpoint, gradients cleared,
    # optimizer state fresh: no update is replayed while branching
    fresh = make_model(seed=99)
    fresh_opt = optim.Adam(fresh.parameters(), lr=0.01)
    before = [p.data.copy() for p in fresh.parameters()]
    branch.restore_branch(fresh, optimizer=fresh_opt)
    for p, trained in zip(fresh.parameters(), model.parameters()):
        assert np.allclose(p.data, trained.data)
        assert p.grad is None
    assert fresh_opt.t == 0
    assert any(not np.allclose(a, b.data) for a, b in zip(before, fresh.parameters()))


def test_branch_requires_completed_step(tmp_path):
    log = ad.TrainingEventLog(tmp_path / "log.jsonl")
    with log.use(), log.begin_step(0, ["reserve"]) as tx:
        tx.run_effect("reserve", lambda: {"id": 1})
        tx.fail("interrupted")
    with pytest.raises(ad.IncompleteTrainingStepError):
        log.create_branch("nope", 0, tmp_path / "b.jsonl")


def test_compensation_processing_does_not_touch_model(tmp_path):
    log = ad.TrainingEventLog(tmp_path / "log.jsonl")
    model, opt = run_completed_step(log, 0, effects=["register"])
    log.decide_compensation(0, "register", "compensate")
    params_before = [p.data.copy() for p in model.parameters()]
    log.run_compensation(0, "register", lambda: {"offset": 1})
    for p, before in zip(model.parameters(), params_before):
        assert np.array_equal(p.data, before)


# ------------------------------------------------------- recovery
def test_recovery_after_interrupted_compensation(tmp_path):
    path = tmp_path / "log.jsonl"
    log = ad.TrainingEventLog(path)
    run_completed_step(log, 0, effects=["register"])
    log.decide_compensation(0, "register", "compensate", quantity=2.0, note="human decision")
    with pytest.raises(ad.ExternalEffectUncertain):
        log.run_compensation(0, "register", lambda: (_ for _ in ()).throw(ad.ExternalEffectUncertain("crash")))

    # process restart: original fact, human decision and compensation
    # outcome are all still traceable
    reopened = ad.TrainingEventLog(path)
    assert reopened.effect_statuses(0)["register"] == "occurred"
    rec = reopened.compensation(0, "register")
    assert rec["decision"] == "compensate"
    assert rec["status"] == "unconfirmed"
    reopened.verify_compensation(0, "register", True, {"offset": 2})
    assert reopened.compensation(0, "register")["quantity_compensated"] == 2.0
    kinds = {e["kind"] for e in reopened.audit_events(0)}
    assert {"training_step", "training_effect", "compensation_decision", "compensation", "compensation_verification"} <= kinds


def test_plain_step_without_compensation_still_compatible(tmp_path):
    path = tmp_path / "log.jsonl"
    log = ad.TrainingEventLog(path)
    model, _ = run_completed_step(log, 0, effects=["reserve", "submit"])
    assert log.is_step_complete(0)
    assert log.compensations(0) == {}
    report = log.continuation(0)
    # occurred effects without any decision still require a human arrangement
    assert report["conflicting"] == ["reserve", "submit"]
    log.decide_compensation(0, "reserve", "retain")
    log.decide_compensation(0, "submit", "retain")
    assert log.continuation(0)["step_continuable"]
    # reopening the log preserves everything
    reopened = ad.TrainingEventLog(path)
    assert reopened.is_step_complete(0)
    assert reopened.continuation(0)["step_continuable"]


def test_random_compute_and_multiple_backwards_around_compensation(tmp_path):
    path = tmp_path / "log.jsonl"
    log = ad.TrainingEventLog(path)
    ad.manual_seed(7)
    model = make_model()
    opt = optim.SGD(model.parameters(), lr=0.1)
    x = tensor(np.random.default_rng(3).standard_normal((4, 4)), requires_grad=True)
    with log.use(), log.begin_step(0, ["batch"]) as tx:
        batch = tx.run_effect("batch", lambda: {"seed": 1}, lambda p: tensor(np.random.default_rng(p["seed"]).standard_normal((4, 4))))
        h = model(batch + x)
        loss = ad.sum(h ** 2)
        loss.backward()
        loss.backward()  # accumulated gradients, single graph
        tx.complete(model, optimizer=opt, loss=loss)
    assert log.is_step_complete(0)
    log.decide_compensation(0, "batch", "compensate")
    log.run_compensation(0, "batch", lambda: {"released": True})
    assert log.continuation(0)["step_continuable"]
    # a fresh process can still replay the occurred effect deterministically
    reopened = ad.TrainingEventLog(path)
    step = reopened.resume_step(0)
    again = step.run_effect("batch", replay=lambda p: tensor(np.random.default_rng(p["seed"]).standard_normal((4, 4))))
    assert np.allclose(again.data, batch.data)


# ------------------------------------------------------- approval boundary
def test_over_approval_request_is_rejected_not_trimmed(tmp_path):
    log = ad.TrainingEventLog(tmp_path / "log.jsonl")
    run_completed_step(log, 0, effects=["register"])
    log.decide_compensation(0, "register", "compensate", quantity=5.0)

    external = []
    log.run_compensation(0, "register", lambda: external.append(3.0) or {"offset": 3}, quantity=3.0)
    # approved 5, already compensated 3: requesting 4 must be refused before
    # any external action, not silently trimmed to 2 in the ledger only
    with pytest.raises(ad.CompensationError):
        log.run_compensation(0, "register", lambda: external.append(4.0) or {"offset": 4}, quantity=4.0)
    assert external == [3.0]
    rec = log.compensation(0, "register")
    assert rec["quantity_compensated"] == 3.0
    assert rec["quantity_pending"] == 0.0
    # exactly the remaining balance still executes
    log.run_compensation(0, "register", lambda: external.append(2.0) or {"offset": 2}, quantity=2.0)
    assert external == [3.0, 2.0]
    assert log.compensation(0, "register")["quantity_compensated"] == 5.0
    # approval exhausted: nothing more may leave the process
    with pytest.raises(ad.CompensationError):
        log.run_compensation(0, "register", lambda: external.append(1.0), quantity=1.0)
    assert external == [3.0, 2.0]


def test_lost_receipt_holds_pending_and_late_receipt_settles(tmp_path):
    log = ad.TrainingEventLog(tmp_path / "log.jsonl")
    run_completed_step(log, 0, effects=["register"])
    log.decide_compensation(0, "register", "compensate", quantity=5.0)
    log.run_compensation(0, "register", lambda: {"offset": 3}, quantity=3.0)

    # the external action may have run but the receipt was lost
    with pytest.raises(ad.ExternalEffectUncertain):
        log.run_compensation(
            0, "register",
            lambda: (_ for _ in ()).throw(ad.ExternalEffectUncertain("receipt lost")),
            quantity=2.0,
        )
    rec = log.compensation(0, "register")
    assert rec["status"] == "unconfirmed"
    assert rec["quantity_compensated"] == 3.0
    assert rec["quantity_pending"] == 2.0
    # retrying must not consume the same approval again, nor assume nothing happened
    with pytest.raises(ad.UnconfirmedEffectError):
        log.run_compensation(0, "register", lambda: {"offset": 2}, quantity=2.0)
    # the pending amount is reserved: the approval cannot be lowered beneath it
    with pytest.raises(ad.CompensationError):
        log.decide_compensation(0, "register", "compensate", quantity=4.0)
    # a receipt confirming more than was attempted is rejected
    with pytest.raises(ad.CompensationError):
        log.verify_compensation(0, "register", True, {"offset": 3}, quantity=3.0)
    # the late receipt settles exactly the pending amount
    rec = log.verify_compensation(0, "register", True, {"offset": 2}, note="late receipt")
    assert rec["quantity_compensated"] == 5.0
    assert rec["quantity_pending"] == 0.0
    # a duplicate late receipt is audited but never compensates twice
    rec = log.verify_compensation(0, "register", True, {"offset": 2}, note="duplicate")
    assert rec["quantity_compensated"] == 5.0
    kinds = [e["kind"] for e in log.audit_events(0)]
    assert kinds.count("compensation_attempt") == 1
    assert kinds.count("compensation_verification") == 2


def test_pending_quantity_survives_reopen(tmp_path):
    path = tmp_path / "log.jsonl"
    log = ad.TrainingEventLog(path)
    run_completed_step(log, 0, effects=["register"])
    log.decide_compensation(0, "register", "compensate", quantity=4.0)
    log.run_compensation(0, "register", lambda: {"offset": 1}, quantity=1.0)
    with pytest.raises(ad.ExternalEffectUncertain):
        log.run_compensation(
            0, "register",
            lambda: (_ for _ in ()).throw(ad.ExternalEffectUncertain("crash after send")),
            quantity=2.0,
        )

    reopened = ad.TrainingEventLog(path)
    rec = reopened.compensation(0, "register")
    assert rec["quantity_compensated"] == 1.0
    assert rec["quantity_pending"] == 2.0
    assert rec["status"] == "unconfirmed"
    with pytest.raises(ad.UnconfirmedEffectError):
        reopened.run_compensation(0, "register", lambda: {"offset": 2}, quantity=2.0)
    # human check: the attempt definitely did not happen; the reservation is
    # released and only then may the balance be executed again
    rec = reopened.verify_compensation(0, "register", False, note="provider confirms nothing ran")
    assert rec["quantity_pending"] == 0.0
    assert rec["quantity_compensated"] == 1.0
    external = []
    reopened.run_compensation(0, "register", lambda: external.append(2.0) or {"offset": 2}, quantity=2.0)
    assert external == [2.0]
    assert reopened.compensation(0, "register")["quantity_compensated"] == 3.0


def test_two_concurrent_branches_reconcile_against_inherited_facts(tmp_path):
    main = ad.TrainingEventLog(tmp_path / "main.jsonl")
    run_completed_step(main, 0, effects=["register"])
    main.decide_compensation(0, "register", "compensate", quantity=3.0)
    main.run_compensation(0, "register", lambda: {"offset": 1}, quantity=1.0)

    branch_a = main.create_branch("a", 0, tmp_path / "a.jsonl")
    branch_b = main.create_branch("b", 0, tmp_path / "b.jsonl")

    for branch in (branch_a, branch_b):
        # the inherited confirmed compensation consumes part of the approval
        rec = branch.decide_compensation(0, "register", "compensate", quantity=3.0)
        assert rec["quantity_compensated"] == 1.0
        # requesting the full approval again would reuse it: refused
        with pytest.raises(ad.CompensationError):
            branch.run_compensation(0, "register", lambda: {"offset": 3}, quantity=3.0)

    # branch A settles its remaining balance directly
    branch_a.run_compensation(0, "register", lambda: {"offset": 2}, quantity=2.0)
    assert branch_a.compensation(0, "register")["quantity_compensated"] == 3.0

    # branch B interleaves a lost receipt and its late arrival
    with pytest.raises(ad.ExternalEffectUncertain):
        branch_b.run_compensation(
            0, "register",
            lambda: (_ for _ in ()).throw(ad.ExternalEffectUncertain("no ack")),
            quantity=2.0,
        )
    with pytest.raises(ad.UnconfirmedEffectError):
        branch_b.run_compensation(0, "register", lambda: {"offset": 2}, quantity=2.0)
    rec = branch_b.verify_compensation(0, "register", True, {"offset": 2}, note="late receipt")
    assert rec["quantity_compensated"] == 3.0
    assert rec["quantity_pending"] == 0.0
    # each branch reconciles: inherited 1 + own 2 = approved 3, never more
    for branch in (branch_a, branch_b):
        rec = branch.compensation(0, "register")
        assert rec["quantity_compensated"] == 3.0
        assert rec["quantity_allowed"] == 3.0


def test_partial_verification_preserves_facts_and_attempt_basis(tmp_path):
    log = ad.TrainingEventLog(tmp_path / "log.jsonl")
    run_completed_step(log, 0, effects=["register"])
    log.decide_compensation(0, "register", "compensate", quantity=5.0)
    with pytest.raises(ad.ExternalEffectUncertain):
        log.run_compensation(
            0, "register",
            lambda: (_ for _ in ()).throw(ad.ExternalEffectUncertain("timeout")),
            quantity=4.0,
        )
    # human confirms only part of the attempt actually happened
    rec = log.verify_compensation(0, "register", True, {"offset": 1.5}, quantity=1.5, note="partial: 1.5 of 4")
    assert rec["quantity_compensated"] == 1.5
    assert rec["quantity_pending"] == 2.5
    assert rec["status"] == "unconfirmed"
    # the rest is later confirmed as not having happened: reservation released
    rec = log.verify_compensation(0, "register", False, note="remainder never ran")
    assert rec["quantity_pending"] == 0.0
    assert rec["quantity_compensated"] == 1.5
    # the original fact and every attempt/verification stay on record
    assert log.effect_statuses(0)["register"] == "occurred"
    audits = [e for e in log.audit_events(0) if e["kind"].startswith("compensation_")]
    quantities = [e["payload"].get("quantity") for e in audits]
    assert 4.0 in quantities and 1.5 in quantities
    notes = [e["payload"].get("note", "") for e in audits]
    assert any("partial" in n for n in notes) and any("remainder" in n for n in notes)
    # the released balance is executable again, up to the approval boundary
    log.run_compensation(0, "register", lambda: {"offset": 3.5}, quantity=3.5)
    assert log.compensation(0, "register")["quantity_compensated"] == 5.0


def test_multiple_recipients_reallocation_and_late_receipt_boundary(tmp_path):
    path = tmp_path / "log.jsonl"
    log = ad.TrainingEventLog(path)
    run_completed_step(log, 0, effects=["register"])
    log.decide_compensation(
        0, "register", "compensate",
        quantity=10.0, allocations={"vendor-a": 4.0, "vendor-b": 6.0},
    )

    external_actions = []
    log.run_compensation(
        0, "register",
        lambda: external_actions.append(("vendor-a", 1.0)) or {"offset": 1.0},
        quantity=1.0, recipient_id="vendor-a", attempt_id="a-ok",
    )
    with pytest.raises(ad.ExternalEffectUncertain):
        log.run_compensation(
            0, "register",
            lambda: (_ for _ in ()).throw(ad.ExternalEffectUncertain("a timeout")),
            quantity=2.0, recipient_id="vendor-a", attempt_id="a-pending",
        )
    # vendor-b is independent and can continue while vendor-a waits.
    log.run_compensation(
        0, "register",
        lambda: external_actions.append(("vendor-b", 2.0)) or {"offset": 2.0},
        quantity=2.0, recipient_id="vendor-b", attempt_id="b-ok",
    )

    reopened = ad.TrainingEventLog(path)
    allowance = reopened.compensation_allowance(0, "register")
    assert allowance["total_quantity_approved"] == 10.0
    assert allowance["total_quantity_allocated"] == 10.0
    assert allowance["total_quantity_compensated"] == 3.0
    assert allowance["total_quantity_pending"] == 2.0
    assert allowance["recipients"]["vendor-a"]["quantity_available"] == 1.0
    assert allowance["recipients"]["vendor-b"]["quantity_available"] == 4.0
    # The 2 pending units remain owned by vendor-a and cannot be moved.
    with pytest.raises(ad.CompensationError):
        reopened.reallocate_compensation(0, "register", "vendor-a", 2.0, "vendor-b")

    options = reopened.compensation_reallocation_options(
        0, "register", "vendor-a", quantity=1.0,
        target_recipient_ids=["vendor-b"],
    )
    assert options["retain"]["recipients"]["vendor-a"]["quantity_allowed"] == 4.0
    assert options["withdraw"]["total_quantity_allocated"] == 9.0
    assert options["withdraw"]["total_quantity_withdrawn"] == 1.0
    assert options["transfers"]["vendor-b"]["recipients"]["vendor-b"]["quantity_allowed"] == 7.0
    # Computing options never mutates the ledger.
    assert reopened.compensation_allowance(0, "register")["total_quantity_allocated"] == 10.0

    reopened.reallocate_compensation(
        0, "register", "vendor-a", 1.0, "vendor-b", note="move only unused quota",
    )
    reopened.run_compensation(
        0, "register",
        lambda: external_actions.append(("vendor-b", 1.0)) or {"offset": 3.0},
        quantity=1.0, recipient_id="vendor-b", attempt_id="b-transferred",
    )
    # The old receipt stays tied to vendor-a; presenting it for vendor-b fails.
    with pytest.raises(ad.CompensationError):
        reopened.verify_compensation(
            0, "register", True, {"offset": 2.0},
            quantity=2.0, recipient_id="vendor-b", attempt_id="a-pending",
            receipt_id="old-a-receipt",
        )
    rec = reopened.verify_compensation(
        0, "register", True, {"offset": 1.0},
        quantity=1.0, recipient_id="vendor-a", attempt_id="a-pending",
        receipt_id="partial-a-receipt", note="one of two arrived",
    )
    with pytest.raises(ad.CompensationError):
        reopened.verify_compensation(
            0, "register", True, {"offset": 1.0},
            recipient_id="vendor-b", receipt_id="partial-a-receipt",
        )
    assert rec["quantity_compensated"] == 5.0
    assert rec["quantity_pending"] == 1.0
    duplicate = reopened.verify_compensation(
        0, "register", True, {"offset": 1.0},
        quantity=1.0, recipient_id="vendor-a", attempt_id="a-pending",
        receipt_id="partial-a-receipt",
    )
    assert duplicate["quantity_compensated"] == 5.0
    rec = reopened.verify_compensation(
        0, "register", False,
        recipient_id="vendor-a", attempt_id="a-pending",
        receipt_id="a-final-receipt", note="remaining one never ran",
    )
    assert rec["quantity_compensated"] == 5.0
    assert rec["quantity_pending"] == 0.0

    final = reopened.compensation_allowance(0, "register")
    assert final["total_quantity_approved"] == 10.0
    assert final["total_quantity_allocated"] == 10.0
    assert final["total_quantity_compensated"] == 5.0
    assert final["recipients"]["vendor-a"]["quantity_compensated"] == 2.0
    assert final["recipients"]["vendor-b"]["quantity_compensated"] == 3.0
    assert external_actions == [
        ("vendor-a", 1.0), ("vendor-b", 2.0), ("vendor-b", 1.0),
    ]
    audit_notes = [event["payload"].get("note", "") for event in reopened.audit_events(0)]
    assert "move only unused quota" in audit_notes
