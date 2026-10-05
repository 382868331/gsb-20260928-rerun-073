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


# ----------------------------------------- multiple independent payees
def _multi_payee_log(tmp_path, total=10.0, allocations=("alice", "bob", "carol")):
    log = ad.TrainingEventLog(tmp_path / "log.jsonl")
    run_completed_step(log, 0, effects=["payout"])
    log.decide_compensation(0, "payout", "compensate", quantity=total)
    for payee in allocations:
        log.allocate_compensation(0, "payout", payee, float(total) / len(allocations))
    return log


def test_total_and_per_target_approvals_are_separately_reconcilable(tmp_path):
    log = _multi_payee_log(tmp_path, total=9.0)
    rec = log.compensation(0, "payout")
    assert rec["quantity_allowed"] == 9.0
    assert rec["quantity_allocated"] == 9.0
    assert rec["quantity_unallocated"] == 0.0
    targets = {t["target"]: t for t in rec["allocations"]}
    assert set(targets) == {"alice", "bob", "carol"}
    for target in targets.values():
        assert target["quantity_allowed"] == 3.0
        assert target["quantity_compensated"] == 0.0
        assert target["quantity_pending"] == 0.0

    # the total approval cannot be over-committed across targets
    with pytest.raises(ad.CompensationError):
        log.allocate_compensation(0, "payout", "dave", 1.0)
    # an existing target is topped up, not duplicated
    log.withdraw_compensation_allocation(0, "payout", "carol", 1.0)
    log.allocate_compensation(0, "payout", "dave", 0.5)
    log.allocate_compensation(0, "payout", "dave", 0.5)
    rec = log.compensation(0, "payout")
    assert rec["quantity_allocated"] == 9.0
    assert rec["quantity_unallocated"] == 0.0
    dave = next(t for t in rec["allocations"] if t["target"] == "dave")
    assert dave["quantity_allowed"] == 1.0

    # a target cannot spend beyond its own allowance, even though other
    # targets still hold unused approval
    actions = []
    with pytest.raises(ad.CompensationError):
        log.run_compensation(0, "payout", lambda: actions.append(4.0) or {"to": "alice"}, target="alice", quantity=4.0)
    assert actions == []
    log.run_compensation(0, "payout", lambda: actions.append(2.0) or {"to": "alice"}, target="alice", quantity=2.0)
    rec = log.compensation(0, "payout")
    assert rec["quantity_compensated"] == 2.0
    alice = next(t for t in rec["allocations"] if t["target"] == "alice")
    assert alice["quantity_compensated"] == 2.0 and alice["quantity_remaining"] == 1.0


def test_unused_target_allowance_needs_explicit_reallocation(tmp_path):
    log = _multi_payee_log(tmp_path, total=9.0)
    # bob uses part; carol never executes: alice may not silently take it
    log.run_compensation(0, "payout", lambda: {"to": "bob", "n": 2}, target="bob", quantity=2.0)
    with pytest.raises(ad.CompensationError):
        log.run_compensation(0, "payout", lambda: {"to": "alice"}, target="alice", quantity=3.5)

    # options can be compared without performing any external action
    options = log.compensation_reallocation_options(0, "payout", "carol")
    assert options["keep"] == {"target": "carol", "quantity_available": 3.0}
    assert options["withdraw"] == {"target": "carol", "quantity_released": 3.0, "quantity_unallocated_after": 3.0}
    partial = {m["target"]: m for m in options["transfer"]}
    assert partial["alice"]["quantity_transferable"] == 3.0
    assert partial["alice"]["alice"] == 6.0
    assert partial["alice"]["carol"] == 0.0
    # carol's own unused allowance is still intact after the comparison
    assert log.compensation(0, "payout").get("quantity_unallocated", 0.0) == 0.0

    # explicit partial transfer moves only the named amount
    log.reallocate_compensation(0, "payout", "carol", "alice", 2.0)
    rec = log.compensation(0, "payout")
    targets = {t["target"]: t for t in rec["allocations"]}
    assert targets["carol"]["quantity_allowed"] == 1.0
    assert targets["alice"]["quantity_allowed"] == 5.0
    assert rec["quantity_allocated"] == 9.0
    log.run_compensation(0, "payout", lambda: {"to": "alice"}, target="alice", quantity=3.5)
    targets = {t["target"]: t for t in log.compensation(0, "payout")["allocations"]}
    assert targets["alice"]["quantity_compensated"] == 3.5
    assert targets["carol"]["quantity_compensated"] == 0.0


def test_pending_send_holds_source_allowance_and_late_receipt_settles_once(tmp_path):
    log = _multi_payee_log(tmp_path, total=9.0)
    # an action to bob was sent but its receipt is unknown: bob's allowance
    # stays reserved and cannot be given to anyone else meanwhile
    with pytest.raises(ad.ExternalEffectUncertain):
        log.run_compensation(
            0, "payout",
            lambda: (_ for _ in ()).throw(ad.ExternalEffectUncertain("ack lost")),
            target="bob", quantity=3.0,
        )
    with pytest.raises(ad.CompensationError):
        log.reallocate_compensation(0, "payout", "bob", "alice", 1.0)
    with pytest.raises(ad.CompensationError):
        log.withdraw_compensation_allocation(0, "payout", "bob", 1.0)
    # other targets remain executable while bob is pending
    log.run_compensation(0, "payout", lambda: {"to": "alice", "n": 1}, target="alice", quantity=1.0)

    # process restart preserves the reservation
    reopened = ad.TrainingEventLog(log.path)
    rec = reopened.compensation(0, "payout")
    targets = {t["target"]: t for t in rec["allocations"]}
    assert targets["bob"]["quantity_pending"] == 3.0
    assert targets["bob"]["quantity_remaining"] == 0.0
    # late receipt settles the original target only
    rec = reopened.verify_compensation(0, "payout", True, {"to": "bob", "n": 3}, target="bob")
    targets = {t["target"]: t for t in rec["allocations"]}
    assert targets["bob"]["quantity_compensated"] == 3.0
    assert targets["bob"]["quantity_pending"] == 0.0
    assert targets["alice"]["quantity_compensated"] == 1.0
    # a duplicate late receipt must never be booked against another target
    dup = reopened.verify_compensation(0, "payout", True, {"to": "bob", "n": 3}, target="bob", note="duplicate late")
    targets = {t["target"]: t for t in dup["allocations"]}
    assert targets["bob"]["quantity_compensated"] == 3.0
    assert targets["carol"]["quantity_compensated"] == 0.0
    assert dup["quantity_compensated"] == 4.0
    kinds = [e["kind"] for e in reopened.audit_events(0)]
    assert kinds.count("compensation_verification") == 2


def test_reallocation_after_not_occurred_verification_releases_reservation(tmp_path):
    log = _multi_payee_log(tmp_path, total=9.0)
    with pytest.raises(ad.ExternalEffectUncertain):
        log.run_compensation(
            0, "payout",
            lambda: (_ for _ in ()).throw(ad.ExternalEffectUncertain("timeout")),
            target="bob", quantity=2.0,
        )
    # human check proves nothing ran externally: the reservation is released
    rec = log.verify_compensation(0, "payout", False, target="bob", note="provider says nothing ran")
    targets = {t["target"]: t for t in rec["allocations"]}
    assert targets["bob"]["quantity_pending"] == 0.0
    assert targets["bob"]["quantity_remaining"] == 3.0
    # withdrawal and transfer of the released allowance are now both possible
    log.reallocate_compensation(0, "payout", "bob", "alice", 2.0)
    log.withdraw_compensation_allocation(0, "payout", "bob", 1.0)
    rec = log.compensation(0, "payout")
    targets = {t["target"]: t for t in rec["allocations"]}
    assert targets["bob"]["quantity_allowed"] == 0.0
    assert targets["alice"]["quantity_allowed"] == 5.0
    assert rec["quantity_unallocated"] == 1.0
    # the withdrawn slice can be handed out again explicitly
    log.allocate_compensation(0, "payout", "carol", 1.0)
    assert log.compensation(0, "payout")["quantity_unallocated"] == 0.0


def test_parallel_compensation_never_consumes_total_approval_twice(tmp_path):
    import threading
    log = ad.TrainingEventLog(tmp_path / "log.jsonl")
    run_completed_step(log, 0, effects=["payout"])
    log.decide_compensation(0, "payout", "compensate", quantity=6.0)
    log.allocate_compensation(0, "payout", "alice", 6.0)

    sent = []
    errors = []
    barrier = threading.Barrier(3)

    def worker():
        barrier.wait()
        try:
            log.run_compensation(
                0, "payout", lambda: sent.append(2.0) or {"to": "alice", "n": 2},
                target="alice", quantity=2.0,
            )
        except Exception as exc:  # noqa: BLE001 - collect the refused races
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rec = log.compensation(0, "payout")
    assert rec["quantity_compensated"] == 6.0
    assert rec["quantity_pending"] == 0.0
    assert len(sent) == 3
    assert errors == []
    # after exhaustion a replay sends nothing again
    before = len(sent)
    out = log.run_compensation(0, "payout", lambda: sent.append(2.0), target="alice", quantity=2.0, replay_only=True)
    assert out == {"to": "alice", "n": 2}
    assert len(sent) == before


def test_branch_inherits_per_target_confirmed_compensation(tmp_path):
    main = _multi_payee_log(tmp_path, total=9.0)
    main.run_compensation(0, "payout", lambda: {"to": "alice"}, target="alice", quantity=2.0)
    with pytest.raises(ad.ExternalEffectUncertain):
        main.run_compensation(
            0, "payout",
            lambda: (_ for _ in ()).throw(ad.ExternalEffectUncertain("lost")),
            target="bob", quantity=1.0,
        )
    branch = main.create_branch("replan", 0, tmp_path / "branch.jsonl")
    inherited = [e for e in branch.inherited_effects() if e.get("name") == "payout"]
    comp = next(e for e in inherited if e.get("target") == "alice")
    assert comp["quantity_compensated"] == 2.0
    # bob's unconfirmed send is not inherited as an occurrence
    assert not [e for e in inherited if e.get("target") == "bob" and e.get("quantity_pending")]
