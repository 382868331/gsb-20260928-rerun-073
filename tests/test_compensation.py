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

    # quantity is clamped to the remaining scope: no double offset
    log.run_compensation(0, "register", lambda: calls.append(1) or {"offset": 2}, quantity=5.0)
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
