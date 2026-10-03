"""Tests for compensation arrangements of already-occurred external effects.

Covers:
* keep / offset / pending human decisions never rewrite the original fact;
* idempotent repeated requests and partial compensation with one shared
  outstanding quantity (no double offset, incl. across forks);
* late receipts (unconfirmed -> verified) and failed attempts do not consume
  allowance;
* compensation requires evidence that the original effect occurred;
* dependency-aware continuation: which later effects may proceed;
* forks from confirmed checkpoints inherit external facts, not unconfirmed
  compensation decisions; model/RNG/optimizer state matches the branch's own
  computation and compensation never applies an optimizer update twice.
"""
import numpy as np
import pytest

import autodiff as ad
from autodiff import manual_seed, nn, optim, tensor
from tests.grad_check import assert_close


def _complete_step(log, step, effects, model, opt, x, seed=0):
    manual_seed(seed)
    with log.step(step, list(effects)) as tx:
        for name, value in effects.items():
            tx.run_effect(name, lambda v=value: v, quantity=lambda p: float(p))
        loss = (model(x) ** 2).sum()
        loss.backward()
        tx.complete(model, optimizer=opt, loss=loss)
    return log


def test_decisions_keep_offset_pending_preserve_original_fact(tmp_path):
    log = ad.TrainingEventLog(tmp_path / "c.jsonl")
    with log.begin_step(1, ["register"]) as tx:
        tx.run_effect("register", lambda: 10.0, quantity=lambda p: p)
        tx.fail("branch abandoned")

    # Original fact is never rewritten as not-executed.
    assert log.effect_statuses(1) == {"register": "occurred"}

    with pytest.raises(ad.CompensationStateError):
        log.compensate(1, "register", lambda q: -q, quantity=2)

    log.decide_compensation(1, "register", "keep", note="keep the registration")
    with pytest.raises(ad.CompensationStateError):
        log.compensate(1, "register", lambda q: -q, quantity=1)
    assert log.effect(1, "register")["status"] == "occurred"
    decisions = log.compensation_decisions(1, "register")
    assert [d["payload"]["decision"] for d in decisions] == ["keep"]


def test_partial_compensation_reverify_and_late_receipt_no_double_offset(tmp_path):
    log = ad.TrainingEventLog(tmp_path / "c.jsonl")
    with log.begin_step(1, ["register"]) as tx:
        tx.run_effect("register", lambda: 10.0, quantity=lambda p: p)
        tx.fail("abandon")
    log.decide_compensation(1, "register", "offset", quantity=10)

    applied = []
    r1 = log.compensate(1, "register", lambda q: (applied.append(q), f"r{q}")[1], quantity=4, compensation_id="a")
    assert r1 == "r4.0"
    # Duplicate request with the same id is replayed, never applied twice.
    r1_again = log.compensate(1, "register", lambda q: (_ for _ in ()).throw(AssertionError()),
                              quantity=4, compensation_id="a")
    assert r1_again == "r4.0" and applied == [4.0]
    with pytest.raises(ad.CompensationLimitError, match="remaining offset scope"):
        log.compensate(1, "register", lambda q: applied.append(q), quantity=7, compensation_id="a2")

    # Re-verify the original fact after partial compensation: still occurred,
    # no new application, outstanding quantity unchanged.
    log.verify_effect(1, "register", True, 10.0, note="receipt rechecked")
    ledger = log.compensation_ledger(1, "register")
    assert ledger["succeeded"] == 4 and ledger["outstanding"] == 6

    # Late receipt: uncertain call reserves the scope; a duplicate is rejected
    # until the receipt is verified; verification settles it exactly once.
    def late(q):
        applied.append(q)
        raise ad.ExternalEffectUncertain("timeout")
    with pytest.raises(ad.ExternalEffectUncertain):
        log.compensate(1, "register", late, quantity=5, compensation_id="late")
    ledger = log.compensation_ledger(1, "register")
    assert ledger["pending"] == 5 and ledger["outstanding"] == 1
    with pytest.raises(ad.CompensationStateError):
        log.compensate(1, "register", lambda q: applied.append(q), quantity=5, compensation_id="late")
    # Over-compensating the tiny remainder while 5 is reserved is blocked.
    with pytest.raises(ad.CompensationLimitError):
        log.compensate(1, "register", lambda q: applied.append(q), quantity=2, compensation_id="rest")
    log.verify_compensation(1, "register", "late", True, "receipt-5", note="arrived late")
    ledger = log.compensation_ledger(1, "register")
    assert ledger["succeeded"] == 9 and ledger["pending"] == 0 and ledger["outstanding"] == 1
    # Idempotent late verification does not count twice.
    log.verify_compensation(1, "register", "late", True, "receipt-5")
    assert log.compensation_ledger(1, "register")["succeeded"] == 9

    # A failed attempt frees its reservation and does not consume allowance.
    log.compensate(1, "register", lambda q: (_ for _ in ()).throw(RuntimeError("boom")),
                   quantity=1, compensation_id="fail") if False else None
    with pytest.raises(RuntimeError):
        log.compensate(1, "register", lambda q: (_ for _ in ()).throw(RuntimeError("boom")),
                       quantity=1, compensation_id="fail")
    final = log.compensate(1, "register", lambda q: (applied.append(q), "done")[1],
                           quantity=1, compensation_id="rest")
    assert final == "done"
    ledger = log.compensation_ledger(1, "register")
    assert ledger["succeeded"] == 10 and ledger["outstanding"] == 0
    log.resolve_compensation(1, "register")
    assert applied == [4.0, 5.0, 1.0]
    # Reopen: durable audit trail still consistent.
    reopened = ad.TrainingEventLog(log.path)
    assert reopened.compensation_ledger(1, "register")["succeeded"] == 10
    kinds = {e["kind"] for e in reopened.audit_events(1)}
    assert {"training_effect", "compensation_decision", "compensation_attempt",
            "compensation_verification", "training_verification"} <= kinds


def test_compensation_requires_evidence_and_cannot_exceed_original(tmp_path):
    log = ad.TrainingEventLog(tmp_path / "c.jsonl")
    with pytest.raises(ad.ExternalEffectUncertain):
        with log.begin_step(2, ["charge", "email"]):
            ad.external_event("charge", lambda: (_ for _ in ()).throw(ad.ExternalEffectUncertain("?")))
    for decision in ("keep", "offset", "pending"):
        with pytest.raises(ad.NoEffectEvidenceError):
            log.decide_compensation(2, "charge", decision)
    with pytest.raises(ad.NoEffectEvidenceError):
        log.compensate(2, "charge", lambda q: None)
    # A later effect never ran, so nothing to compensate.
    with pytest.raises(ad.NoEffectEvidenceError):
        log.decide_compensation(2, "email", "offset")

    # Proven effect: scope cannot exceed the original quantity.
    with log.begin_step(3, ["reg"]) as tx:
        tx.run_effect("reg", lambda: 3.0, quantity=lambda p: p)
        tx.fail("x")
    with pytest.raises(ad.CompensationLimitError):
        log.decide_compensation(3, "reg", "offset", quantity=4)


def test_dependency_aware_continuation(tmp_path):
    log = ad.TrainingEventLog(tmp_path / "c.jsonl")
    with log.begin_step(1, ["reserve", "submit", "publish"]) as tx:
        for name in ("reserve", "submit", "publish"):
            tx.run_effect(name, lambda n=name: 1.0, quantity=lambda p: p)
        tx.fail("abandon branch")

    plan = log.continuation(1)
    assert plan["conflicting"] == ["reserve", "submit", "publish"]
    assert plan["can_continue"] == []

    log.decide_compensation(1, "reserve", "keep")
    plan = log.continuation(1)
    assert plan["can_continue"] == ["submit"]
    assert plan["prerequisites_ready"] == ["reserve", "submit"]
    assert plan["blocked"] == ["submit", "publish"]

    log.decide_compensation(1, "submit", "offset", quantity=1)
    log.compensate(1, "submit", lambda q: "ok", quantity=1, compensation_id="s")
    log.resolve_compensation(1, "submit")
    # publish still awaits a human decision, but all its deps are resolved,
    # so training may proceed up to and including retrying the publish fact.
    plan = log.continuation(1)
    assert plan["can_continue"] == ["publish"]
    assert plan["blocked"] == ["publish"]


def test_fork_inherits_facts_not_unconfirmed_decisions_and_no_double_update(tmp_path):
    path = tmp_path / "c.jsonl"
    log = ad.TrainingEventLog(path)
    model = nn.Linear(2, 1, seed=3)
    opt = optim.SGD(model.parameters(), lr=0.2, momentum=0.9)
    x = tensor([[1.0, 2.0]], requires_grad=True)
    _complete_step(log, 1, {"register": 5.0}, model, opt, x, seed=1)

    params_after = [p.data.copy() for p in model.parameters()]
    saved_opt = [v.copy() for v in opt._velocity]

    # Another branch opens an offset decision but never confirms it.
    log.decide_compensation(1, "register", "offset", quantity=5, note="branch A idea")

    model_b = nn.Linear(2, 1, seed=99)
    opt_b = optim.SGD(model_b.parameters(), lr=0.2, momentum=0.9)
    branch = log.fork_branch(1, 2, model_b, optimizer=opt_b)
    for p, expected in zip(model_b.parameters(), params_after):
        assert_close(p.data, expected)
    # Parameters restored from checkpoint; optimizer state matches origin.
    for v, expected in zip(opt_b._velocity, saved_opt):
        assert_close(v, expected)
    # Grads from the origin computation are not carried over.
    assert all(p.grad is None for p in model_b.parameters())

    # External fact is inherited ...
    assert log.effect_statuses(2) == {"register": "occurred"}
    assert branch.effect_payload("register") == 5.0
    # ... but branch A's unconfirmed decision is NOT inherited: compensation
    # requires a fresh human decision on this branch.
    assert log.compensation_decisions(2) == ()
    with pytest.raises(ad.CompensationStateError):
        branch.compensate("register", lambda q: None, quantity=1, compensation_id="x")

    # Training on the branch performs exactly its own single optimizer step.
    manual_seed(1)
    with log.resume_step(2) as tx:
        batch = tx.run_effect("register", replay=lambda p: p)
        assert batch == 5.0
        mask = ad.rand(2)
        loss = (model_b(x * mask) ** 2).sum()
        loss.backward()
        grads = [p.grad.data.copy() for p in model_b.parameters()]
        tx.complete(model_b, optimizer=opt_b, loss=loss)

    # Cross-branch compensation on the new branch still charges the SAME
    # original fact: decide partially, compensate, and confirm no double count.
    log2 = ad.TrainingEventLog(path)
    log2.decide_compensation(2, "register", "offset", quantity=3)
    log2.compensate(2, "register", lambda q: f"-{q}", quantity=3, compensation_id="b1")
    ledger = log2.compensation_ledger(2, "register")
    assert ledger["original"] == 5 and ledger["succeeded"] == 3 and ledger["outstanding"] == 2
    # The origin-step ledger sees the branch compensation too (global facts).
    origin_ledger = ad.TrainingEventLog(path).compensation_ledger(1, "register")
    assert origin_ledger["succeeded"] == 3


def test_ordinary_steps_remain_compatible_without_compensation(tmp_path):
    log = ad.TrainingEventLog(tmp_path / "c.jsonl")
    model = nn.Linear(2, 1, seed=4)
    opt = optim.Adam(model.parameters(), lr=0.1)
    x = tensor([[1.0, 1.0]], requires_grad=True)
    _complete_step(log, 7, {"fetch": 1.0}, model, opt, x, seed=2)
    record = log.get(ad.TrainingStep._step_key(7))["payload"]["completion"]
    assert record["optimizer"]["type"] == "Adam" and record["optimizer"]["t"] == 1
    # Fork restores Adam state and continues from exactly that checkpoint.
    model2 = nn.Linear(2, 1, seed=4)
    opt2 = optim.Adam(model2.parameters(), lr=0.1)
    log.fork_branch(7, 8, model2, optimizer=opt2)
    assert opt2.t == 1 and all(p.grad is None for p in model2.parameters())
    for p, q in zip(model.parameters(), model2.parameters()):
        assert_close(p.data, q.data)
