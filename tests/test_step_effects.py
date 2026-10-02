"""Step-scoped dependent external effects: failure classification, user
verification, resume, and consistency with gradients / model state."""
from __future__ import annotations

import json

import numpy as np
import pytest

import autodiff as ad
from autodiff import nn, optim
from autodiff.events import (
    EventExistsError,
    StepEffectError,
    UnconfirmedEffectError,
)


def make_model():
    ad.manual_seed(0)
    return nn.Sequential(nn.Linear(3, 8, seed=11), nn.ReLU(), nn.Linear(8, 2, seed=22))


def make_opt(model):
    return optim.SGD(model.parameters(), lr=0.05)


BATCH_X = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
BATCH_Y = [0, 1]


def run_step(log, model, opt, step_id, effects, backwards=1):
    """One training step whose effects may be supplied by test doubles."""
    with log.step(step_id, plan=["fetch", "confirm", "submit"]) as step:
        x = step.effect("fetch", effects["fetch"], lambda p: ad.tensor(np.array(p["x"])))
        ok = step.effect("confirm", effects["confirm"], lambda p: bool(p))
        if ok:
            step.effect("submit", effects["submit"], lambda p: p)
        loss = nn.softmax_ce_loss(model(x), BATCH_Y)
        for _ in range(backwards):
            loss.backward()
        opt.step()
        opt.zero_grad()
    return step


def good_effects(calls):
    return {
        "fetch": lambda: (calls.append("fetch"), {"x": BATCH_X})[1],
        "confirm": lambda: (calls.append("confirm"), True)[1],
        "submit": lambda: (calls.append("submit"), "ack")[1],
    }


def params_of(model):
    return [p.data.copy() for p in model.parameters()]


def test_failure_report_classifies_effects(tmp_path):
    log = ad.TrainingEventLog(tmp_path / "events.jsonl")
    model, opt = make_model(), None
    opt = make_opt(model)
    calls = []

    def boom():
        calls.append("confirm")
        raise IOError("gateway timeout after request sent")

    effects = good_effects(calls)
    effects["confirm"] = boom
    with pytest.raises(StepEffectError) as info:
        run_step(log, model, opt, "s0", effects)

    report = info.value.report
    assert report["applied"] == ["fetch"]
    assert report["unconfirmed"] == ["confirm"]
    assert report["pending"] == ["submit"]
    assert "gateway timeout" in report["reason"]

    # Facts are durable: the applied effect stays, the failed attempt keeps
    # its original error, and the step is NOT saved as completed.
    assert log.get("effect/s0/fetch")["status"] == "applied"
    failed = log.get("effect/s0/confirm")
    assert failed["status"] == "unconfirmed"
    assert "gateway timeout" in failed["error"]
    assert "step/s0" not in log
    interruptions = [e for e in log.events if e["id"].startswith("step/s0/interrupted/")]
    assert len(interruptions) == 1
    assert interruptions[0]["payload"]["report"]["applied"] == ["fetch"]


def test_resume_requires_verification_then_reapplies_once(tmp_path):
    path = tmp_path / "events.jsonl"
    log = ad.TrainingEventLog(path)
    model, opt = make_model(), make_opt(make_model())
    model = make_model()
    opt = make_opt(model)
    calls = []

    def boom():
        calls.append("confirm")
        raise IOError("timeout")

    effects = good_effects(calls)
    effects["confirm"] = boom
    with pytest.raises(StepEffectError):
        run_step(log, model, opt, "s0", effects)
    assert calls == ["fetch", "confirm"]

    # Mid-way restart: a fresh log object on the same file. Applied effects
    # replay without re-executing; the unconfirmed one blocks the resume.
    log2 = ad.TrainingEventLog(path)
    with pytest.raises(UnconfirmedEffectError) as info:
        run_step(log2, model, opt, "s0", good_effects(calls))
    assert info.value.event_id == "effect/s0/confirm"
    assert calls == ["fetch", "confirm"]  # nothing re-executed

    # User verifies the effect did NOT happen; only then may it re-apply.
    record = log2.verify("effect/s0/confirm", False, note="gateway log shows 0 requests")
    again = log2.verify("effect/s0/confirm", False, note="gateway log shows 0 requests")
    assert again is record or again == record  # idempotent, no duplicate effect
    with pytest.raises(EventExistsError):
        log2.verify("effect/s0/confirm", True, payload=True)

    effects2 = good_effects(calls)
    step = run_step(log2, model, opt, "s0", effects2, backwards=2)
    assert step.completed
    # fetch replayed (still 1 call), confirm re-applied exactly once as a
    # linked second attempt, submit applied once.
    assert calls == ["fetch", "confirm", "confirm", "submit"]
    assert log2.get("effect/s0/confirm#2")["attempt"] == 2
    assert log2.get("step/s0")["payload"]["status"] == "completed"

    # Verification trail is traceable.
    verification = log2.get("verify/effect/s0/confirm")
    assert verification["payload"]["occurred"] is False
    assert "gateway log" in verification["payload"]["note"]


def test_verified_occurred_replays_without_reapply(tmp_path):
    path = tmp_path / "events.jsonl"
    log = ad.TrainingEventLog(path)
    model = make_model()
    opt = make_opt(model)
    calls = []

    def boom():
        calls.append("confirm")
        raise IOError("response lost")

    effects = good_effects(calls)
    effects["confirm"] = boom
    with pytest.raises(StepEffectError):
        run_step(log, model, opt, "s0", effects)

    log.verify("effect/s0/confirm", True, payload=True, note="provider shows receipt")
    calls.clear()
    step = run_step(ad.TrainingEventLog(path), model, opt, "s0", good_effects(calls))
    assert step.completed
    assert calls == ["submit"]  # confirm not re-applied; fetch replayed


def test_gradients_and_params_match_reference_round(tmp_path):
    # Reference: one clean training round without any event machinery.
    ref_model = make_model()
    ref_opt = make_opt(ref_model)
    x = ad.tensor(np.array(BATCH_X))
    loss = nn.softmax_ce_loss(ref_model(x), BATCH_Y)
    loss.backward()
    loss.backward()  # accumulated twice, like the event run
    ref_opt.step()

    path = tmp_path / "events.jsonl"
    log = ad.TrainingEventLog(path)
    model = make_model()
    opt = make_opt(model)
    init_params = params_of(model)
    calls = []

    def boom():
        calls.append("confirm")
        raise IOError("flaky")

    effects = good_effects(calls)
    effects["confirm"] = boom
    with pytest.raises(StepEffectError):
        run_step(log, model, opt, "s0", effects, backwards=2)
    # Partial effects happened but computation did not finish: no completed
    # step, and parameters must be untouched by the failed attempt.
    assert "step/s0" not in log
    for got, want in zip(params_of(model), init_params):
        np.testing.assert_array_equal(got, want)
    log.verify("effect/s0/confirm", False)

    # Restart interleaved with random ops and a doubled backward.
    ad.manual_seed(123)
    _ = ad.randn(4)  # unrelated random op between attempts
    log2 = ad.TrainingEventLog(path)
    step = run_step(log2, model, opt, "s0", good_effects(calls), backwards=2)
    assert step.completed
    for got, want in zip(params_of(model), params_of(ref_model)):
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)

    # A completed step refuses further effects: its gradients, parameters
    # and records belong to one finished training round.
    with pytest.raises(EventExistsError):
        run_step(log2, model, opt, "s0", good_effects(calls))


def test_conditional_effect_chain(tmp_path):
    # confirm decides whether submit runs at all; a False confirmation must
    # leave submit pending (never executed), not failed.
    log = ad.TrainingEventLog(tmp_path / "events.jsonl")
    model = make_model()
    opt = make_opt(model)
    calls = []
    effects = good_effects(calls)
    effects["confirm"] = lambda: (calls.append("confirm"), False)[1]
    step = run_step(log, model, opt, "s0", effects)
    assert step.completed
    assert calls == ["fetch", "confirm"]
    report = log.step_report("s0", ["fetch", "confirm", "submit"])
    assert report["applied"] == ["fetch", "confirm"]
    assert report["pending"] == ["submit"]
