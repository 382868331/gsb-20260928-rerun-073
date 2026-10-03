"""Abandon a training branch, settle its real external effects, then fork.

Run: ``python examples/compensation_branch.py``

The scenario:

1. step 1 registers a result externally (quantity 5) and completes a real
    optimizer update;
2. the branch is abandoned; a human keeps one dependent fact and arranges a
    partial offset for another, including a late compensation receipt;
3. step 2 is forked from the confirmed checkpoint: it inherits the external
    facts and the post-update model/optimizer state, but not the abandoned
    branch's unconfirmed compensation decisions.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import autodiff as ad
from autodiff import nn, optim, tensor


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log = ad.TrainingEventLog(Path(tmp) / "events.jsonl")
        model = nn.Sequential(nn.Linear(3, 4, seed=1), nn.ReLU(), nn.Linear(4, 1, seed=2))
        opt = optim.Adam(model.parameters(), lr=0.05)
        x = tensor(np.ones((2, 3)), requires_grad=True)
        y = tensor(np.ones((2, 1)))

        with log.begin_step(1, ["register", "notify"]) as step:
            step.run_effect("register", lambda: 5.0, quantity=lambda q: q)
            step.run_effect("notify", lambda: 1.0, quantity=lambda q: q)
            loss = nn.mse_loss(model(x), y)
            loss.backward()
            step.complete(model, optimizer=opt, loss=loss)
        print("step 1 completed; external register=5, notify=1")

        # The branch is given up.  The facts stay on the audit log; the human
        # decides what to do about each one.
        log.decide_compensation(1, "notify", "keep", note="notification is harmless")
        log.decide_compensation(1, "register", "offset", quantity=5, note="cancel the registration")

        log.compensate(1, "register", lambda q: f"offset {q}", quantity=3, compensation_id="part-1")
        print("partial offset 3/5 applied")

        # A late receipt: request timed out after possibly reaching the system.
        try:
            log.compensate(
                1, "register",
                lambda q: (_ for _ in ()).throw(ad.ExternalEffectUncertain("timeout")),
                quantity=2, compensation_id="part-2",
            )
        except ad.ExternalEffectUncertain:
            pass
        plan = log.continuation(1)
        print("while waiting for the receipt, continuation:", plan["can_continue"])
        log.verify_compensation(1, "register", "part-2", True, "offset 2", note="receipt found")
        log.resolve_compensation(1, "register")
        print("register fully compensated; ledger:", log.compensation_ledger(1, "register"))

        # Fork from the confirmed computation checkpoint.
        model2 = nn.Sequential(nn.Linear(3, 4, seed=7), nn.ReLU(), nn.Linear(4, 1, seed=8))
        opt2 = optim.Adam(model2.parameters(), lr=0.05)
        branch = log.fork_branch(1, 2, model2, optimizer=opt2)
        print("forked to step 2; inherited statuses:", branch.statuses())
        print("inherited decisions on the new branch:", log.compensation_decisions(2))

        with log.resume_step(2) as step2:
            register = step2.run_effect("register", replay=lambda payload: payload)
            loss = nn.mse_loss(model2(x) * register, y)
            loss.backward()
            step2.complete(model2, optimizer=opt2, loss=loss)
        print("step 2 ran its own single optimizer update; final state:", log.step_state(2))


if __name__ == "__main__":
    main()
