"""Tests for external-effect handling in recomputation intervals.

Covers:
* external effects happen once: checkpoint replay and repeated backward do
  not re-invoke the user callback;
* replay records are distinguished from user-visible events;
* unsupported external operations are rejected *before* producing an effect;
* shared parameters + random ops + multiple backwards stay correct;
* durable journals survive restart (no double submit), pending effects
  require explicit reconciliation and are never erased, and a failed step
  rollback cannot wipe a real effect from the journal.
"""
import json

import numpy as np
import pytest

import autodiff as ad
from autodiff import (
    EventLog, PendingExternalEventError, ReplayExternalEventError,
    checkpoint, is_replaying, manual_seed, nn, record_external_event,
    tensor, use_event_log,
)
from tests.grad_check import assert_close


def test_external_effect_runs_once_but_replay_sees_same_value(tmp_path):
    calls = []

    def side_effect():
        calls.append(1)
        return np.array([3.0, -1.0])

    x = tensor([1.0, 1.0], requires_grad=True)
    log = EventLog(tmp_path / "events.jsonl")

    def seg(t):
        bias = np.asarray(record_external_event("fetch-bias", side_effect))
        return (t * 2.0 + tensor(bias)) ** 2

    with use_event_log(log):
        out = checkpoint(seg, x).sum()
    assert calls == [1]  # happened exactly once during forward

    out.backward()  # recomputes the interval; callback must not run again
    out.backward()  # a second replay also must not run it again
    assert calls == [1]

    assert [e["id"] for e in log.events()] == ["fetch-bias"]
    assert all(e["status"] == "committed" for e in log.events())
    assert [r["event_id"] for r in log.replays()] == ["fetch-bias", "fetch-bias"]
    assert_close(x.grad.data, 2 * np.array([2 * 2.0 * (2 + 3), 2 * 2.0 * (2 - 1)]))


def test_unregistered_external_op_rejected_before_effect_on_replay(tmp_path):
    attempted = []

    def dangerous():
        attempted.append(1)  # must never be reached during replay
        return 1.0

    x = tensor([0.5], requires_grad=True)
    log = EventLog(tmp_path / "events.jsonl")

    def seg(t):
        value = record_external_event("e1", lambda: 7.0)
        if is_replaying():
            # A second, distinct external op never seen during the forward:
            # replay cannot know its effect, so it must refuse up front.
            record_external_event("e2", dangerous)
        return t * value

    with use_event_log(log):
        out = checkpoint(seg, x).sum()
    with pytest.raises(ReplayExternalEventError, match="e2"):
        out.backward()
    assert attempted == []
    assert not log.is_committed("e2")


def test_external_op_without_log_rejected_before_effect():
    attempted = []
    x = tensor([0.5], requires_grad=True)

    def seg(t):
        value = record_external_event("network-write", lambda: attempted.append(1) or 2.0)
        return t * value

    out = checkpoint(seg, x).sum()
    assert attempted == [1]  # outside any log it is a normal call
    with pytest.raises(ReplayExternalEventError, match="network-write"):
        out.backward()
    assert attempted == [1]  # replay refused to duplicate the write


def test_shared_params_randomness_and_multiple_backward(tmp_path):
    manual_seed(2024)
    w = tensor(np.random.default_rng(0).standard_normal((4, 4)) * 0.3, requires_grad=True)
    x = tensor(np.random.default_rng(1).standard_normal((3, 4)), requires_grad=True)
    drop = nn.Dropout(0.35)
    log = EventLog(tmp_path / "events.jsonl")

    def seg(t):
        scale = float(record_external_event("scale", lambda: 1.5))
        return drop(t @ w) * scale @ w + drop(t * scale) @ w

    with use_event_log(log):
        a = checkpoint(seg, x)
        b = checkpoint(seg, x)
    # Reusing the same id outside replay is the idempotent fast path: the
    # committed result comes back without invoking the new callback.
    attempts = []
    with use_event_log(log):
        assert record_external_event("scale", lambda: attempts.append(1) or 9.9) == 1.5
    assert attempts == []
    loss = (a * b).sum() + a.sum()
    loss.backward()
    loss.backward()  # accumulation; replays must reproduce masks exactly

    # Fully-retained reference: rebuild the same forward by replaying RNG.
    manual_seed(2024)
    w2 = tensor(w.data, requires_grad=True)
    x2 = tensor(x.data, requires_grad=True)
    drop2 = nn.Dropout(0.35)
    a2 = drop2(x2 @ w2) * 1.5 @ w2 + drop2(x2 * 1.5) @ w2
    b2 = drop2(x2 @ w2) * 1.5 @ w2 + drop2(x2 * 1.5) @ w2
    loss2 = (a2 * b2).sum() + a2.sum()
    loss2.backward()
    loss2.backward()
    assert_close(w.grad.data, w2.grad.data)
    assert_close(x.grad.data, x2.grad.data)
    # Four replays (two checkpoint calls x two backwards), each user-visible
    # effect committed exactly once.
    assert len(log.replays()) == 4
    assert sorted(e["id"] for e in log.events()) == ["scale"]


def test_journal_reopen_does_not_resubmit_committed_event(tmp_path):
    path = tmp_path / "events.jsonl"
    calls = []
    log = EventLog(path)
    with use_event_log(log):
        value = record_external_event("step-0", lambda: calls.append(1) or 42)
    assert value == 42

    # Simulate a fresh process: reopening the journal and rerunning the same
    # training step returns the recorded effect without calling the callback.
    reopened = EventLog(path)
    with use_event_log(reopened):
        again = record_external_event("step-0", lambda: calls.append(1) or 99)
    assert again == 42
    assert calls == [1]

    raw = [json.loads(line) for line in path.read_text().splitlines()]
    statuses = [(e["type"], e.get("status")) for e in raw if e["type"] == "event"]
    assert statuses == [("event", "submitted"), ("event", "committed")]


def test_failed_effect_keeps_submitted_record_and_requires_reconciliation(tmp_path):
    path = tmp_path / "events.jsonl"

    def flaky():
        raise ConnectionError("crashed after the remote side charged a fee")

    log = EventLog(path)
    with pytest.raises(ConnectionError):
        log.record("payment", flaky)

    # The submitted intent is durable; nothing is erased or rolled back.
    raw = [json.loads(line) for line in path.read_text().splitlines()]
    assert raw[-1]["status"] == "submitted"

    # A naive reopen must not blindly retry (the fee may already be charged).
    with pytest.raises(PendingExternalEventError, match="payment"):
        EventLog(path)

    # After manually confirming the remote system, reconcile explicitly.
    log.resolve_pending("payment", result={"charged": True})
    reopened = EventLog(path)
    assert reopened.result("payment") == {"charged": True}
    assert reopened.events()[-1].get("reconciled") is True
    with use_event_log(reopened):
        assert record_external_event("payment", flaky) == {"charged": True}


def test_model_state_and_event_log_resume_training(tmp_path):
    journal = tmp_path / "events.jsonl"
    manual_seed(7)
    model = nn.Sequential(nn.Linear(4, 8, seed=1), nn.ReLU(), nn.Dropout(0.2), nn.Linear(8, 3, seed=2))
    opt = ad.optim.Adam(model.parameters(), lr=0.05)
    rng = np.random.default_rng(11)
    X, y = rng.standard_normal((16, 4)), rng.integers(0, 3, 16)

    fetched = []

    def train_some(num_steps, start):
        log = EventLog(journal)
        for step in range(start, start + num_steps):
            opt.zero_grad()
            with use_event_log(log):
                loss = checkpoint(
                    lambda t: nn.softmax_ce_loss(
                        model(t)
                        + float(record_external_event(f"step-{step}-bias", lambda: fetched.append(step) or 0.01)),
                        y,
                    ), tensor(X),
                    requires_grad=True,
                )
            loss.backward()
            opt.step()
        return log

    train_some(5, 0)
    snapshot = model.state_dict()
    calls_after_first_phase = len(fetched)

    # "Restart": reopen the journal, reload parameters, and redo the same
    # steps (e.g. an at-least-once scheduler). Effects must not duplicate.
    model.load_state_dict(snapshot)
    log = train_some(5, 0)
    assert len(fetched) == calls_after_first_phase
    assert [e["id"] for e in log.events()] == [f"step-{s}-bias" for s in range(5)]
    # Five replays from each phase (the reopened log keeps the first five).
    assert len(log.replays()) == 10
