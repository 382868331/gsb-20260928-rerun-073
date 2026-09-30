"""Tests for gradient checkpointing (activation recomputation).

Covers:
* long-sequence recomputation vs fully-retained control (loss + grads);
* different checkpoint intervals give identical results;
* random ops (dropout) resample identically on recompute and do not
  advance the global RNG a second time (no duplicated side effects);
* shared parameters / multi-branch gradient accumulation;
* in-place parameter modification still invalidates checkpointed graphs;
* continuous training + save/restore never mixes stale graphs.
"""
import numpy as np
import pytest

import autodiff as ad
from autodiff import Tensor, checkpoint, checkpoint_sequential, manual_seed, nn, optim, tensor
from autodiff.core import _rng_state
from tests.grad_check import assert_close


def make_deep(n_layers=48, dim=8, seed=0):
    layers = []
    for i in range(n_layers):
        layers.append(nn.Linear(dim, dim, seed=seed * 1000 + i))
        layers.append(nn.ReLU())
    return layers


def run_full(layers, x):
    h = x
    for layer in layers:
        h = layer(h)
    return h


def grads_of(params):
    return [p.grad.data.copy() for p in params]


# ------------------------------------------------------- interval sweeps
@pytest.mark.parametrize("segments", [1, 2, 5, 16, 96])
def test_checkpoint_intervals_match_full_retain(segments):
    x = tensor(np.random.default_rng(7).standard_normal((4, 8)), requires_grad=True)

    layers_ref = make_deep()
    loss_ref = run_full(layers_ref, x).sum()
    loss_ref.backward()
    ref = grads_of([p for l in layers_ref if isinstance(l, nn.Linear) for p in (l.weight, l.bias)])
    ref_x = x.grad.data.copy()

    x.grad = None
    layers_ck = make_deep()
    loss_ck = checkpoint_sequential(layers_ck, x, segments=segments).sum()
    loss_ck.backward()
    got = grads_of([p for l in layers_ck if isinstance(l, nn.Linear) for p in (l.weight, l.bias)])

    assert_close(loss_ck.data, loss_ref.data)
    assert_close(x.grad.data, ref_x)
    for g0, g1 in zip(got, ref):
        assert_close(g1, g0)


def test_short_sequence_full_retain_control():
    # Short sequence kept fully resident is the control; checkpointing the
    # same short sequence must reproduce it exactly.
    x = tensor(np.random.default_rng(3).standard_normal((2, 4)), requires_grad=True)
    w = tensor(np.random.default_rng(4).standard_normal((4, 4)), requires_grad=True)

    full = ad.sum(ad.tanh(x @ w) ** 2)
    full.backward()
    gx, gw = x.grad.data.copy(), w.grad.data.copy()
    x.grad = w.grad = None

    ck = checkpoint(lambda t: ad.sum(ad.tanh(t @ w) ** 2), x)
    ck.backward()
    assert_close(x.grad.data, gx)
    assert_close(w.grad.data, gw)


# ------------------------------------------------- rng / side effects
def test_dropout_recompute_same_mask_and_no_extra_rng_consumption():
    manual_seed(123)
    x = tensor(np.random.default_rng(0).standard_normal((6, 8)), requires_grad=True)
    w = tensor(np.random.default_rng(1).standard_normal((8, 8)) * 0.2, requires_grad=True)

    drop = nn.Dropout(p=0.4)

    def seg(t):
        return drop(t @ w)

    out = checkpoint(seg, x)
    state_after_forward = _rng_state()
    loss = (out ** 2).sum()
    loss.backward()
    state_after_backward = _rng_state()
    # Recompute must not advance the global RNG a second time.
    assert state_after_backward == state_after_forward

    # Compare against a fully-retained forward seeded identically.
    manual_seed(123)
    x2 = tensor(x.data, requires_grad=True)
    w2 = tensor(w.data, requires_grad=True)
    drop2 = nn.Dropout(p=0.4) if False else nn.Dropout(p=0.4)
    loss2 = (drop2(x2 @ w2) ** 2).sum()
    loss2.backward()
    assert_close(loss.data, loss2.data)
    assert_close(x.grad.data, x2.grad.data)
    assert_close(w.grad.data, w2.grad.data)


def test_repeated_backward_recomputes_consistently():
    manual_seed(9)
    x = tensor(np.random.default_rng(5).standard_normal((3, 4)), requires_grad=True)
    drop = nn.Dropout(p=0.3)
    out = checkpoint(lambda t: drop(t * 2.0), x).sum()
    out.backward()
    g1 = x.grad.data.copy()
    out.backward()  # accumulate semantics, recompute draws identical mask
    assert_close(x.grad.data, 2 * g1)


# ------------------------------------------- shared params / branches
def test_shared_param_across_segments_and_branches():
    w = tensor(np.random.default_rng(11).standard_normal((4, 4)), requires_grad=True)
    x = tensor(np.random.default_rng(12).standard_normal((3, 4)), requires_grad=True)

    def branch(t):
        return ad.tanh(t @ w) @ w  # w used twice inside one segment

    a = checkpoint(branch, x)
    b = checkpoint(branch, x)  # same w shared across two segments
    loss = (a * b).sum() + (a @ w).sum()
    loss.backward()
    got = w.grad.data.copy()

    w2 = tensor(w.data, requires_grad=True)
    x2 = tensor(x.data, requires_grad=True)
    a2 = ad.tanh(x2 @ w2) @ w2
    b2 = ad.tanh(x2 @ w2) @ w2
    loss2 = (a2 * b2).sum() + (a2 @ w2).sum()
    loss2.backward()
    assert_close(got, w2.grad.data)
    assert_close(x.grad.data, x2.grad.data)


# ------------------------------------------- staleness / invalidation
def test_inplace_modification_invalidates_checkpointed_graph():
    w = tensor(np.random.default_rng(2).standard_normal((4, 4)), requires_grad=True)
    x = tensor(np.random.default_rng(3).standard_normal((3, 4)), requires_grad=True)
    loss = checkpoint(lambda t, w: ad.tanh(t @ w), x, w).sum()
    w.mul_(1.01)  # in-place update after forward
    with pytest.raises(RuntimeError, match="modified in place"):
        loss.backward()


def test_retain_graph_false_frees_checkpoint():
    x = tensor(np.random.default_rng(6).standard_normal((2, 2)), requires_grad=True)
    loss = checkpoint(lambda t: (t * t).sum(), x)
    loss.backward(retain_graph=False)
    with pytest.raises(RuntimeError):
        loss.backward()


# ------------------------------------------- training + save/restore
def test_continuous_training_and_state_restore():
    manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 16, seed=1), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 3, seed=2))
    opt = optim.Adam(model.parameters(), lr=0.05)
    rng = np.random.default_rng(42)
    X, y = rng.standard_normal((32, 4)), rng.integers(0, 3, 32)

    losses = []
    stale_loss = None
    for step in range(30):
        opt.zero_grad()
        loss = checkpoint(lambda t: nn.softmax_ce_loss(model(t), y), tensor(X), requires_grad=True)
        loss.backward()
        opt.step()  # in-place update invalidates this step graph
        losses.append(loss.data)
        if step == 10:
            stale_loss = loss
            snapshot = model.state_dict()
    assert losses[-1] < losses[0]

    # Graph from an earlier step must not be reusable after param updates.
    with pytest.raises(RuntimeError, match="modified in place"):
        stale_loss.backward()

    # Save/restore: loading bumps versions, so the pre-load graph is dead too.
    loss_now = checkpoint(lambda t: nn.softmax_ce_loss(model(t), y), tensor(X), requires_grad=True)
    model.load_state_dict(snapshot)
    with pytest.raises(RuntimeError, match="modified in place"):
        loss_now.backward()

    # After restore, a fresh forward reproduces the snapshot behaviour.
    manual_seed(0)
    fresh = checkpoint(lambda t: nn.softmax_ce_loss(model(t), y), tensor(X), requires_grad=True).data
    manual_seed(0)
    model2 = nn.Sequential(nn.Linear(4, 16, seed=1), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 3, seed=2))
    model2.load_state_dict(snapshot)
    fresh2 = checkpoint(lambda t: nn.softmax_ce_loss(model2(t), y), tensor(X), requires_grad=True).data
    assert_close(np.asarray(fresh), np.asarray(fresh2))


def test_checkpoint_under_no_grad_is_plain_value():
    x = tensor([1.0, 2.0], requires_grad=True)
    with ad.no_grad():
        out = checkpoint(lambda t: t * 2, x)
    assert not out.requires_grad
    assert_close(out.data, np.array([2.0, 4.0]))


def test_checkpoint_rejects_non_tensor_output():
    x = tensor([1.0], requires_grad=True)
    with pytest.raises(TypeError):
        checkpoint(lambda t: 3.14, x)
