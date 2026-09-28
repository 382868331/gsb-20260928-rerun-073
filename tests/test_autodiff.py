import numpy as np
import pytest

import autodiff as ad
from autodiff import Tensor, nn, optim, no_grad, tensor
from autodiff.nn import mse_loss, softmax_ce_loss, binary_cross_entropy_with_logits
from tests.grad_check import numerical_grad, assert_close


# ------------------------------------------------------------- scalar basics
def test_add_mul_div_pow_scalar():
    x = tensor(2.0, requires_grad=True)
    y = tensor(3.0, requires_grad=True)
    out = ((x + y) * y - x / y) ** 2 + ad.exp(x - 1)
    out.backward()

    def f(a):
        xx, yy = a[0], a[1]
        return np.array(((xx + yy) * yy - xx / yy) ** 2 + np.exp(xx - 1))

    num = numerical_grad(f, np.array([2.0, 3.0]))
    assert_close(np.array([x.grad.data]), num[0:1])
    assert_close(np.array([y.grad.data]), num[1:2])


def test_chain_of_unary_ops():
    x = ad.randn(4, 3, seed=0, requires_grad=True)
    out = ad.sum(ad.tanh(ad.relu(x)) + ad.sigmoid(x * 0.5) + ad.log(ad.relu(x) + 1))
    out.backward()
    num = numerical_grad(
        lambda a: np.tanh(np.maximum(a, 0)) + 1.0 / (1 + np.exp(-a * 0.5))
        + np.log(np.maximum(a, 0) + 1),
        x.data.copy(),
    )
    assert_close(x.grad.data, num)


# ------------------------------------------------------------ broadcasting
def test_broadcast_matmul_affine():
    W = ad.randn(5, 2, seed=1, requires_grad=True)
    b = ad.randn(2, seed=2, requires_grad=True)
    x = ad.randn(7, 5, seed=3, requires_grad=True)
    out = (x @ W + b).sum()
    out.backward()

    def fw(a):
        xx = a[:35].reshape(7, 5)
        ww = a[35:45].reshape(5, 2)
        bb = a[45:47]
        return (xx @ ww + bb).sum()

    flat = np.concatenate([x.data.ravel(), W.data.ravel(), b.data.ravel()])
    num = numerical_grad(fw, flat)
    assert_close(x.grad.data.ravel(), num[:35], rtol=1e-3)
    assert_close(W.grad.data.ravel(), num[35:45], rtol=1e-3)
    assert_close(b.grad.data.ravel(), num[45:47], rtol=1e-3)


def test_broadcast_shapes_and_unsqueeze_like():
    # (3, 1, 4) + (2, 4) -> (3, 2, 4); multiply by (4,)
    a = ad.randn(3, 1, 4, seed=4, requires_grad=True)
    b = ad.randn(2, 4, seed=5, requires_grad=True)
    c = ad.randn(4, seed=6, requires_grad=True)
    loss = ((a + b) * c).sum()
    loss.backward()

    def fw(vec):
        aa = vec[:12].reshape(3, 1, 4)
        bb = vec[12:20].reshape(2, 4)
        cc = vec[20:]
        return ((aa + bb) * cc).sum()

    vec = np.concatenate([a.data.ravel(), b.data.ravel(), c.data.ravel()])
    num = numerical_grad(fw, vec)
    assert_close(a.grad.data.ravel(), num[:12], rtol=1e-3)
    assert_close(b.grad.data.ravel(), num[12:20], rtol=1e-3)
    assert_close(c.grad.data.ravel(), num[20:], rtol=1e-3)


def test_broadcast_incompatible_raises():
    with pytest.raises(ValueError):
        ad.broadcast_shapes((3, 2), (3, 4))
    a = tensor(np.zeros((3, 2)))
    b = tensor(np.zeros((3, 4)))
    with pytest.raises(ValueError):
        _ = a + b


def test_reductions_mean_axis():
    x = ad.randn(4, 3, seed=7, requires_grad=True)
    for axes, kd in [(None, False), (1, False), (0, True), ((0, 1), False)]:
        x.zero_grad()
        out = x.mean(axis=axes, keepdims=kd).sum()
        out.backward()
        ax = tuple(range(2)) if axes is None else (axes if isinstance(axes, tuple) else (axes,))
        expected = np.ones((4, 3)) / (4 if 0 in ax else 1) / (3 if 1 in ax else 1)
        assert_close(x.grad.data, expected)


# ------------------------------------------------------- shared / branching
def test_shared_intermediate_and_branch_join():
    x = ad.randn(3, 3, seed=8, requires_grad=True)
    shared = ad.relu(x @ x)
    branch1 = (shared * 2).sum()
    branch2 = (shared - 1).pow(2).sum()
    loss = branch1 + branch2
    loss.backward()

    def fw(a):
        s = np.maximum(a @ a, 0)
        return (s * 2).sum() + ((s - 1) ** 2).sum()

    assert_close(x.grad.data, numerical_grad(fw, x.data.copy()), rtol=1e-3)


def test_same_variable_reused_in_expression():
    x = tensor(0.7, requires_grad=True)
    # f(x) = x*x + x + 1/x; same leaf used 3 times
    loss = x * x + x + 1.0 / x
    loss.backward()
    assert_close(np.atleast_1d(float(x.grad.data)),
                 np.array([2 * 0.7 + 1 - 1 / 0.7 ** 2]), rtol=1e-4)


def test_two_backwards_accumulate_and_repeat():
    x = ad.randn(2, 2, seed=9, requires_grad=True)
    loss1 = (x ** 2).sum()
    loss1.backward()
    first = x.grad.data.copy()
    # second backward on the *same* graph must accumulate
    loss1.backward()
    assert_close(x.grad.data, 2 * first)
    # a new independent graph, gradients keep accumulating (PyTorch semantics)
    loss2 = (x ** 3).sum()
    loss2.backward()
    num = 2 * 2 * x.data + 3 * x.data ** 2
    assert_close(x.grad.data, num, rtol=1e-3)
    x.zero_grad()
    assert x.grad is None


# ------------------------------------------------------ in-place invalidation
def test_inplace_modification_invalidates_old_graph():
    w = ad.randn(3, 2, seed=10, requires_grad=True)
    x = ad.randn(4, 3, seed=11)
    loss = (x @ w).sum()
    w.add_(np.ones(2))  # optimizer-style update before backward
    with pytest.raises(RuntimeError, match="modified in place"):
        loss.backward()


def test_direct_buffer_mutation_is_rejected():
    w = ad.randn(3, 2, seed=12, requires_grad=True)
    with pytest.raises(ValueError):
        w.data[0, 0] = 9.0  # read-only numpy buffer
    # graph itself is untouched and still valid
    loss = (w ** 2).sum()
    loss.backward()
    assert_close(w.grad.data, 2 * w.data)


def test_retain_graph_false_frees_graph():
    x = tensor(1.5, requires_grad=True)
    loss = (x ** 2).sum()
    loss.backward(retain_graph=False)
    with pytest.raises(RuntimeError):
        loss.backward()


# ----------------------------------------------------------- graph lifetime
def test_no_global_graph_state_between_independent_graphs():
    # Two graphs built interleaved and kept only through their output tensors.
    a = tensor(2.0, requires_grad=True)
    b = tensor(3.0, requires_grad=True)
    ga_out = a * 5
    gb_out = b + 1
    del a, b  # only the outputs remain, each keeps its own parent alive
    gb_out.backward()
    ga_out.backward()
    # parents were local-only to the graph; values check out independently
    assert ga_out.data == 10.0 and gb_out.data == 4.0


def test_no_grad_context():
    with no_grad():
        x = tensor(np.ones(3), requires_grad=True)
        y = (x + 1) * 2
    assert y.requires_grad is False and y._grad_fn is None
    with pytest.raises(RuntimeError):
        y.backward()


# ------------------------------------------------------------------ losses
def test_softmax_cross_entropy_grad():
    logits = ad.randn(6, 4, seed=13, requires_grad=True)
    labels = np.array([0, 1, 2, 3, 0, 2])
    loss = softmax_ce_loss(logits, labels)
    loss.backward()

    def f(a):
        a = a.reshape(6, 4)
        e = np.exp(a - a.max(axis=-1, keepdims=True))
        p = e / e.sum(axis=-1, keepdims=True)
        return -np.log(p[np.arange(6), labels]).mean()

    assert_close(logits.grad.data.ravel(), numerical_grad(f, logits.data.copy().ravel()), rtol=1e-3)


def test_bce_with_logits_grad():
    z = ad.randn(5, seed=14, requires_grad=True)
    y = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
    loss = binary_cross_entropy_with_logits(z, tensor(y))
    loss.backward()
    f = lambda a: (np.maximum(a, 0) - a * y + np.log1p(np.exp(-np.abs(a)))).mean()
    assert_close(z.grad.data, numerical_grad(f, z.data.copy()), rtol=1e-3)


# ------------------------------------------------------------- end-to-end
def _make_dataset(n=128, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    true_w = np.array([1.5, -2.0, 0.5])
    logits = X @ true_w + 0.25
    y = (logits > 0).astype(np.float64)
    return X, y


@pytest.mark.parametrize("opt_cls,kwargs", [
    (optim.SGD, {"lr": 0.2, "momentum": 0.9}),
    (optim.Adam, {"lr": 0.05}),
])
def test_logistic_regression_trains(opt_cls, kwargs):
    X, y = _make_dataset()
    model = nn.Linear(3, 1, seed=1)
    opt = opt_cls(model.parameters(), **kwargs)
    xt, yt = tensor(X), tensor(y.reshape(-1, 1))

    losses = []
    for _ in range(100):
        opt.zero_grad()
        logits = model(xt)
        loss = binary_cross_entropy_with_logits(logits, yt)
        loss.backward()
        opt.step()
        losses.append(float(loss.data))
    assert losses[-1] < losses[0] * 0.6, losses
    pred = 1.0 / (1.0 + np.exp(-model(xt).data.ravel()))
    acc = ((pred > 0.5) == y.astype(bool)).mean()
    assert acc > 0.85


def test_multiclass_mlp_trains():
    rng = np.random.default_rng(1)
    n = 200
    X = rng.standard_normal((n, 4))
    W = rng.standard_normal((4, 3))
    logits = X @ W
    labels = logits.argmax(axis=1)
    model = nn.Sequential(nn.Linear(4, 16, seed=2), nn.ReLU(), nn.Linear(16, 3, seed=3))
    opt = optim.SGD(model.parameters(), lr=0.2)
    xt = tensor(X)
    losses = []
    for _ in range(300):
        opt.zero_grad()
        loss = softmax_ce_loss(model(xt), labels)
        loss.backward()
        opt.step()
        losses.append(float(loss.data))
    assert losses[-1] < 0.2
    acc = (model(xt).data.argmax(axis=1) == labels).mean()
    assert acc > 0.9


def test_inference_after_training_is_detached():
    X, y = _make_dataset(n=32)
    model = nn.Linear(3, 1, seed=4)
    opt = optim.SGD(model.parameters(), lr=0.1)
    for _ in range(10):
        opt.zero_grad()
        loss = mse_loss(model(tensor(X)), tensor(y.reshape(-1, 1)))
        loss.backward()
        opt.step()
    with no_grad():
        out = model(tensor(X))
    assert out._grad_fn is None
