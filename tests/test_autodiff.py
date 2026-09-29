"""Tests for the autodiff engine: correctness, accumulation and lifecycle."""

import numpy as np
import pytest

from autodiff import AutogradError, Tensor, nn, ops, optim
from autodiff.grad_check import check_gradients


# ------------------------------------------------------------ numerical check
def test_gradcheck_broadcast_mul_div_pow():
    rng = np.random.default_rng(0)
    a = Tensor(rng.normal(size=(3, 4)) + 2.0, requires_grad=True)
    b = Tensor(rng.normal(size=(4,)) + 2.0, requires_grad=True)
    c = Tensor(rng.normal(size=(3, 1)), requires_grad=True)

    def fn():
        return (((a * b + c) / b) ** 2.1).sum()

    report = check_gradients(fn, [a, b, c], atol=1e-5, rtol=1e-3)
    assert all(abs_err < 1e-5 for _, abs_err, _ in report)


def test_gradcheck_matmul_and_activations():
    rng = np.random.default_rng(1)
    x = Tensor(rng.normal(size=(5, 3)), requires_grad=True)
    w1 = Tensor(rng.normal(size=(3, 4)), requires_grad=True)
    w2 = Tensor(rng.normal(size=(4, 1)), requires_grad=True)
    bias = Tensor(rng.normal(size=(4,)), requires_grad=True)

    def fn():
        h = ops.tanh(x @ w1 + bias)
        out = ops.relu(h @ w2)
        return (out * out + ops.sigmoid(out)).mean()

    check_gradients(fn, [x, w1, w2, bias], atol=1e-5, rtol=1e-3)


def test_gradcheck_transcendentals_and_reductions():
    rng = np.random.default_rng(2)
    a = Tensor(rng.normal(size=(2, 3, 4)) + 0.5, requires_grad=True)

    def fn2():
        return (ops.exp(-a) + ops.sqrt(a * a + 0.5) + ops.sin(a)).sum()

    check_gradients(fn2, [a], atol=1e-5, rtol=1e-3)


def test_gradcheck_reshape_transpose_slice():
    rng = np.random.default_rng(3)
    a = Tensor(rng.normal(size=(3, 4)), requires_grad=True)

    def fn():
        b = a.transpose(1, 0).reshape(2, 6)
        return (b[0] * b[1]).sum()

    check_gradients(fn, [a], atol=1e-5, rtol=1e-3)


def test_gradcheck_bce_logits():
    rng = np.random.default_rng(4)
    logits = Tensor(rng.normal(size=(10,)), requires_grad=True)
    y = (rng.normal(size=(10,)) > 0).astype(np.float64)

    def fn():
        return nn.binary_cross_entropy_with_logits(logits, y)

    check_gradients(fn, [logits], atol=1e-5, rtol=1e-3)


# ------------------------------------------------------- accumulation / DAG
def test_shared_intermediate_accumulates():
    # y = f(x) + f(x): gradient through the shared node must be doubled.
    x = Tensor([1.5, -0.7], requires_grad=True)
    shared = x * x
    out = (shared + shared).sum()
    out.backward()
    expected = 4.0 * x.numpy()
    np.testing.assert_allclose(x.grad, expected)


def test_diamond_branch_merge():
    # Diamond: x -> a,b -> merge -> loss, both paths must contribute.
    x = Tensor(2.0, requires_grad=True)
    left = ops.exp(x)
    right = ops.log(x)
    merged = left * right + left - right
    merged.backward()
    eps = 1e-7
    numeric = (
        float((ops.exp(Tensor(2.0 + eps)) * ops.log(Tensor(2.0 + eps))
               + ops.exp(Tensor(2.0 + eps)) - ops.log(Tensor(2.0 + eps))).data)
        - float((ops.exp(Tensor(2.0 - eps)) * ops.log(Tensor(2.0 - eps))
                 + ops.exp(Tensor(2.0 - eps)) - ops.log(Tensor(2.0 - eps))).data)
    ) / (2 * eps)
    assert x.grad == pytest.approx(numeric, rel=1e-5)


def test_reused_variable_in_chained_expression():
    # One variable used many times across one expression.
    x = Tensor(0.9, requires_grad=True)
    out = x + x * x + x ** 3 - x / 2
    out.backward()
    # d/dx: 1 + 2x + 3x^2 - 0.5
    assert x.grad == pytest.approx(1 + 2 * 0.9 + 3 * 0.9 ** 2 - 0.5, rel=1e-12)


def test_two_backward_calls_accumulate_and_retain():
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = (x * x).sum()
    y.backward(retain_graph=True)
    first = x.grad.copy()
    # Second backward over the same retained graph adds again.
    y.backward()
    np.testing.assert_allclose(x.grad, 2 * first)


def test_graph_freed_after_backward():
    x = Tensor(1.0, requires_grad=True)
    y = ops.exp(x)
    z = y * y
    z.backward()
    assert y._creator is None and z._creator is None
    assert x._creator is None  # leaf stays a usable leaf
    # Re-running backward on a freed graph is an explicit error, not silent junk.
    with pytest.raises(AutogradError):
        z.backward()


# --------------------------------------------------------------- broadcasting
@pytest.mark.parametrize(
    "shape_a,shape_b",
    [((3, 4), (4,)), ((4,), (3, 4)), ((2, 1, 4), (3, 4)), ((), (3, 4))],
)
def test_broadcast_grad_shapes(shape_a, shape_b):
    rng = np.random.default_rng(5)
    a = Tensor(rng.normal(size=shape_a), requires_grad=True)
    b = Tensor(rng.normal(size=shape_b), requires_grad=True)

    def fn():
        return ops.tanh(a + b * 2.0).sum()

    check_gradients(fn, [a, b], atol=1e-5, rtol=1e-3)


def test_axis_reduction_shapes():
    rng = np.random.default_rng(6)
    a = Tensor(rng.normal(size=(4, 5, 6)), requires_grad=True)
    for axis, keepdims in [(None, False), (1, False), ((0, 2), True), (0, True)]:
        target = a

        def fn(axis=axis, keepdims=keepdims):
            return target.sum(axis=axis, keepdims=keepdims).sum()

        # sum of sums = total sum, gradient should be ones
        out = fn()
        out.backward()
        np.testing.assert_allclose(a.grad, np.ones_like(a.numpy()))
        a.grad = None


# -------------------------------------------------------- in-place / versions
def test_direct_inplace_mutation_rejected():
    x = Tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(AutogradError):
        x.data[0] = 9.0
    with pytest.raises(AutogradError):
        x._data[0] = 9.0
    with pytest.raises(AutogradError):
        x._data *= 2


def test_stale_graph_after_leaf_replace_rejected():
    w = Tensor(1.0, requires_grad=True)
    x = Tensor(0.5)
    y = ops.exp(w * x)
    # Simulate an external/sanctioned buffer replacement mid-graph-life.
    w._data.replace(np.asarray(3.0))
    w._version += 1
    with pytest.raises(AutogradError, match="version mismatch"):
        y.backward()


def test_optimizer_step_invalidates_old_graph():
    w = Tensor(1.0, requires_grad=True)
    y = w * w
    y.backward(retain_graph=True)
    opt = optim.SGD([w], lr=0.1)
    opt.step()
    assert w.numpy() == pytest.approx(0.8)
    with pytest.raises(AutogradError, match="version mismatch"):
        y.backward()
    # Fresh graph built after the step works normally.
    (w * w).backward()
    assert w.grad == pytest.approx(1.6)


# --------------------------------------------------------------- optimizers
def test_sgd_reduces_quadratic():
    w = Tensor([3.0, -4.0], requires_grad=True)
    opt = optim.SGD([w], lr=0.05)
    for _ in range(200):
        opt.zero_grad()
        (w * w).sum().backward()
        opt.step()
    np.testing.assert_allclose(w.numpy(), [0.0, 0.0], atol=1e-8)


def test_sgd_momentum_step_math():
    w = Tensor(1.0, requires_grad=True)
    opt = optim.SGD([w], lr=0.1, momentum=0.9)
    (w * w).backward()  # grad = 2
    opt.step()
    assert w.numpy() == pytest.approx(0.8)  # v=g=2, w -= 0.1*2
    (w * w).backward()  # grad = 1.6
    opt.step()
    # v = 0.9*2 + 1.6 = 3.4
    assert w.numpy() == pytest.approx(0.8 - 0.1 * 3.4)


def test_adam_reduces_quadratic():
    w = Tensor([3.0, -4.0], requires_grad=True)
    opt = optim.Adam([w], lr=0.1)
    for _ in range(300):
        opt.zero_grad()
        (w * w).sum().backward()
        opt.step()
    np.testing.assert_allclose(w.numpy(), [0.0, 0.0], atol=1e-3)


# ------------------------------------------------------------ training e2e
def test_tiny_training_loop_overfits():
    rng = np.random.default_rng(7)
    # Learn XOR-ish: 4 points, 2-8-1 tanh network.
    x = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64))
    y = np.array([[0.0], [1.0], [1.0], [0.0]])
    model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
    opt = optim.Adam(model.parameters(), lr=0.05)
    losses = []
    for _ in range(400):
        opt.zero_grad()
        loss = nn.mse_loss(model(x), y)
        losses.append(float(loss.data))
        loss.backward()
        opt.step()
    assert losses[-1] < 0.02
    assert losses[-1] < losses[0] / 10


# --------------------------------------------------------- lifecycle/globals
def test_independent_graphs_coexist():
    # Two graphs with separate lives must not interfere (no global tape).
    a = Tensor(2.0, requires_grad=True)
    ga = ops.exp(a)
    b = Tensor(3.0, requires_grad=True)
    gb = b * b
    gb.backward()
    assert b.grad == pytest.approx(6.0)
    assert a.grad is None
    # First graph still intact and backward-able after the second was freed.
    ga.backward()
    assert a.grad == pytest.approx(np.exp(2.0))


def test_non_scalar_requires_explicit_grad():
    x = Tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(AutogradError):
        (x * x).backward()


def test_integer_tensor_cannot_require_grad():
    with pytest.raises(TypeError):
        Tensor([1, 2], requires_grad=True)
