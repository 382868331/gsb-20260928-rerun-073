"""Differentiable primitive operations.

Each primitive is a :class:`~autodiff.tensor.Function` with an explicit
forward (numpy math + saved context) and backward (vector-Jacobian product).
Elementwise binary ops follow numpy broadcasting rules; reductions follow
numpy ``axis`` / ``keepdims`` semantics.
"""

from __future__ import annotations

import numpy as np

from .tensor import Context, Function, Tensor, _unbroadcast


# --------------------------------------------------------------- arithmetic
class Add(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx.save(shape_a=a.shape, shape_b=b.shape)
        return a + b

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        return _unbroadcast(grad, ctx.saved["shape_a"]), _unbroadcast(
            grad, ctx.saved["shape_b"]
        )


class Sub(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx.save(shape_a=a.shape, shape_b=b.shape)
        return a - b

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        return _unbroadcast(grad, ctx.saved["shape_a"]), _unbroadcast(
            -grad, ctx.saved["shape_b"]
        )


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(Tensor(a), Tensor(b))
        ctx.save(shape_a=a.shape, shape_b=b.shape)
        return a * b

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        a, b = ctx.saved_tensors
        return (
            _unbroadcast(grad * b.data, ctx.saved["shape_a"]),
            _unbroadcast(grad * a.data, ctx.saved["shape_b"]),
        )


class Div(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(Tensor(a), Tensor(b))
        ctx.save(shape_a=a.shape, shape_b=b.shape)
        return a / b

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        a, b = ctx.saved_tensors
        return (
            _unbroadcast(grad / b.data, ctx.saved["shape_a"]),
            _unbroadcast(-grad * a.data / (b.data * b.data), ctx.saved["shape_b"]),
        )


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        return -a

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        return -grad


class PowScalar(Function):
    """``a ** n`` for a python-number exponent (d/d a = n * a**(n-1))."""

    @staticmethod
    def forward(ctx: Context, a: np.ndarray, n: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(Tensor(a))
        ctx.save(n=float(n))
        return np.power(a, n)

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        (a,) = ctx.saved_tensors
        n = ctx.saved["n"]
        return grad * n * np.power(a.data, n - 1.0), None

    @classmethod
    def apply(cls, a: Tensor, n: float) -> Tensor:  # type: ignore[override]
        exponent = Tensor(np.asarray(n, dtype=a.dtype))
        return super().apply(a, exponent)


class ScalarMul(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, scale: np.ndarray) -> np.ndarray:
        ctx.save(scale=float(scale))
        return a * scale

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        return grad * ctx.saved["scale"], None


# ------------------------------------------------------------------ math fns
class Exp(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        out = np.exp(a)
        ctx.save(out=out)
        return out

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        return grad * ctx.saved["out"]


class Log(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(Tensor(a))
        return np.log(a)

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        (a,) = ctx.saved_tensors
        return grad / a.data


class Sqrt(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        out = np.sqrt(a)
        ctx.save(out=out)
        return out

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        return grad / (2.0 * ctx.saved["out"])


class Sin(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(Tensor(a))
        return np.sin(a)

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        (a,) = ctx.saved_tensors
        return grad * np.cos(a.data)


# --------------------------------------------------------------- activations
class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        ctx.save(mask=(a > 0))
        return np.maximum(a, 0.0)

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        return grad * ctx.saved["mask"]


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        out = 1.0 / (1.0 + np.exp(-a))
        ctx.save(out=out)
        return out

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        out = ctx.saved["out"]
        return grad * out * (1.0 - out)


class Tanh(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        out = np.tanh(a)
        ctx.save(out=out)
        return out

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        out = ctx.saved["out"]
        return grad * (1.0 - out * out)


# ---------------------------------------------------------------- matmul
class MatMul(Function):
    """``a @ b`` supporting 1-D / 2-D operands (the common NN cases)."""

    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(Tensor(a), Tensor(b))
        return a @ b

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        a, b = ctx.saved_tensors
        x, w = a.data, b.data
        if x.ndim == 1 and w.ndim == 1:
            return grad * w, grad * x
        if x.ndim == 1 and w.ndim == 2:
            # x: (K,), w: (K, N), out: (N,)
            return grad @ w.T, np.outer(x, grad)
        if x.ndim == 2 and w.ndim == 1:
            # x: (M, K), w: (K,), out: (M,)
            return np.outer(grad, w), x.T @ grad
        # 2-D @ 2-D: (M, K) @ (K, N) -> (M, N)
        return grad @ w.T, x.T @ grad


# -------------------------------------------------------------- reductions
class Sum(Function):
    @staticmethod
    def _normalize(axis, ndim):
        if axis is None:
            return tuple(range(ndim))
        if isinstance(axis, int):
            axis = (axis,)
        return tuple(ax % ndim for ax in axis)

    @staticmethod
    def forward(ctx: Context, a: np.ndarray, axis=None, keepdims=False) -> np.ndarray:
        norm = Sum._normalize(axis, a.ndim)
        # Always compute in keepdims form first so the axes stay identifiable;
        # squeeze afterwards only when the caller asked for a plain scalar.
        kept = a.sum(axis=norm, keepdims=True)
        squeeze_all = (not keepdims) and len(set(norm)) == a.ndim
        ctx.save(shape=a.shape, axis=norm, keepdims=keepdims, squeeze_all=squeeze_all)
        if squeeze_all:
            return kept.reshape(())
        if keepdims:
            return kept
        return np.squeeze(kept, axis=tuple(sorted(norm)))

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        shape, axis, keepdims = ctx.saved["shape"], ctx.saved["axis"], ctx.saved["keepdims"]
        if ctx.saved.get("squeeze_all", False):
            grad = grad.reshape((1,) * len(shape))
        elif not keepdims:
            for ax in sorted(axis):
                grad = np.expand_dims(grad, ax)
        return np.broadcast_to(grad, shape).copy()


class Mean(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, axis=None, keepdims=False) -> np.ndarray:
        norm_axes = Sum._normalize(axis, a.ndim)
        count = int(np.prod([a.shape[ax] for ax in norm_axes]))
        kept = a.mean(axis=norm_axes, keepdims=True)
        squeeze_all = (not keepdims) and len(set(norm_axes)) == a.ndim
        ctx.save(shape=a.shape, axis=norm_axes, keepdims=keepdims, count=count,
                 squeeze_all=squeeze_all)
        if squeeze_all:
            return kept.reshape(())
        if keepdims:
            return kept
        return np.squeeze(kept, axis=tuple(sorted(norm_axes)))

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        shape, axis, keepdims, count = (
            ctx.saved["shape"],
            ctx.saved["axis"],
            ctx.saved["keepdims"],
            ctx.saved["count"],
        )
        if ctx.saved.get("squeeze_all", False):
            grad = grad.reshape((1,) * len(shape))
        elif not keepdims:
            for ax in sorted(axis):
                grad = np.expand_dims(grad, ax)
        return np.broadcast_to(grad / count, shape).copy()


# --------------------------------------------------------------- reshaping
class Reshape(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, shape=None) -> np.ndarray:
        ctx.save(shape=a.shape)
        return a.reshape(shape)

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        return grad.reshape(ctx.saved["shape"])


class Transpose(Function):
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, axes=None) -> np.ndarray:
        if axes is None:
            axes = tuple(reversed(range(a.ndim)))
        ctx.save(inverse=np.argsort(axes))
        return np.transpose(a, axes)

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        return np.transpose(grad, ctx.saved["inverse"])


class GetItem(Function):
    """Basic slicing/indexing; gradient is scattered back into the full shape."""

    @staticmethod
    def forward(ctx: Context, a: np.ndarray, key=None) -> np.ndarray:
        ctx.save(shape=a.shape, key=key)
        return a[key]

    @staticmethod
    def backward(ctx: Context, grad: np.ndarray):
        out = np.zeros(ctx.saved["shape"], dtype=grad.dtype)
        out[ctx.saved["key"]] = grad
        return out


# --------------------------------------------------------- functional API
def _unary(fn_cls, x: Tensor) -> Tensor:
    return fn_cls.apply(x if isinstance(x, Tensor) else Tensor(x))


def exp(x: Tensor) -> Tensor:
    return _unary(Exp, x)


def log(x: Tensor) -> Tensor:
    return _unary(Log, x)


def sqrt(x: Tensor) -> Tensor:
    return _unary(Sqrt, x)


def sin(x: Tensor) -> Tensor:
    return _unary(Sin, x)


def relu(x: Tensor) -> Tensor:
    return _unary(ReLU, x)


def sigmoid(x: Tensor) -> Tensor:
    return _unary(Sigmoid, x)


def tanh(x: Tensor) -> Tensor:
    return _unary(Tanh, x)


def sum(x: Tensor, axis=None, keepdims=False) -> Tensor:
    return x.sum(axis=axis, keepdims=keepdims)


def mean(x: Tensor, axis=None, keepdims=False) -> Tensor:
    return x.mean(axis=axis, keepdims=keepdims)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return MatMul.apply(a, b)


__all__ = [
    "Add", "Sub", "Mul", "Div", "Neg", "PowScalar",
    "Exp", "Log", "Sqrt", "Sin",
    "ReLU", "Sigmoid", "Tanh", "MatMul",
    "Sum", "Mean", "Reshape", "Transpose", "GetItem",
    "exp", "log", "sqrt", "sin", "relu", "sigmoid", "tanh",
    "sum", "mean", "matmul",
]
