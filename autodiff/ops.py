"""Differentiable operations.

Broadcasting follows NumPy:
  * shapes are right-aligned; dimensions must be equal or one of them be 1
    (or absent, treated as 1);
  * the result shape is the elementwise maximum;
  * a gradient flowing back to an input is summed over every axis that was
    inserted (new leading dims) or stretched (input size 1), then reshaped
    back to the input shape.
"""
from __future__ import annotations

import numpy as np

from .core import Function, Tensor


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...], ndim: int) -> np.ndarray:
    # axes inserted on the left
    for _ in range(grad.ndim - ndim):
        grad = grad.sum(axis=0)
    # axes that were stretched from size 1
    for axis, size in enumerate(shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


def _as_tensor(x) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(np.asarray(x, dtype=np.float64))


# --------------------------------------------------------------- arithmetic
class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        return a + b

    def backward(self, g):
        a, b = self.parents
        return _unbroadcast(g, a.shape, a.ndim), _unbroadcast(g, b.shape, b.ndim)


class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        return a - b

    def backward(self, g):
        a, b = self.parents
        return _unbroadcast(g, a.shape, a.ndim), _unbroadcast(-g, b.shape, b.ndim)


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward()
        ctx.a, ctx.b = a, b
        return a * b

    def backward(self, g):
        a, b = self.parents
        return (
            _unbroadcast(g * self.b, a.shape, a.ndim),
            _unbroadcast(g * self.a, b.shape, b.ndim),
        )


class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.a, ctx.b = a, b
        return a / b

    def backward(self, g):
        a, b = self.parents
        ga = g / self.b
        gb = -g * self.a / (self.b ** 2)
        return _unbroadcast(ga, a.shape, a.ndim), _unbroadcast(gb, b.shape, b.ndim)


class Neg(Function):
    @staticmethod
    def forward(ctx, a):
        return -a

    def backward(self, g):
        return (-g,)


class Pow(Function):
    """x ** y where y may be a scalar or a broadcastable tensor."""

    @staticmethod
    def forward(ctx, x, y):
        ctx.x, ctx.y = x, y
        ctx.y_is_scalar = y.ndim == 0
        return np.power(x, y)

    def backward(self, g):
        x, y = self.parents
        out = g * self.y * np.power(self.x, self.y - 1)
        if self.y_is_scalar:
            return out, None
        gy = g * np.power(self.x, self.y) * np.log(self.x)
        return out, _unbroadcast(gy, y.shape, y.ndim)


class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.a, ctx.b = a, b
        return a @ b

    def backward(self, g):
        a, b = self.parents
        ga, gb = _matmul_grads(self.a, self.b, g)
        return ga, gb


def _matmul_grads(a: np.ndarray, b: np.ndarray, g: np.ndarray):
    # Vector / matrix cases used by the engine.
    if a.ndim == 1 and b.ndim == 1:
        return g * b, g * a
    if a.ndim == 1 and b.ndim == 2:
        # a: (k,), b: (k, n), out: (n,)
        return g @ b.T, np.outer(a, g)
    if a.ndim == 2 and b.ndim == 1:
        # a: (m, k), b: (k,), out: (m,)
        return np.outer(g, b), g @ a
    if a.ndim == 2 and b.ndim == 2:
        return g @ b.T, a.T @ g
    raise NotImplementedError(f"matmul grad not implemented for ndims {a.ndim}, {b.ndim}")


# --------------------------------------------------------------- reductions
def _normalize_axes(axes, ndim):
    if axes is None:
        return tuple(range(ndim))
    if isinstance(axes, (int, np.integer)):
        axes = (axes,)
    return tuple(int(a % ndim) for a in axes)


class Sum(Function):
    @staticmethod
    def forward(ctx, a, axis, keepdims):
        ctx.shape = a.shape
        ctx.axes = _normalize_axes(axis, a.ndim)
        ctx.keepdims = keepdims
        return a.sum(axis=axis if axis is None else tuple(ctx.axes), keepdims=keepdims)


    def backward(self, g):
        shape = list(self.shape)
        if not self.keepdims:
            g = np.expand_dims(g, self.axes)
        return (np.broadcast_to(g, self.shape).copy(),)


class Mean(Function):
    @staticmethod
    def forward(ctx, a, axis, keepdims):
        ctx.shape = a.shape
        ctx.axes = _normalize_axes(axis, a.ndim)
        ctx.keepdims = keepdims
        ctx.count = int(np.prod([ctx.shape[i] for i in ctx.axes])) if ctx.axes else 1
        return a.mean(axis=tuple(ctx.axes), keepdims=keepdims)


    def backward(self, g):
        if not self.keepdims:
            g = np.expand_dims(g, self.axes)
        return (np.broadcast_to(g / self.count, self.shape).copy(),)


# ------------------------------------------------------------------- shape
class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        ctx.in_shape = a.shape
        return a.reshape(shape)

    def backward(self, g):
        return (g.reshape(self.in_shape),)


class Transpose(Function):
    @staticmethod
    def forward(ctx, a, axes):
        ctx.axes = axes
        return np.transpose(a, axes)

    def backward(self, g):
        if self.axes is None:
            return (g.T,)
        inv = np.argsort(self.axes)
        return (np.transpose(g, tuple(inv)),)


# ------------------------------------------------------------------ unary
class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        out = np.exp(a)
        ctx.out = out
        return out

    def backward(self, g):
        return (g * self.out,)


class Log(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.a = a
        return np.log(a)

    def backward(self, g):
        return (g / self.a,)


class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.mask = a > 0
        return np.maximum(a, 0)

    def backward(self, g):
        return (g * self.mask,)


class Sigmoid(Function):
    @staticmethod
    def forward(ctx, a):
        out = 1.0 / (1.0 + np.exp(-a))
        ctx.out = out
        return out

    def backward(self, g):
        return (g * self.out * (1 - self.out),)


class Tanh(Function):
    @staticmethod
    def forward(ctx, a):
        out = np.tanh(a)
        ctx.out = out
        return out

    def backward(self, g):
        return (g * (1 - self.out ** 2),)


class Softmax(Function):
    """Softmax over the last axis."""

    @staticmethod
    def forward(ctx, a):
        shifted = a - a.max(axis=-1, keepdims=True)
        e = np.exp(shifted)
        out = e / e.sum(axis=-1, keepdims=True)
        ctx.out = out
        return out

    def backward(self, g):
        # dL/dx = s * (g - sum(g*s)) per softmax row
        dot = (g * self.out).sum(axis=-1, keepdims=True)
        return (self.out * (g - dot),)


# ---------------------------------------------------------------- public API
def add(a: Tensor, b: Tensor) -> Tensor: return Add.apply(_as_tensor(a), _as_tensor(b))
def sub(a: Tensor, b: Tensor) -> Tensor: return Sub.apply(_as_tensor(a), _as_tensor(b))
def mul(a: Tensor, b: Tensor) -> Tensor: return Mul.apply(_as_tensor(a), _as_tensor(b))
def div(a: Tensor, b: Tensor) -> Tensor: return Div.apply(_as_tensor(a), _as_tensor(b))
def neg(a: Tensor) -> Tensor: return Neg.apply(_as_tensor(a))
def pow(a: Tensor, p) -> Tensor:
    p = p if isinstance(p, Tensor) else Tensor(np.asarray(p, dtype=np.float64))
    return Pow.apply(_as_tensor(a), p)
def matmul(a: Tensor, b: Tensor) -> Tensor: return MatMul.apply(_as_tensor(a), _as_tensor(b))

def sum(a: Tensor, axis=None, keepdims=False) -> Tensor:
    return Sum.apply(a, axis=axis, keepdims=keepdims)

def mean(a: Tensor, axis=None, keepdims=False) -> Tensor:
    return Mean.apply(a, axis=axis, keepdims=keepdims)

def reshape(a: Tensor, shape) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return Reshape.apply(a, shape=tuple(shape))

def transpose(a: Tensor, axes=None) -> Tensor:
    return Transpose.apply(a, axes=None if axes is None else tuple(axes))

def exp(a: Tensor) -> Tensor: return Exp.apply(_as_tensor(a))
def log(a: Tensor) -> Tensor: return Log.apply(_as_tensor(a))
def relu(a: Tensor) -> Tensor: return ReLU.apply(_as_tensor(a))
def sigmoid(a: Tensor) -> Tensor: return Sigmoid.apply(_as_tensor(a))
def tanh(a: Tensor) -> Tensor: return Tanh.apply(_as_tensor(a))
def softmax(a: Tensor) -> Tensor: return Softmax.apply(_as_tensor(a))
