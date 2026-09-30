"""Core tensor / autograd machinery.

Graph lifetime: every :class:`Tensor` produced by a differentiable op owns a
``_grad_fn`` node; that node references its parent tensors.  A whole graph is
therefore kept alive purely by ordinary Python references (the tensors the
user still holds) -- there is no global tape / global graph registry.
"""
from __future__ import annotations

import weakref
from contextlib import contextmanager
from typing import Any, Callable, Iterable

import numpy as np


_GRAD_STACK: list[bool] = []

# Weak registry of every live Tensor; checkpointing snapshots versions
# from it so in-place modification of *closed-over* tensors (not just
# explicit arguments) also invalidates the saved segment.
_ALL_TENSORS: weakref.WeakSet = weakref.WeakSet()

# Process-level RNG used by random ops (``randn``/``rand`` without an explicit
# ``seed``).  Checkpoint records each draw as an internal event, so recomputation
# reuses recorded samples and does not advance the generator again.
_GLOBAL_RNG = np.random.default_rng()


def manual_seed(seed: int) -> None:
    """Seed the global RNG used by random ops."""
    global _GLOBAL_RNG
    _GLOBAL_RNG = np.random.default_rng(seed)


def _rng_state() -> dict:
    return _GLOBAL_RNG.bit_generator.state


def _restore_rng(state: dict) -> None:
    _GLOBAL_RNG.bit_generator.state = state


def get_rng_state() -> dict:
    """Return a JSON-compatible snapshot of the global random generator."""
    return _rng_state()


def set_rng_state(state: dict) -> None:
    """Restore a snapshot returned by :func:`get_rng_state`."""
    _restore_rng(state)


@contextmanager
def no_grad():
    """Temporarily disable graph construction (inference / optimizer updates)."""
    _GRAD_STACK.append(False)
    try:
        yield
    finally:
        _GRAD_STACK.pop()


def _grad_enabled() -> bool:
    return all(_GRAD_STACK) if _GRAD_STACK else True


def _freeze(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if not np.issubdtype(a.dtype, np.floating):
        a = a.astype(np.float64)
    a.setflags(write=False)
    return a


class Function:
    """Base class for differentiable ops.

    Subclasses implement ``forward`` (pure numpy) and ``backward`` (returns one
    gradient w.r.t. data per tensor input).
    """

    needs_input_versions = True

    @staticmethod
    def forward(ctx: "Function", *inputs_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_out: np.ndarray) -> tuple[np.ndarray | None, ...]:
        raise NotImplementedError

    @classmethod
    def apply(cls, *inputs: "Tensor", **params: Any) -> "Tensor":
        ctx = cls()
        ctx.params = params
        ctx.needs_grad = [t.requires_grad for t in inputs]
        # Snapshot versions of every input; staleness is checked at backward.
        ctx.input_versions = [t._version for t in inputs]
        ctx.parents = inputs
        out_data = ctx.forward(ctx, *(t.data for t in inputs), **params)
        requires = _grad_enabled() and any(t.requires_grad for t in inputs)
        return Tensor(out_data, requires_grad=requires, grad_fn=ctx if requires else None)

    def save_for_backward(self, *tensors: "Tensor") -> None:
        self.saved_tensors = tensors

    def _check_versions(self) -> None:
        for t, v in zip(self.parents, self.input_versions):
            if t._version != v:
                raise RuntimeError(
                    "one of the tensors needed by this graph was modified in place; "
                    "the graph is no longer valid for backward()"
                )


class Tensor:
    """A float tensor wrapping a read-only numpy array plus autograd state."""

    def __init__(
        self,
        data: Any,
        requires_grad: bool = False,
        grad_fn: Function | None = None,
    ):
        self._data = _freeze(data)
        self.requires_grad = bool(requires_grad)
        self._grad_fn = grad_fn
        self.grad: Tensor | None = None
        self._version = 0
        _ALL_TENSORS.add(self)

    # ------------------------------------------------------------------ data
    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, value: Any) -> None:
        # ``x.data = new_array`` also invalidates any graph through x.
        self._data = _freeze(value)
        self._version += 1

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def size(self) -> int:
        return int(self._data.size)

    def __repr__(self) -> str:
        return (
            f"Tensor(array={np.array2string(self._data, separator=', ')}, "
            f"requires_grad={self.requires_grad})"
        )

    # ----------------------------------------------------------- graph ops
    def backward(self, grad: "Tensor | np.ndarray | None" = None, retain_graph: bool = True) -> None:
        if not self.requires_grad:
            raise RuntimeError("backward() called on a tensor that does not require grad")
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be supplied for non-scalar outputs")
            g = np.ones_like(self.data)
        else:
            g = grad.data if isinstance(grad, Tensor) else np.asarray(grad, dtype=np.float64)
            if g.shape != self.data.shape:
                raise RuntimeError(f"grad shape {g.shape} does not match output shape {self.data.shape}")

        # Topological order via iterative DFS over grad_fn nodes.
        order: list[Function] = []
        visited: set[int] = set()

        def visit(node: Function) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            for p in node.parents:
                if p._grad_fn is not None:
                    visit(p._grad_fn)
            order.append(node)

        if self._grad_fn is not None:
            visit(self._grad_fn)

        node_grads: dict[int, np.ndarray] = {id(self._grad_fn): g} if self._grad_fn else {}
        tensor_grads: dict[int, np.ndarray] = {}
        if self._grad_fn is None:
            # scalar leaf (or result built under no_grad): seed goes to itself
            tensor_grads[id(self)] = g

        for node in reversed(order):
            if getattr(node, "_freed", False):
                raise RuntimeError(
                    "the graph used by this backward() call was freed by an earlier "
                    "backward(retain_graph=False)"
                )
            node._check_versions()
            upstream = node_grads[id(node)]
            grads = node.backward(upstream)
            for parent, pg in zip(node.parents, grads):
                if pg is None or not parent.requires_grad:
                    continue
                if pg.shape != parent.data.shape:
                    raise RuntimeError(
                        f"op {type(node).__name__} produced grad shape {pg.shape}, "
                        f"expected {parent.data.shape}"
                    )
                if parent._grad_fn is None:
                    key = id(parent)
                    tensor_grads[key] = tensor_grads.get(key, np.zeros_like(parent.data)) + pg
                else:
                    key = id(parent._grad_fn)
                    node_grads[key] = node_grads.get(key, np.zeros_like(parent.data)) + pg

        collected: set[int] = set()

        def collect(t: "Tensor") -> None:
            if not t.requires_grad or t._grad_fn is not None or id(t) in collected:
                return
            collected.add(id(t))
            pg = tensor_grads.get(id(t))
            if pg is None:
                return
            if t.grad is None:
                t.grad = Tensor(pg)
            else:  # shared inputs / branch joins / repeated backward accumulate
                t.grad = Tensor(t.grad.data + pg)

        for node in order:
            for parent in node.parents:
                collect(parent)
        collect(self)

        if not retain_graph:
            for node in order:
                node._freed = True
                node.parents = ()
                if hasattr(node, "saved_tensors"):
                    node.saved_tensors = ()

    def zero_grad(self) -> None:
        self.grad = None

    # --------------------------------------------------------- in-place ops
    # In-place updates are intentionally limited to graph-detached data
    # mutation.  They bump ``_version``, which invalidates every graph that
    # reads this tensor (checked lazily in backward).  The underlying numpy
    # buffer is read-only, so ad-hoc mutation of ``t.data[...]`` raises
    # immediately instead of silently corrupting the graph.
    def _mutate(self, fn: Callable[[np.ndarray], np.ndarray]) -> "Tensor":
        self._data = _freeze(fn(self._data))
        self._version += 1
        self._grad_fn = None
        return self

    def add_(self, other: "Tensor | np.ndarray | float") -> "Tensor":
        other = other.data if isinstance(other, Tensor) else other
        return self._mutate(lambda a: a + other)

    def mul_(self, other: "Tensor | np.ndarray | float") -> "Tensor":
        other = other.data if isinstance(other, Tensor) else other
        return self._mutate(lambda a: a * other)

    def sub_(self, other: "Tensor | np.ndarray | float") -> "Tensor":
        other = other.data if isinstance(other, Tensor) else other
        return self._mutate(lambda a: a - other)

    # ------------------------------------------------------------- operators
    def __add__(self, other): return _binary(add, self, other)
    def __radd__(self, other): return _binary(add, other, self)
    def __sub__(self, other): return _binary(sub, self, other)
    def __rsub__(self, other): return _binary(sub, other, self)
    def __mul__(self, other): return _binary(mul, self, other)
    def __rmul__(self, other): return _binary(mul, other, self)
    def __truediv__(self, other): return _binary(div, self, other)
    def __rtruediv__(self, other): return _binary(div, other, self)
    def __pow__(self, other): return pow_(self, other)
    def pow(self, other): return pow_(self, other)
    def __neg__(self): return neg(self)
    def __matmul__(self, other): return matmul(self, other)

    def sum(self, axis=None, keepdims=False): return sum_(self, axis=axis, keepdims=keepdims)
    def mean(self, axis=None, keepdims=False): return mean(self, axis=axis, keepdims=keepdims)
    def reshape(self, *shape): return reshape(self, shape)

    @property
    def T(self):
        from .ops import transpose
        return transpose(self)


def _binary(fn, a, b):
    if not isinstance(a, Tensor):
        a = Tensor(np.asarray(a, dtype=np.float64))
    if not isinstance(b, Tensor):
        b = Tensor(np.asarray(b, dtype=np.float64))
    return fn(a, b)


# ---------------------------------------------------------------- constructors
def tensor(data, requires_grad: bool = False) -> Tensor:
    return Tensor(np.asarray(data, dtype=np.float64), requires_grad=requires_grad)


def zeros(*shape, requires_grad: bool = False) -> Tensor:
    return Tensor(np.zeros(_shape(shape)), requires_grad=requires_grad)


def ones(*shape, requires_grad: bool = False) -> Tensor:
    return Tensor(np.ones(_shape(shape)), requires_grad=requires_grad)


def zeros_like(t: Tensor) -> Tensor:
    return Tensor(np.zeros_like(t.data))


def ones_like(t: Tensor) -> Tensor:
    return Tensor(np.ones_like(t.data))


def randn(*shape, seed: int | None = None, requires_grad: bool = False) -> Tensor:
    shape_tuple = _shape(shape)
    if seed is not None:
        value = np.random.default_rng(seed).standard_normal(shape_tuple)
    else:
        from .events import _random_event
        draw = lambda: _GLOBAL_RNG.standard_normal(shape_tuple)
        value = _random_event(
            "internal_random", f"randn:{','.join(map(str, shape_tuple))}", shape_tuple, draw
        )
    return Tensor(value, requires_grad=requires_grad)


def rand(*shape, seed: int | None = None, requires_grad: bool = False) -> Tensor:
    shape_tuple = _shape(shape)
    if seed is not None:
        value = np.random.default_rng(seed).uniform(0.0, 1.0, shape_tuple)
    else:
        from .events import _random_event
        draw = lambda: _GLOBAL_RNG.uniform(0.0, 1.0, shape_tuple)
        value = _random_event(
            "internal_random", f"rand:{','.join(map(str, shape_tuple))}", shape_tuple, draw
        )
    return Tensor(value, requires_grad=requires_grad)


def _shape(shape: Iterable) -> tuple[int, ...]:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        return tuple(shape[0])
    return tuple(int(s) for s in shape)


def broadcast_shapes(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    """NumPy broadcasting rules; raises ValueError if shapes are incompatible."""
    return tuple(np.broadcast_shapes(*shapes))


# imported late to avoid a cycle at module import
from .ops import add, sub, mul, div, neg, pow as pow_, matmul, sum as sum_, mean, reshape, transpose  # noqa: E402
