"""Core tensor type and reverse-mode automatic differentiation.

The graph is carried explicitly by each ``Tensor`` (its ``creator`` /
``inputs`` edges).  There is deliberately no module-level global graph or
global tape: building, keeping alive and freeing a graph is purely a matter of
normal Python object lifetimes, so independent graphs coexist naturally.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np


class AutogradError(RuntimeError):
    """Raised for autodiff misuse (invalidated graph, double backward, ...)."""


class _TrackedArray(np.ndarray):
    """ndarray subclass that rejects in-place writes through ``arr[...] = x``.

    In-place operations on a leaf are the classic way to silently invalidate
    a recorded graph (the recorded computation read the *old* buffer).  We
    reject them eagerly instead of producing wrong gradients.  ``replace``
    below is the sanctioned, version-bumping escape hatch used by optimizers,
    which only act between ``backward`` calls.
    """

    def __setitem__(self, key, value):  # noqa: D401 - numpy protocol
        raise AutogradError(
            "in-place modification of a Tensor's data is forbidden: it would "
            "invalidate graphs recorded against the old value; build a new "
            "Tensor, or let an Optimizer update leaves between backward() calls"
        )

    def _reject_inplace(self, *args, **kwargs):
        raise AutogradError(
            "in-place arithmetic (+=, *=, ...) on a Tensor's data is forbidden: "
            "it would invalidate recorded graphs"
        )

    __iadd__ = __isub__ = __imul__ = __itruediv__ = _reject_inplace
    __ifloordiv__ = __imod__ = __ipow__ = _reject_inplace

    def replace(self, value: np.ndarray) -> None:
        """Overwrite the buffer unconditionally (optimizer-only escape hatch)."""
        super().__setitem__(Ellipsis, value)


class Context:
    """Per-op scratch space shared between the forward and backward passes."""

    __slots__ = ("saved_tensors", "saved", "needs_input_grad")

    def __init__(self, needs_input_grad: tuple[bool, ...]):
        self.saved_tensors: list[Tensor] = []
        self.saved: dict[str, Any] = {}
        self.needs_input_grad = needs_input_grad

    def save_for_backward(self, *tensors: "Tensor") -> None:
        self.saved_tensors.extend(tensors)

    def save(self, **kwargs: Any) -> None:
        self.saved.update(kwargs)


class Function:
    """Base class for a differentiable primitive (define-by-run style).

    Subclasses implement ``forward(ctx, *raw_ndarrays) -> ndarray`` and
    ``backward(ctx, grad_ndarray) -> ndarray | None``.  ``apply`` wires the
    numpy-level math into the tensor-level graph.
    """

    @staticmethod
    def forward(ctx: Context, *inputs: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):  # pragma: no cover
        raise NotImplementedError

    @classmethod
    def apply(cls, *inputs: "Tensor", **forward_kwargs) -> "Tensor":
        ctx = Context(tuple(x.requires_grad for x in inputs))
        raw_inputs = [x.data for x in inputs]
        # Snapshot the leaf versions used by this node.  If any input is later
        # modified in place, its version changes and traversing the stale edge
        # raises instead of reading a clobbered buffer.
        input_versions = tuple(x._version for x in inputs)
        out_data = cls.forward(ctx, *raw_inputs, **forward_kwargs)
        requires_grad = any(x.requires_grad for x in inputs)
        out = Tensor(out_data, requires_grad=requires_grad)
        if requires_grad:
            out._creator = (cls, ctx, inputs, input_versions)
        return out


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Reduce ``grad`` so it matches the pre-broadcast ``shape``.

    Broadcasting rules (numpy / pytorch style):
      * dimensions of size 1 are summed (and kept),
      * missing leading dimensions are summed away entirely,
      * every other dimension must already match.
    """
    # Leading dimensions inserted by broadcasting.
    for _ in range(grad.ndim - len(shape)):
        grad = grad.sum(axis=0)
    # Trailing dimensions that were stretched from size 1.
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """A multi-dimensional array with an optional autograd history.

    Parameters
    ----------
    data:
        array-like initial value, or a numpy ndarray.
    requires_grad:
        whether gradients are tracked for this leaf.  Floating point tensors
        default to ``False``; integer/boolean tensors never require grad.
    dtype:
        optional numpy dtype for the stored buffer.
    """

    def __init__(
        self,
        data: Any,
        requires_grad: bool = False,
        dtype: Optional[np.dtype | type] = None,
    ):
        if isinstance(data, np.ndarray):
            arr = data
        else:
            arr = np.asarray(data)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        if requires_grad:
            if not np.issubdtype(arr.dtype, np.floating):
                raise TypeError(
                    f"only floating point tensors can require gradients, got {arr.dtype}"
                )
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)
        arr = arr.view(_TrackedArray)
        self._data = arr
        self.requires_grad = bool(requires_grad)
        self.grad: Optional[np.ndarray] = None
        # Bumped on every sanctioned in-place buffer replacement (optimizer).
        self._version = 0
        # (Function class, ctx, input tensors, input versions) or None for leaves.
        self._creator: Optional[tuple] = None

    # ------------------------------------------------------------------ basic
    @property
    def data(self) -> np.ndarray:
        """The underlying numpy buffer (read-only view; never write in place)."""
        return self._data

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._data.shape)

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def size(self) -> int:
        return int(self._data.size)

    def __repr__(self) -> str:
        return (
            f"Tensor({np.array2string(self._data, separator=', ')}, "
            f"requires_grad={self.requires_grad})"
        )

    def detach(self) -> "Tensor":
        """Return a new tensor sharing no graph history with this one."""
        return Tensor(np.array(self._data, copy=True))

    def numpy(self) -> np.ndarray:
        return np.array(self._data, copy=True)

    def zero_grad(self) -> None:
        self.grad = None

    # --------------------------------------------------------------- backward
    def backward(self, grad_output: Any = None, retain_graph: bool = False) -> None:
        """Accumulate gradients into ``.grad`` of reachable leaves.

        * Scalar tensors may omit ``grad_output`` (it defaults to 1.0).
        * Gradients *accumulate*: shared intermediates and merging branches
          receive the sum of contributions from all outgoing paths.  Calling
          ``backward`` twice accumulates a second time; pass
          ``retain_graph=True`` on every call but the last to keep the graph.
        """
        if not self.requires_grad:
            raise AutogradError("backward() called on a tensor that does not require grad")
        if grad_output is None:
            if self._data.size != 1:
                raise AutogradError(
                    "grad_output is required for backward() on a non-scalar tensor"
                )
            grad_output = np.ones_like(self._data)
        else:
            grad_output = np.asarray(grad_output, dtype=self._data.dtype)
            if grad_output.shape != self.shape:
                raise AutogradError(
                    f"grad_output shape {grad_output.shape} does not match {self.shape}"
                )

        # Build the reverse topological order.
        topo: list[Tensor] = []
        visited: set[int] = set()

        def visit(t: Tensor) -> None:
            if id(t) in visited:
                return
            visited.add(id(t))
            if t._creator is not None:
                _, _, inputs, _ = t._creator
                for inp in inputs:
                    visit(inp)
                topo.append(t)

        visit(self)

        grads: dict[int, np.ndarray] = {id(self): grad_output}
        for node in reversed(topo):
            fn_cls, ctx, inputs, input_versions = node._creator
            grad_node = grads.pop(id(node))
            # Version guard: a leaf feeding this node was replaced after the
            # node was recorded, so this graph no longer describes its data.
            for inp, version in zip(inputs, input_versions):
                if inp._version != version:
                    raise AutogradError(
                        "trying to backward through a graph whose input was "
                        "modified in place (version mismatch); rebuild the graph "
                        "after the optimizer step"
                    )
            input_grads = fn_cls.backward(ctx, grad_node)
            if not isinstance(input_grads, tuple):
                input_grads = (input_grads,)
            for inp, g in zip(inputs, input_grads):
                if g is None or not inp.requires_grad:
                    continue
                if id(inp) in grads:
                    grads[id(inp)] = grads[id(inp)] + g
                else:
                    grads[id(inp)] = g
                if inp._creator is None:
                    inp.grad = g.copy() if inp.grad is None else inp.grad + g

        if not retain_graph:
            # Release graph edges; tensor values/leaves remain usable for a
            # fresh forward pass.
            for node in topo:
                node._creator = None
                node.requires_grad = False

    # -------------------------------------------------------------- operators
    def __neg__(self):
        from .ops import Neg

        return Neg.apply(self)

    def __add__(self, other):
        from .ops import Add

        return Add.apply(self, as_tensor(other))

    def __radd__(self, other):
        from .ops import Add

        return Add.apply(as_tensor(other), self)

    def __sub__(self, other):
        from .ops import Sub

        return Sub.apply(self, as_tensor(other))

    def __rsub__(self, other):
        from .ops import Sub

        return Sub.apply(as_tensor(other), self)

    def __mul__(self, other):
        from .ops import Mul

        return Mul.apply(self, as_tensor(other))

    def __rmul__(self, other):
        from .ops import Mul

        return Mul.apply(as_tensor(other), self)

    def __truediv__(self, other):
        from .ops import Div

        return Div.apply(self, as_tensor(other))

    def __rtruediv__(self, other):
        from .ops import Div

        return Div.apply(as_tensor(other), self)

    def __pow__(self, exponent):
        from .ops import PowScalar

        if isinstance(exponent, Tensor):
            return NotImplemented
        return PowScalar.apply(self, float(exponent))

    def __matmul__(self, other):
        from .ops import MatMul

        return MatMul.apply(self, as_tensor(other))

    # ----------------------------------------------------------- reductions
    def sum(self, axis: Any = None, keepdims: bool = False) -> "Tensor":
        from .ops import Sum

        return Sum.apply(self, axis=axis, keepdims=keepdims)

    def mean(self, axis: Any = None, keepdims: bool = False) -> "Tensor":
        from .ops import Mean

        return Mean.apply(self, axis=axis, keepdims=keepdims)

    # --------------------------------------------------------------- reshapes
    def reshape(self, *shape: Any) -> "Tensor":
        from .ops import Reshape

        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Reshape.apply(self, shape=tuple(shape))

    def transpose(self, *axes: Any) -> "Tensor":
        from .ops import Transpose

        if len(axes) == 0:
            axes = None
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        else:
            axes = tuple(axes)
        return Transpose.apply(self, axes=axes)

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    # -------------------------------------------------------------- indexing
    def __getitem__(self, key) -> "Tensor":
        from .ops import GetItem

        return GetItem.apply(self, key=key)


def as_tensor(x: Any) -> Tensor:
    """Coerce scalars / arrays / tensors into a detached constant ``Tensor``."""
    if isinstance(x, Tensor):
        return x
    return Tensor(x)


def tensor(x: Any, requires_grad: bool = False, dtype=None) -> Tensor:
    """Convenience constructor mirroring ``numpy.array``-style creation."""
    return Tensor(x, requires_grad=requires_grad, dtype=dtype)


__all__ = ["Tensor", "Context", "Function", "AutogradError", "as_tensor", "tensor"]
