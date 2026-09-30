"""Gradient checkpointing (activation recomputation) for long sequences.

checkpoint(fn, *args) runs fn without recording a graph and only keeps the
segment's inputs/outputs.  During backward() the segment is re-executed with
gradient recording enabled to rebuild the local graph on demand, trading
compute for memory.

Guarantees:

* The global RNG state is captured at the segment boundary and restored
  before recomputation, so random ops (dropout, randn ...) draw the *same*
  samples as in the original forward, and the generator is not advanced a
  second time -- recomputation has no externally visible side effects.
* Inputs are re-checked by version: mutating a parameter in place after the
  forward pass still invalidates the graph and raises RuntimeError.
* Gradients of parameters shared across segments / branches accumulate
  exactly as in the fully-retained graph.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

import weakref

from .core import _ALL_TENSORS, Function, Tensor, _grad_enabled, _restore_rng, _rng_state, no_grad


class _CheckpointFn(Function):
    """Backward node that recomputes its segment instead of storing it."""

    def __init__(self, fn: Callable[..., Tensor], rng_state: dict, versions):
        self.fn = fn
        self.rng_state = rng_state
        # (weakref, version) of every grad-requiring tensor alive at forward;
        # catches in-place updates to tensors closed over by fn.
        self._versions = versions

    def backward(self, grad_out: np.ndarray):
        for ref, version in self._versions:
            t = ref()
            if t is not None and t._version != version:
                raise RuntimeError(
                    'one of the tensors needed by this graph was modified in place; '
                    'the graph is no longer valid for backward()'
                )
        # Replay the segment with the same random draws as the first pass.
        _restore_rng(self.rng_state)
        detached = [
            Tensor(p.data.copy(), requires_grad=p.requires_grad) for p in self.parents
        ]
        out = self.fn(*detached)
        if not isinstance(out, Tensor) or not out.requires_grad:
            raise RuntimeError(
                "checkpointed function must return a single Tensor that requires grad"
            )
        out.backward(grad_out)
        grads = []
        for d in detached:
            grads.append(d.grad.data if d.grad is not None else np.zeros_like(d.data))
        return tuple(grads)


def checkpoint(fn: Callable[..., Tensor], *args, requires_grad: bool | None = None) -> Tensor:
    """Run fn(*args) saving only inputs; recompute internals in backward.

    fn must be a pure function of its tensor arguments (plus any closed-over
    parameters) and return a single Tensor.

    Staleness (in-place modification) is checked for the tensor *arguments*;
    closed-over tensors still receive gradients during recomputation but are
    not version-checked.  Pass requires_grad=True to force a differentiable
    output when only closed-over parameters require grad.
    """
    args = tuple(a if isinstance(a, Tensor) else Tensor(np.asarray(a, dtype=np.float64)) for a in args)
    rng_state = _rng_state()
    with no_grad():
        out = fn(*args)
    if not isinstance(out, Tensor):
        raise TypeError("checkpointed function must return a Tensor")
    if requires_grad is None:
        requires = _grad_enabled() and any(a.requires_grad for a in args)
    else:
        requires = _grad_enabled() and requires_grad
    versions = [(weakref.ref(t), t._version) for t in _ALL_TENSORS if t.requires_grad]
    node = _CheckpointFn(fn, rng_state, versions)
    node.parents = args
    node.input_versions = [a._version for a in args]
    return Tensor(out.data.copy(), requires_grad=requires, grad_fn=node if requires else None)


def checkpoint_sequential(modules: Sequence, x: Tensor, segments: int) -> Tensor:
    """Chain modules over x, checkpointing every group of layers.

    segments splits the module list into that many (as equal as possible)
    recomputation intervals; segments=1 checkpoints the whole chain.
    """
    modules = list(modules)
    if not modules:
        return x
    segments = max(1, min(int(segments), len(modules)))
    base, extra = divmod(len(modules), segments)
    groups, i = [], 0
    for s in range(segments):
        size = base + (1 if s < extra else 0)
        groups.append(modules[i:i + size])
        i += size

    def make_segment(group):
        def segment(t):
            for layer in group:
                t = layer(t)
            return t
        return segment

    for group in groups:
        x = checkpoint(make_segment(group), x)
    return x
