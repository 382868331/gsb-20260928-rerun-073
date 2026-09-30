"""Gradient checkpointing for long computation sequences.

A checkpointed region is differentiated normally on its first forward pass.
Only its boundary tensors and output values are retained. During backward the
region is replayed to rebuild its temporary graph. Random draws and externally
visible callbacks are recorded on the original forward and replayed, so the
recomputation is deterministic and does not observe side effects twice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .core import Function, Tensor


@dataclass
class _Replay:
    random_draws: list[Tensor] = field(default_factory=list)
    side_effects: list[Any] = field(default_factory=list)
    replay_random: bool = False
    replay_side_effects: bool = False
    random_index: int = 0
    side_effect_index: int = 0
    random_start: int = 0
    side_effect_start: int = 0


_ACTIVE_REPLAY: _Replay | None = None


def checkpoint_randn(*shape, seed: int | None = None, requires_grad: bool = False) -> Tensor:
    """Random draw that is replayed with the same sample during recomputation."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    else:
        shape = tuple(int(s) for s in shape)

    replay = _ACTIVE_REPLAY
    if replay is not None and replay.replay_random:
        sampled = replay.random_draws[replay.random_index]
        replay.random_index += 1
        if sampled.shape != shape or sampled.requires_grad != requires_grad:
            raise RuntimeError("checkpoint recomputation changed its random-draw sequence")
        return Tensor(sampled.data.copy())

    rng = np.random.default_rng(seed)
    result = Tensor(rng.standard_normal(shape), requires_grad=requires_grad)
    if replay is not None:
        replay.random_draws.append(result)
    return result


def checkpoint_side_effect(fn: Callable[[], Any]) -> Any:
    """Run fn on the original forward and return its cached value on replay."""
    replay = _ACTIVE_REPLAY
    if replay is not None and replay.replay_side_effects:
        value = replay.side_effects[replay.side_effect_index]
        replay.side_effect_index += 1
        return value

    value = fn()
    if replay is not None:
        replay.side_effects.append(value)
    return value


class Checkpoint(Function):
    def __init__(self):
        self.fn = None
        self.args = ()
        self.kwargs = {}
        self.replay = _Replay()

    @staticmethod
    def forward(ctx, *args):
        raise RuntimeError("Checkpoint nodes must be created by checkpoint()")

    def backward(self, grad_out):
        raise RuntimeError("Checkpoint nodes use a custom backward implementation")


def checkpoint(fn: Callable[..., Tensor], *args, **kwargs):
    """Checkpoint fn(*args, **kwargs) and return its tensor output."""
    global _ACTIVE_REPLAY
    token = _ACTIVE_REPLAY
    if token is not None and (token.replay_random or token.replay_side_effects):
        replay = _Replay(
            random_draws=token.random_draws,
            side_effects=token.side_effects,
            replay_random=token.replay_random,
            replay_side_effects=token.replay_side_effects,
            random_index=token.random_index,
            side_effect_index=token.side_effect_index,
            random_start=token.random_index,
            side_effect_start=token.side_effect_index,
        )
    else:
        replay = _Replay()
    _ACTIVE_REPLAY = replay
    try:
        outputs = fn(*args, **kwargs)
    finally:
        _ACTIVE_REPLAY = token

    if token is not None and (token.replay_random or token.replay_side_effects):
        token.random_index = replay.random_index
        token.side_effect_index = replay.side_effect_index

    if not isinstance(outputs, Tensor):
        raise TypeError("checkpoint function must return a Tensor")
    output_list = [outputs]

    leaves = _collect_leaves(output_list)
    ctx = Checkpoint()
    ctx.fn = fn
    ctx.args = args
    ctx.kwargs = kwargs
    ctx.parents = tuple(leaves)
    ctx.input_versions = [leaf._version for leaf in leaves]
    ctx.replay = replay

    requires_grad = any(leaf.requires_grad for leaf in leaves)
    detached = Tensor(outputs.data.copy(), requires_grad=requires_grad, grad_fn=ctx if requires_grad else None)
    if not requires_grad:
        ctx.replay.random_draws = []
        ctx.replay.side_effects = []
    return detached


def _collect_leaves(outputs: list[Tensor]) -> list[Tensor]:
    leaves: dict[int, Tensor] = {}
    seen_nodes: set[int] = set()

    def visit_tensor(t: Tensor) -> None:
        if t._grad_fn is None:
            leaves.setdefault(id(t), t)
            return
        visit_node(t._grad_fn)

    def visit_node(node: Function) -> None:
        if id(node) in seen_nodes:
            return
        seen_nodes.add(id(node))
        for parent in node.parents:
            visit_tensor(parent)

    for output in outputs:
        visit_tensor(output)
    return list(leaves.values())


def _backward_checkpoint(node: Checkpoint, upstream: list[np.ndarray]) -> dict[int, np.ndarray]:
    global _ACTIVE_REPLAY
    boundary = {id(t): t for t in node.parents}
    node.replay.random_index = node.replay.random_start
    node.replay.side_effect_index = node.replay.side_effect_start
    node.replay.replay_random = True
    node.replay.replay_side_effects = True
    token = _ACTIVE_REPLAY
    _ACTIVE_REPLAY = node.replay
    try:
        outputs = node.fn(*node.args, **node.kwargs)
    finally:
        _ACTIVE_REPLAY = token
        node.replay.replay_random = False
        node.replay.replay_side_effects = False

    if not isinstance(outputs, Tensor):
        raise RuntimeError("checkpoint recomputation changed its output type")
    output_list = [outputs]
    if len(output_list) != len(upstream) or outputs.data.shape != upstream[0].shape:
        raise RuntimeError("checkpoint recomputation changed its number of outputs")

    order: list[Function] = []
    visited: set[int] = set()

    def visit(fn_node: Function) -> None:
        if id(fn_node) in visited:
            return
        visited.add(id(fn_node))
        for parent in fn_node.parents:
            if parent._grad_fn is not None:
                visit(parent._grad_fn)
        order.append(fn_node)

    node_grads: dict[int, np.ndarray] = {}
    leaf_grads: dict[int, np.ndarray] = {}
    for output, grad in zip(output_list, upstream):
        if output._grad_fn is None:
            if id(output) in boundary:
                leaf_grads[id(output)] = leaf_grads.get(id(output), np.zeros_like(output.data)) + grad
        else:
            visit(output._grad_fn)
            key = id(output._grad_fn)
            node_grads[key] = node_grads.get(key, np.zeros_like(output.data)) + grad

    for fn_node in reversed(order):
        fn_node._check_versions()
        grad = node_grads[id(fn_node)]
        if isinstance(fn_node, Checkpoint):
            nested_grads = _backward_checkpoint(fn_node, [grad])
            for nested_parent in fn_node.parents:
                nested_grad = nested_grads.get(id(nested_parent), np.zeros_like(nested_parent.data))
                if nested_parent._grad_fn is None:
                    if id(nested_parent) in boundary:
                        leaf_grads[id(nested_parent)] = (
                            leaf_grads.get(id(nested_parent), np.zeros_like(nested_parent.data))
                            + nested_grad
                        )
                else:
                    key = id(nested_parent._grad_fn)
                    node_grads[key] = node_grads.get(key, np.zeros_like(nested_parent.data)) + nested_grad
            continue

        parent_grads = fn_node.backward(grad)
        for parent, parent_grad in zip(fn_node.parents, parent_grads):
            if parent_grad is None:
                continue
            if parent_grad.shape != parent.data.shape:
                raise RuntimeError(
                    f"op {type(fn_node).__name__} produced grad shape {parent_grad.shape}, "
                    f"expected {parent.data.shape}"
                )
            if parent._grad_fn is None:
                if id(parent) in boundary:
                    leaf_grads[id(parent)] = (
                        leaf_grads.get(id(parent), np.zeros_like(parent.data)) + parent_grad
                    )
            else:
                key = id(parent._grad_fn)
                node_grads[key] = node_grads.get(key, np.zeros_like(parent.data)) + parent_grad

    return leaf_grads
