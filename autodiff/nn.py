"""Tiny neural-network building blocks and loss functions."""
from __future__ import annotations

from typing import Iterator

import numpy as np

from .core import Function, Tensor
from .ops import add, matmul, mean, relu, sigmoid


class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self) -> list[Tensor]:
        params: list[Tensor] = []
        for value in vars(self).values():
            if isinstance(value, Tensor):
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def state_dict(self) -> dict:
        """Snapshot parameters as plain numpy arrays (name -> array)."""
        return {f"param_{i}": p.data.copy() for i, p in enumerate(self.parameters())}

    def load_state_dict(self, state: dict) -> None:
        """Restore parameters; assigning .data bumps versions, so any graph
        built before the load is invalidated and cannot be backpropagated."""
        params = self.parameters()
        if len(state) != len(params):
            raise ValueError(f"state has {len(state)} entries, model has {len(params)} parameters")
        for i, p in enumerate(params):
            value = np.asarray(state[f"param_{i}"], dtype=np.float64)
            if value.shape != p.shape:
                raise ValueError(f"shape mismatch for param_{i}: {value.shape} vs {p.shape}")
            p.data = value


class Linear(Module):
    """y = x @ W + b; W: (in_features, out_features), b: (out_features,)."""

    def __init__(self, in_features: int, out_features: int, seed: int | None = None):
        rng = np.random.default_rng(seed)
        bound = np.sqrt(1.0 / in_features)
        self.weight = Tensor(rng.uniform(-bound, bound, (in_features, out_features)), requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return add(matmul(x, self.weight), self.bias)


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return relu(x)


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return sigmoid(x)



class Dropout(Module):
    """Inverted dropout; draws from the global RNG (seed with manual_seed).

    Inside checkpoint segments the mask is an internal recorded event and is
    replayed during recomputation without advancing the global RNG again.
    """

    def __init__(self, p: float = 0.5):
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout probability must be in [0, 1)")
        self.p = float(p)
        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        from .core import rand
        mask = rand(*x.shape).data >= self.p
        return x * Tensor(mask.astype(np.float64) / (1.0 - self.p))


class Sequential(Module):
    def __init__(self, *layers: Module):
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ------------------------------------------------------------------- losses
def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean squared error over all elements (scalar output)."""
    diff = pred + (-target)
    return mean(diff * diff)


def binary_cross_entropy_with_logits(logits: Tensor, target: Tensor) -> Tensor:
    """Numerically stable BCE mean: max(z,0) - z*y + log(1+e^{-|z|})."""
    z = logits
    pos = relu(z)
    from .ops import exp, log
    loss = pos - z * target + log(1.0 + exp(-abs_tensor(z)))
    return mean(loss)


def abs_tensor(t: Tensor) -> Tensor:
    return relu(t) + relu(-t)


def softmax_ce_loss(logits: Tensor, labels: np.ndarray) -> Tensor:
    """Mean softmax cross entropy with fused gradient:
    dL/dlogits = (softmax(logits) - onehot(labels)) / N.
    """
    return SoftmaxCELoss.apply(logits, labels=np.asarray(labels))


class SoftmaxCELoss(Function):
    @staticmethod
    def forward(ctx, a, labels):
        ctx.labels = labels
        shifted = a - a.max(axis=-1, keepdims=True)
        e = np.exp(shifted)
        probs = e / e.sum(axis=-1, keepdims=True)
        ctx.probs = probs
        n = a.shape[0]
        return -np.log(probs[np.arange(n), labels]).mean()

    def backward(self, g):
        n = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(n), self.labels] -= 1.0
        return (g * grad / n,)
