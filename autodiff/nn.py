"""Tiny neural-network building blocks and losses on top of the engine."""

from __future__ import annotations

from typing import List

import numpy as np

from . import ops
from .tensor import Tensor, as_tensor


class Parameter(Tensor):
    """A leaf tensor that an optimizer should update."""

    def __init__(self, data, dtype=np.float64):
        super().__init__(data, requires_grad=True, dtype=dtype)


class Module:
    """Minimal ``Module``: parameter registration and recursive collection."""

    def __init__(self):
        self._parameters: dict[str, Parameter] = {}
        self._modules: dict[str, Module] = {}

    def __setattr__(self, name: str, value):
        if isinstance(value, Parameter):
            self.__dict__.setdefault("_parameters", {})[name] = value
        elif isinstance(value, Module):
            self.__dict__.setdefault("_modules", {})[name] = value
        super().__setattr__(name, value)

    def parameters(self) -> List[Parameter]:
        seen: set[int] = set()
        out: List[Parameter] = []

        def walk(module: Module) -> None:
            for p in module._parameters.values():
                if id(p) not in seen:
                    seen.add(id(p))
                    out.append(p)
            for child in module._modules.values():
                walk(child)

        walk(self)
        return out

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


class Linear(Module):
    """``y = x @ W + b`` (Kaiming-uniform weights, zero bias)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        bound = np.sqrt(6.0 / in_features)
        self.weight = Parameter(
            np.random.uniform(-bound, bound, size=(in_features, out_features))
        )
        if bias:
            self.bias = Parameter(np.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias  # bias broadcasts over the batch dimension
        return out


class Sequential(Module):
    def __init__(self, *layers: Module):
        super().__init__()
        for i, layer in enumerate(layers):
            setattr(self, f"layer{i}", layer)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self._modules.values():
            x = layer(x)
        return x


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.tanh(x)


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.sigmoid(x)


# ------------------------------------------------------------------- losses
def mse_loss(pred: Tensor, target) -> Tensor:
    """Mean squared error over every element."""
    target = as_tensor(target)
    diff = pred - target
    return (diff * diff).mean()


def binary_cross_entropy_with_logits(logits: Tensor, target) -> Tensor:
    """Numerically stable BCE: ``max(z,0) - z*y + log(1+exp(-|z|))``."""
    target = as_tensor(target)
    relu_logits = ops.relu(logits)
    abs_z = (logits * logits) ** 0.5
    log_term = ops.log(1.0 + ops.exp(-abs_z))
    loss = relu_logits - logits * target + log_term
    return loss.mean()


__all__ = [
    "Parameter", "Module", "Linear", "Sequential",
    "ReLU", "Tanh", "Sigmoid",
    "mse_loss", "binary_cross_entropy_with_logits",
]
