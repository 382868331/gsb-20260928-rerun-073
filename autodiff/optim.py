"""Optimizers.

Updates are performed *between* ``backward()`` calls, after the previous
graph has been released (the default) or is intentionally retained.  Each
update bumps the parameter's version, so any stale graph referencing it will
be rejected on a later backward instead of silently using clobbered data.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .nn import Parameter


class Optimizer:
    def __init__(self, params: Iterable[Parameter]):
        self.params = list(params)
        if not self.params:
            raise ValueError("optimizer got an empty parameter list")

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    @staticmethod
    def _assign(param: Parameter, new_value: np.ndarray) -> None:
        """Sanctioned in-place leaf update (bumps version, invalidates stale graph)."""
        param._data.replace(new_value)
        param._version += 1
        param.grad = None

    def step(self) -> None:  # pragma: no cover
        raise NotImplementedError


class SGD(Optimizer):
    """Mini-batch SGD with optional momentum and weight decay (L2)."""

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        super().__init__(params)
        if lr <= 0:
            raise ValueError("learning rate must be positive")
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity = [None] * len(self.params)

    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            update = p.grad
            if self.weight_decay:
                update = update + self.weight_decay * p.data
            if self.momentum:
                if self._velocity[i] is None:
                    buf = update.copy()
                else:
                    buf = self.momentum * self._velocity[i] + update
                self._velocity[i] = buf
                update = buf
            self._assign(p, p.data - self.lr * update)


class Adam(Optimizer):
    """Adam (Kingma & Ba, 2015) with bias-corrected first/second moments."""

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [np.zeros_like(p.data) for p in self.params]
        self._t = 0

    def step(self) -> None:
        self._t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad
            if self.weight_decay:
                grad = grad + self.weight_decay * p.data
            self._m[i] = self.beta1 * self._m[i] + (1.0 - self.beta1) * grad
            self._v[i] = self.beta2 * self._v[i] + (1.0 - self.beta2) * grad * grad
            m_hat = self._m[i] / (1.0 - self.beta1 ** self._t)
            v_hat = self._v[i] / (1.0 - self.beta2 ** self._t)
            self._assign(p, p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps))


__all__ = ["Optimizer", "SGD", "Adam"]
