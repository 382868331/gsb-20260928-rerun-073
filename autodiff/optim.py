"""Optimizers. Updates run under ``no_grad`` and use the controlled in-place
path (``add_``), which bumps parameter versions and invalidates the graph from
the previous step, as a real framework does."""
from __future__ import annotations

import numpy as np

from .core import Tensor, no_grad


class Optimizer:
    def __init__(self, params, lr: float):
        self.params = list(params)
        if not self.params:
            raise ValueError("optimizer got an empty parameter list")
        self.lr = lr

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr: float = 0.01, momentum: float = 0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self._velocity = None

    def step(self) -> None:
        with no_grad():
            if self._velocity is None:
                self._velocity = [np.zeros_like(p.data) for p in self.params]
            for p, v in zip(self.params, self._velocity):
                if p.grad is None:
                    continue
                if self.momentum:
                    v *= self.momentum
                    v += p.grad.data
                    p.add_(-self.lr * v)
                else:
                    p.add_(-self.lr * p.grad.data)


class Adam(Optimizer):
    def __init__(self, params, lr: float = 0.01, betas=(0.9, 0.999), eps: float = 1e-8):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        with no_grad():
            self.t += 1
            for p, m, v in zip(self.params, self._m, self._v):
                if p.grad is None:
                    continue
                g = p.grad.data
                m *= self.beta1
                m += (1 - self.beta1) * g
                v *= self.beta2
                v += (1 - self.beta2) * g * g
                m_hat = m / (1 - self.beta1 ** self.t)
                v_hat = v / (1 - self.beta2 ** self.t)
                p.add_(-self.lr * m_hat / (np.sqrt(v_hat) + self.eps))
