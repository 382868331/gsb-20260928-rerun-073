"""Run independent finite-difference gradient checks on a mixed expression.

    python -m examples.gradcheck_demo
"""

from __future__ import annotations

import numpy as np

from autodiff import Tensor, ops
from autodiff.grad_check import check_gradients


def main() -> None:
    rng = np.random.default_rng(42)
    x = Tensor(rng.normal(size=(4, 3)), requires_grad=True)
    w = Tensor(rng.normal(size=(3, 2)), requires_grad=True)
    b = Tensor(rng.normal(size=(2,)), requires_grad=True)

    def fn():
        h = ops.tanh(x @ w + b)
        shared = h * h
        return (shared.sum() + shared.mean() + ops.exp(-h).sum())

    report = check_gradients(fn, [x, w, b], eps=1e-6, atol=1e-6, rtol=1e-4)
    for name, max_abs, max_rel in report:
        print(f"{name}: max_abs_err={max_abs:.3e}  max_rel_err={max_rel:.3e}")
    print("all analytic gradients match central finite differences")


if __name__ == "__main__":
    main()
