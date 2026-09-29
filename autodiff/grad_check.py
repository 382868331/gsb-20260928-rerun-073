"""Independent gradient checking via central finite differences.

The numerical check shares no code with the autograd backward pass: it only
evaluates the forward function on perturbed scalar coordinates, so it is a
genuinely independent cross-check.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .tensor import Tensor


def numerical_gradient(
    fn: Callable[[], Tensor],
    param: Tensor,
    eps: float = 1e-6,
) -> np.ndarray:
    """Central-difference gradient of scalar ``fn()`` w.r.t. every element.

    ``fn`` must return a scalar tensor and must rebuild the graph on each call
    (the usual functional usage).  Buffer replacement is used only to perturb
    the coordinate and restore it afterwards.
    """
    original = param.numpy()
    flat = original.reshape(-1)
    numerical = np.zeros_like(flat)
    for i in range(flat.size):
        plus = flat.copy()
        minus = flat.copy()
        plus[i] += eps
        minus[i] -= eps
        param._data.replace(plus.reshape(original.shape))
        f_plus = float(fn().data)
        param._data.replace(minus.reshape(original.shape))
        f_minus = float(fn().data)
        numerical[i] = (f_plus - f_minus) / (2.0 * eps)
    param._data.replace(original)
    return numerical.reshape(param.shape)


def check_gradients(
    fn: Callable[[], Tensor],
    params: Sequence[Tensor],
    eps: float = 1e-6,
    atol: float = 1e-6,
    rtol: float = 1e-4,
) -> list[tuple[str, float, float]]:
    """Compare analytic vs numeric gradients; return ``(name, max_abs, max_rel)``.

    Raises ``AssertionError`` on mismatch.  ``fn`` is rebuilt for every
    analytic and numeric evaluation.
    """
    report = []
    for idx, param in enumerate(params):
        for p in params:
            p.grad = None
        out = fn()
        out.backward()
        analytic = param.grad.copy()
        for p in params:
            p.grad = None
        numeric = numerical_gradient(fn, param, eps=eps)
        diff = np.abs(analytic - numeric)
        max_abs = float(diff.max())
        scale = np.maximum(np.abs(analytic), np.abs(numeric))
        max_rel = float((diff / np.maximum(scale, 1e-12)).max())
        report.append((f"param{idx}", max_abs, max_rel))
        if not np.allclose(analytic, numeric, atol=atol, rtol=rtol):
            worst = np.unravel_index(np.argmax(diff), diff.shape)
            raise AssertionError(
                f"gradient mismatch for param{idx} at {worst}: "
                f"analytic={analytic[worst]}, numeric={numeric[worst]}, "
                f"max_abs={max_abs:.3e}"
            )
    return report


__all__ = ["numerical_gradient", "check_gradients"]
