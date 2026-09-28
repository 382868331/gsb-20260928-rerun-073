"""Central-difference numerical gradient checking (independent of autograd)."""
import numpy as np


EPS = 1e-6


def numerical_grad(fn, x: np.ndarray, upstream=None, eps=EPS) -> np.ndarray:
    """Central differences; ``fn`` returns a numpy array.

    If the output is non-scalar, ``upstream`` gives the seed vector and the
    gradient of sum(upstream * out) w.r.t. x is returned.
    """
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readonly"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        fp = fn(x)
        x[idx] = old - eps
        fm = fn(x)
        x[idx] = old
        if upstream is None:
            grad[idx] = (fp - fm).sum() / (2 * eps)
        else:
            grad[idx] = ((fp - fm) * upstream).sum() / (2 * eps)
        it.iternext()
    return grad


def assert_close(a: np.ndarray, b: np.ndarray, atol=1e-5, rtol=1e-4, msg=""):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape} {msg}"
    diff = np.abs(a - b)
    tol = atol + rtol * np.maximum(np.abs(a), np.abs(b))
    assert np.all(diff <= tol), (
        f"grad mismatch {msg}\nanalytic=\n{a}\nnumeric=\n{b}\nmax diff={diff.max()}"
    )
