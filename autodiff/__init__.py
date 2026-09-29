"""A small define-by-run reverse-mode autodiff engine (NumPy backend).

Typical usage::

    from autodiff import Tensor
    from autodiff import ops

    x = Tensor([[1.0, 2.0]], requires_grad=True)
    y = ops.relu((x @ W + b)).sum()
    y.backward()
    print(x.grad)
"""

from .tensor import AutogradError, Context, Function, Tensor, as_tensor, tensor
from . import ops, nn, optim

__all__ = [
    "Tensor", "Context", "Function", "AutogradError", "as_tensor", "tensor",
    "ops", "nn", "optim",
]
