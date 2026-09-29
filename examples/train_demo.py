"""End-to-end training demo: fit a tiny MLP to a synthetic task.

Run with::

    python -m examples.train_demo

The target is ``y = sin(x1) + 0.5 * x2**2``; the network only has access to
raw (x1, x2) features, so this is genuine optimization, not symbol printing.
"""

from __future__ import annotations

import numpy as np

from autodiff import Tensor, nn, optim


def make_dataset(n: int = 256, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=(n, 2))
    y = np.sin(x[:, 0]) + 0.5 * x[:, 1] ** 2
    return Tensor(x), y.reshape(-1, 1)


def main() -> None:
    np.random.seed(0)
    x, y = make_dataset()
    x_train, y_train = x[:200], y[:200]
    x_test, y_test = x[200:], y[200:]

    model = nn.Sequential(
        nn.Linear(2, 32),
        nn.Tanh(),
        nn.Linear(32, 32),
        nn.Tanh(),
        nn.Linear(32, 1),
    )
    optimizer = optim.Adam(model.parameters(), lr=0.02)

    def evaluate() -> float:
        pred = model(x_test)
        return float(nn.mse_loss(pred, y_test).data)

    initial = evaluate()
    for epoch in range(200):
        optimizer.zero_grad()
        pred = model(x_train)
        loss = nn.mse_loss(pred, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 40 == 0 or epoch == 199:
            print(
                f"epoch {epoch:3d}  train_mse={float(loss.data):.6f}  "
                f"test_mse={evaluate():.6f}"
            )

    final = evaluate()
    print(f"test MSE {initial:.5f} -> {final:.5f}")
    assert final < 0.02, "demo failed to converge"
    print("training converged: engine works end to end")


if __name__ == "__main__":
    main()
