"""Train a tiny MLP classifier end-to-end with the autodiff engine.

Run:  python examples/train_tiny.py
"""
import numpy as np

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import autodiff as ad
from autodiff import nn, optim, tensor
from autodiff.nn import softmax_ce_loss


def make_data(n=256, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    W = rng.standard_normal((4, 3))
    labels = (X @ W).argmax(axis=1)
    return X, labels


def main():
    X, labels = make_data()
    model = nn.Sequential(
        nn.Linear(4, 16, seed=1),
        nn.ReLU(),
        nn.Linear(16, 3, seed=2),
    )
    opt = optim.SGD(model.parameters(), lr=0.3, momentum=0.9)
    xt = tensor(X)

    for epoch in range(60):
        opt.zero_grad()
        logits = model(xt)
        loss = softmax_ce_loss(logits, labels)
        loss.backward()
        opt.step()
        if epoch % 10 == 0 or epoch == 59:
            acc = (logits.data.argmax(axis=1) == labels).mean()
            print(f"epoch {epoch:3d}  loss={float(loss.data):.4f}  acc={acc:.3f}")

    # Inference is graph-free under no_grad.
    with ad.no_grad():
        out = model(tensor(X[:5]))
    print("sample predictions:", out.data.argmax(axis=1), "true:", labels[:5])


if __name__ == "__main__":
    main()
