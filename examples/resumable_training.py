"""Resumable training with a durable external-event journal.

Run twice:

    python examples/resumable_training.py            # trains steps 0..9, then crashes
    python examples/resumable_training.py --resume   # reopens journal, never
                                                     # duplicates external effects

The model parameters and optimizer/RNG state are checkpoints written after
each step; the event journal (events.jsonl) makes external effects
idempotent across recomputation and process restart.
"""
import argparse
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import autodiff as ad
from autodiff import EventLog, checkpoint, manual_seed, nn, record_external_event, tensor, use_event_log
from autodiff.core import _rng_state, _restore_rng

WORKDIR = pathlib.Path(__file__).resolve().parent / "runs"
MODEL_CKPT = WORKDIR / "model.pkl"
EVENT_LOG = WORKDIR / "events.jsonl"
N_STEPS = 10
CRASH_AFTER = 4


def make_data(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((32, 4))
    W = rng.standard_normal((4, 3))
    y = (X @ W + 0.1 * rng.standard_normal((32, 3))).argmax(axis=1)
    return X, y


def save_training_state(step, model, opt, rng_state):
    payload = {
        "step": step,
        "model": model.state_dict(),
        "rng": rng_state,
        "opt_t": opt.t,
    }
    tmp = MODEL_CKPT.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
        f.flush()
    tmp.replace(MODEL_CKPT)  # atomic rename; the event journal stays append-only


def load_training_state(model, opt):
    with open(MODEL_CKPT, "rb") as f:
        payload = pickle.load(f)
    model.load_state_dict(payload["model"])
    _restore_rng(payload["rng"])
    opt.t = payload["opt_t"]
    return payload["step"]


# A stand-in for a remote/external batch service: in production this would
# be a network call. It MUST only run once per event id.
def remote_fetch_minibatch(step):
    print(f"  [external] fetching minibatch for step {step}")
    rng = np.random.default_rng(1000 + step)
    return {"X": rng.standard_normal((8, 4)), "y": rng.integers(0, 3, 8)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    WORKDIR.mkdir(exist_ok=True)
    manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 16, seed=1), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 3, seed=2))
    opt = ad.optim.Adam(model.parameters(), lr=0.05)
    start_step = 0

    if args.resume and MODEL_CKPT.exists():
        start_step = load_training_state(model, opt)
        print(f"resumed after step {start_step}")

    X_eval, y_eval = make_data()
    log = EventLog(EVENT_LOG)  # raises PendingExternalEventError if a prior
                               # effect is unresolved

    for step in range(start_step, N_STEPS):
        opt.zero_grad()

        def segment(t, step=step):
            batch = record_external_event(f"step-{step}-minibatch", lambda: remote_fetch_minibatch(step))
            xt, yt = tensor(batch["X"]), np.asarray(batch["y"])
            return nn.softmax_ce_loss(model(xt), yt) + 0.0 * t.sum()

        with use_event_log(log):
            loss = checkpoint(segment, tensor(X_eval), requires_grad=True)
        loss.backward()  # recomputes segment; remote fetch is NOT repeated
        opt.step()
        save_training_state(step + 1, model, opt, _rng_state())
        print(f"step {step:2d}  loss={float(loss.data):.4f}  "
              f"events={len(log.events())} replays={len(log.replays())}")

        if not args.resume and step == CRASH_AFTER:
            print("simulated crash (restart with --resume; committed events are not refetched)")
            return


if __name__ == "__main__":
    main()
