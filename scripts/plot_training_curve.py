"""Plot the training learning curve from a real train_log.csv.

Usage (must run through the isolated env):
    micromamba run -n test-py312 python scripts/plot_training_curve.py \
        --log results/metrics/train_log.csv \
        --out results/figures/fig3_train_curve

Produces PNG + SVG from the recorded episodes; nothing is fabricated.
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def _rolling_mean(values, window):
    out = []
    acc = 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        out.append(acc / min(i + 1, window))
    return out


def load_log(path):
    cols = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for name in reader.fieldnames or []:
            cols[name] = []
        for row in reader:
            for name in cols:
                cols[name].append(float(row[name]))
    return cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="results/metrics/train_log.csv")
    ap.add_argument("--out", default="results/figures/fig3_train_curve")
    ap.add_argument("--window", type=int, default=100)
    args = ap.parse_args()

    cols = load_log(args.log)
    ep = cols["episode"]
    ret = _rolling_mean(cols["return"], args.window)
    steps = _rolling_mean(cols["steps"], args.window)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(ep, ret, color="tab:blue")
    axes[0].set_ylabel(f"return ({args.window}-ep mean)")
    axes[0].set_title(f"Training curve -- {os.path.basename(args.log)}")
    axes[0].grid(alpha=0.3)

    axes[1].plot(ep, steps, color="tab:green")
    axes[1].set_ylabel(f"env steps ({args.window}-ep mean)")
    axes[1].set_xlabel("episode")
    axes[1].grid(alpha=0.3)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(args.out + ".svg", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}.png/.svg "
          f"(episodes={int(ep[-1])}, final return={ret[-1]:.3f})")


if __name__ == "__main__":
    main()
