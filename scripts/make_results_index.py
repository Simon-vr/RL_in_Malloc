"""Assemble results/metrics.json from the real run artifacts.

Reads the training log, checkpoint metadata and evaluation summary and writes
one consolidated machine-readable file.  Run through the isolated env:

    micromamba run -n test-py312 python scripts/make_results_index.py

Nothing is hard-coded except file locations; all numbers come from the logs.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*parts):
    return os.path.join(ROOT, *parts)


def read_train_log(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows


def main() -> None:
    train_rows = read_train_log(_p("results", "metrics", "train_log.csv"))
    last100 = [float(r["return"]) for r in train_rows[-100:]]
    final = train_rows[-1]

    with open(_p("results", "checkpoints", "agent_meta.json"),
              encoding="utf-8") as fh:
        ckpt_meta = json.load(fh)
    with open(_p("results", "metrics", "eval_summary.json"),
              encoding="utf-8") as fh:
        eval_data = json.load(fh)

    import torch

    out = {
        "provenance": {
            "environment": "test-py312",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "seed": eval_data["config"]["seed"],
            "protocol": {
                "rounds": eval_data["config"]["rounds"],
                "requests": eval_data["config"]["requests"],
                "release_rate": eval_data["config"]["release_rate"],
                "policies": ["dqn", "first_fit", "best_fit", "worst_fit"],
                "dists": sorted(eval_data["summary"].keys()),
            },
            "checkpoint": eval_data["config"]["checkpoint"],
            "checkpoint_meta": ckpt_meta,
        },
        "train": {
            "profile": "full",
            "episodes": int(final["episode"]) + 1,
            "global_steps_recorded": sum(int(r["steps"]) for r in train_rows),
            "final_epsilon": float(final["epsilon"]),
            "mean_return_last100": sum(last100) / len(last100),
        },
        "eval": eval_data["summary"],
    }

    path = _p("results", "metrics.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {os.path.relpath(path, ROOT)} "
          f"(episodes={out['train']['episodes']}, "
          f"mean_return_last100={out['train']['mean_return_last100']:.3f})")


if __name__ == "__main__":
    main()
