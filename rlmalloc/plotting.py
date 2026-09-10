"""Figure generation (no import side effects).

All plots are produced from real run output; nothing is hand-written.  The
matplotlib backend is forced to "Agg" so the module is safe on headless
machines.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe

import matplotlib.pyplot as plt  # noqa: E402

from .policies import DISPLAY_NAMES  # noqa: E402


def _save(fig, out_png: str, out_svg: str | None = None) -> None:
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    if out_svg:
        fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


def plot_histogram(
    samples,
    out_png: str,
    out_svg: str | None = None,
    title: str = "Allocation Size Distribution",
    bins: int = 64,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(samples, bins=bins, color="skyblue", edgecolor="black")
    ax.set_xlabel("Allocation Size (Bytes)")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    _save(fig, out_png, out_svg)


def plot_distributions(
    samples_by_dist: Dict[str, Sequence[int]],
    out_png: str,
    out_svg: str | None = None,
) -> None:
    n = len(samples_by_dist)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.2), squeeze=False)
    for ax, (name, samples) in zip(axes[0], samples_by_dist.items()):
        ax.hist(samples, bins=64, color="skyblue", edgecolor="black")
        ax.set_title(name)
        ax.set_xlabel("Size (B)")
    axes[0][0].set_ylabel("Frequency")
    fig.suptitle("Test request distributions")
    _save(fig, out_png, out_svg)


def plot_dist_grid(
    display_names: Dict[str, str],
    dist_name: str,
    per_policy_metrics: Dict[str, Dict[str, float]],
    out_png: str,
    out_svg: str | None = None,
) -> None:
    """2x2 grid of metric bars for one distribution.

    ``per_policy_metrics[policy]["<metric>_mean"]`` carries the values.
    """
    metrics = ["occupancy", "duration", "fragmentation", "hhi"]
    policies = list(per_policy_metrics.keys())
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, metric in zip(axes.ravel(), metrics):
        vals = [per_policy_metrics[p][metric + "_mean"] for p in policies]
        errs = [per_policy_metrics[p].get(metric + "_std", 0.0)
                for p in policies]
        ax.bar([display_names.get(p, p) for p in policies], vals,
               yerr=errs, capsize=3, color="steelblue")
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle(f"Policy comparison -- {dist_name}")
    _save(fig, out_png, out_svg)


def plot_comparison(
    summary: Dict[str, Dict[str, Dict[str, float]]],
    metric: str,
    out_png: str,
    out_svg: str | None = None,
    title: str | None = None,
    label: str | None = None,
) -> None:
    """Grouped bar chart: policies (bars) x distributions (groups).

    ``summary[dist][policy][metric] = value``.
    """
    dists = list(summary.keys())
    policies = list(next(iter(summary.values())).keys())

    n_groups = len(dists)
    n_bars = len(policies)
    width = 0.8 / max(n_bars, 1)
    x = range(n_groups)

    fig, ax = plt.subplots(figsize=(2 + 1.4 * n_groups, 4))
    for j, pol in enumerate(policies):
        vals = [summary[d][pol][metric] for d in dists]
        offset = (j - (n_bars - 1) / 2) * width
        ax.bar([xi + offset for xi in x], vals, width,
               label=DISPLAY_NAMES.get(pol, pol))
    ax.set_xticks(list(x))
    ax.set_xticklabels(dists, rotation=15, ha="right")
    ax.set_ylabel(label or metric)
    ax.set_title(title or f"{label or metric} across distributions")
    ax.legend(fontsize=8)
    _save(fig, out_png, out_svg)


if __name__ == "__main__":  # pragma: no cover - manual demo
    import numpy as np

    from .workloads import sample_request_by_name
    from .config import DISTRIBUTIONS

    rng = np.random.default_rng(0)
    demo = np.array([
        sample_request_by_name("lognormal_train", rng) for _ in range(10000)
    ])
    os.makedirs("results/figures", exist_ok=True)
    plot_histogram(
        demo,
        "results/figures/fig3_train_hist.png",
        "results/figures/fig3_train_hist.svg",
        title="Lognormal(ln32, 0.9), clipped [1,512]",
    )
    print("wrote results/figures/fig3_train_hist.png")
