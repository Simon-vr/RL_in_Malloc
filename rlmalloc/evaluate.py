"""Evaluation / benchmark entrypoint.

Fair protocol (paper p.4 IV.A): for each round we generate ONE event list and
run every policy on that exact same list.  Rounds are reproducible via
``make_rng(seed, round_idx)``.

Example
-------
    python -m rlmalloc.evaluate \
        --checkpoint results/checkpoints/agent --seed 0 --device cpu \
        --rounds 1000 --requests 200 --release-rate 0.3 \
        --dists lognormal_train lognormal_large uniform bimodal \
        --out-metrics results/metrics --out-tables results/tables \
        --figures results/figures
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np

from . import config
from .agent import DQNAgent
from .env import MemoryEnv
from .metrics import compute_metrics
from .policies import DISPLAY_NAMES, get_heuristic
from .plotting import plot_comparison, plot_dist_grid, plot_histogram
from .utils import ensure_dir, make_rng, save_json, set_seed, write_csv
from .workloads import generate_workload, sample_request_by_name

METRICS = ["occupancy", "duration", "fragmentation", "hhi", "hhi_complement"]


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark RLmalloc policies.")
    p.add_argument("--checkpoint", default=None,
                   help="DQN checkpoint prefix (without .pt). Optional.")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--rounds", type=int, default=config.N_TEST_ROUNDS)
    p.add_argument("--requests", type=int,
                   default=config.N_REQUESTS_PER_ROUND)
    p.add_argument("--release-rate", type=float, default=config.RELEASE_RATE)
    p.add_argument("--dists", nargs="+", default=config.EVAL_DISTS,
                   choices=sorted(config.DISTRIBUTIONS))
    p.add_argument("--out-metrics", default="results/metrics")
    p.add_argument("--out-tables", default="results/tables")
    p.add_argument("--figures", default="results/figures")
    p.add_argument("--max-steps", type=int,
                   default=config.MAX_STEPS_PER_EPISODE)
    return p.parse_args(argv)


def _run_episode(env: MemoryEnv, policy: str, agent=None,
                 max_steps: int = config.MAX_STEPS_PER_EPISODE):
    state = env.reset()
    done = env.is_done()
    steps = 0
    while not done and steps < max_steps:
        if policy == "dqn":
            action = agent.select_action(state, is_test=True).item()
            state, _reward, done_t = env.step(action)
        else:
            block = get_heuristic(policy)(env.free_blocks,
                                          env.current_request_size)
            if block is None:
                break
            state, _reward, done_t = env.allocate_block(*block)
        done = bool(done_t.item())
        steps += 1

    occupancy, fragmentation, hhi, hhi_complement = compute_metrics(
        env.free_blocks, env.memory_size
    )
    return {
        "occupancy": occupancy,
        "duration": float(env.duration),
        "fragmentation": fragmentation,
        "hhi": hhi,
        "hhi_complement": hhi_complement,
    }


def evaluate(args: argparse.Namespace) -> Dict:
    set_seed(args.seed)
    ensure_dir(args.out_metrics)
    ensure_dir(args.out_tables)
    ensure_dir(args.figures)

    agent = None
    policies: List[str] = []
    if args.checkpoint:
        if not os.path.exists(args.checkpoint + ".pt"):
            raise FileNotFoundError(
                f"checkpoint not found: {args.checkpoint}.pt"
            )
        agent = DQNAgent(device=args.device, seed=args.seed)
        agent.load(args.checkpoint)
        agent.q_network.eval()
        policies.append("dqn")
    policies.extend(["first_fit", "best_fit", "worst_fit"])

    per_round_rows: List[Dict] = []
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}

    for dist in args.dists:
        summary[dist] = {}
        # sample some requests for the fig-4 distribution plot
        fig_rng = make_rng(args.seed, 999_000)
        samples = [sample_request_by_name(dist, fig_rng) for _ in range(5000)]
        plot_histogram(samples,
                       os.path.join(args.figures, f"fig4_{dist}.png"),
                       os.path.join(args.figures, f"fig4_{dist}.svg"),
                       title=f"{dist} request distribution")

        for policy in policies:
            summary[dist][policy] = {}

        for r in range(args.rounds):
            rng = make_rng(args.seed, r)
            events = generate_workload(
                dist, n_allocs=args.requests,
                release_rate=args.release_rate, rng=rng,
            )
            for policy in policies:
                env = MemoryEnv(dist=dist, request_sequence=events)
                metrics = _run_episode(env, policy, agent=agent,
                                       max_steps=args.max_steps)
                row = {"dist": dist, "round": r, "policy": policy}
                row.update(metrics)
                per_round_rows.append(row)

        for policy in policies:
            rows = [x for x in per_round_rows
                    if x["dist"] == dist and x["policy"] == policy]
            for metric in METRICS:
                vals = np.array([x[metric] for x in rows], dtype=float)
                summary[dist][policy][metric + "_mean"] = float(vals.mean())
                summary[dist][policy][metric + "_std"] = float(vals.std())

        print(f"[eval] {dist}: "
              + ", ".join(
                  f"{DISPLAY_NAMES[m]} occ={summary[dist][m]['occupancy_mean']:.4f} "
                  f"dur={summary[dist][m]['duration_mean']:.2f} "
                  f"hhi={summary[dist][m]['hhi_mean']:.4f}"
                  for m in policies
              ))

    # ---- long CSV ----
    write_csv(per_round_rows, os.path.join(args.out_metrics,
                                           "eval_per_round.csv"))

    # ---- summary CSV ----
    summary_rows: List[Dict] = []
    for dist in args.dists:
        for policy in policies:
            row = {"dist": dist, "policy": policy,
                   "policy_display": DISPLAY_NAMES[policy]}
            row.update(summary[dist][policy])
            summary_rows.append(row)
    write_csv(summary_rows,
              os.path.join(args.out_metrics, "eval_summary.csv"))

    # ---- summary JSON ----
    save_json({
        "config": {
            "seed": args.seed, "rounds": args.rounds,
            "requests": args.requests, "release_rate": args.release_rate,
            "checkpoint": args.checkpoint,
        },
        "summary": summary,
    }, os.path.join(args.out_metrics, "eval_summary.json"))

    # ---- markdown table ----
    write_summary_markdown(summary, policies, args.dists,
                           os.path.join(args.out_tables, "summary.md"))

    # ---- figures ----
    for dist in args.dists:
        plot_dist_grid(DISPLAY_NAMES, dist, summary[dist],
                       os.path.join(args.figures, f"fig5_comparison_{dist}.png"),
                       os.path.join(args.figures, f"fig5_comparison_{dist}.svg"))
    for metric in ["occupancy", "duration", "fragmentation", "hhi"]:
        plot_comparison(summary, metric + "_mean",
                        os.path.join(args.figures, f"fig5_{metric}_all.png"),
                        os.path.join(args.figures, f"fig5_{metric}_all.svg"),
                        label=metric)

    return {"summary": summary, "policies": policies, "dists": args.dists}


def write_summary_markdown(summary, policies, dists, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    lines = ["# Evaluation summary", "",
             "Mean over rounds. HHI follows the paper convention "
             "(`sum((l_i/S)^2)`, higher = more concentrated).", ""]
    header = ["distribution", "policy", "occupancy", "duration",
              "fragmentation", "hhi", "hhi_complement"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for dist in dists:
        for policy in policies:
            s = summary[dist][policy]
            lines.append("| {} | {} | {:.4f} | {:.2f} | {:.4f} | {:.4f} | {:.4f} |".format(
                dist, DISPLAY_NAMES[policy],
                s["occupancy_mean"], s["duration_mean"],
                s["fragmentation_mean"], s["hhi_mean"],
                s["hhi_complement_mean"],
            ))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    evaluate(args)


if __name__ == "__main__":
    main()
