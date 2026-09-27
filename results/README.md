# RLmalloc — experiment artifacts

This directory holds every artifact produced by the re-run of the experiments.
All numbers come from real runs; nothing here is hand-entered. The commands
below were executed from `/home/yangsch/RLmalloc` using only the isolated
environment (`micromamba run -n test-py312 ...`).

## 1. How these artifacts were generated

```bash
# (a) fast test suite — 22 tests
micromamba run -n test-py312 python -m pytest -q tests
# -> 22 passed in ~1s

# (b) bounded smoke training (fresh, deterministic, CPU)
bash scripts/train_quick.sh
# -> episodes=300 steps=439 final_eps=0.7407 mean_return_last100=-0.234 elapsed=1.7s

# (c) full paper-scale evaluation using the committed 10000-episode checkpoint
#     (4 policies x 4 distributions x 1000 rounds x 200 requests)
bash scripts/evaluate.sh
# -> 21.8s wall clock; metrics below

# (d) extra figures + consolidated machine-readable index
micromamba run -n test-py312 python -m rlmalloc.plotting
micromamba run -n test-py312 python scripts/plot_training_curve.py \
    --log results/metrics/train_log.csv --out results/figures/fig3_train_curve
micromamba run -n test-py312 python scripts/make_results_index.py
```

The full 10 000-episode training that produced
`checkpoints/agent.pt` was executed with:

```bash
micromamba run -n test-py312 python -m rlmalloc.train \
  --dist lognormal_train --episodes 10000 --max-steps 2000 \
  --seed 0 --device cuda --out results/checkpoints/agent \
  --log results/metrics/train_log.csv
# -> episodes=10000 steps=568654 final_eps=0.0100
#    mean_return_last100=74.345 elapsed=1275.6s (~21.3 min, RTX 5060 Laptop GPU)
```

`train_log.csv` in this directory is that run's real log. The bounded quick
profile was re-run here to confirm the pipeline, and two identical `--seed 0`
quick runs were diffed: the CSVs are byte-identical (deterministic).

## 2. Artifact catalog

| Path | What it is |
|---|---|
| `metrics.json` | **Consolidated machine-readable results**: provenance (env, protocol, checkpoint meta, train summary) + full eval summary. |
| `metrics/eval_summary.json` | Per-distribution, per-policy mean/std for all 5 metrics. |
| `metrics/eval_summary.csv` | Same summary as a flat CSV (one row per dist×policy). |
| `metrics/eval_per_round.csv` | Long-format raw data: 16 000 rows (4 dists × 1000 rounds × 4 policies). Primary source for any re-analysis. |
| `metrics/train_log.csv` | Full training log, 10 000 episodes (return, steps, epsilon, loss, occupancy). |
| `metrics/train_log_quick.csv` | Bounded quick-profile training log (300 episodes). |
| `tables/summary.md` | Human-readable markdown summary table. |
| `figures/fig3_train_hist.{png,svg}` | Training request-size distribution (lognormal, clipped). |
| `figures/fig3_train_curve.{png,svg}` | Learning curve from the real `train_log.csv` (100-episode rolling mean of return and env steps). |
| `figures/fig4_<dist>.{png,svg}` | Request-size histograms for the four test distributions. |
| `figures/fig5_comparison_<dist>.{png,svg}` | 2×2 policy comparison (occupancy, duration, fragmentation, HHI) per distribution. |
| `figures/fig5_{occupancy,duration,fragmentation,hhi}_all.{png,svg}` | Cross-distribution grouped bars for each metric. |
| `checkpoints/agent.pt`, `agent_target.pt`, `agent_meta.json` | Full trained DQN (10 000 episodes) + target net + metadata. |
| `checkpoints/agent_quick*.pt`, `agent_quick_meta.json` | Bounded quick-profile checkpoint. |
| `original/result.txt` | Copy of the user's original legacy `result.txt` (preserved unchanged). |

Generated `.pt` checkpoints are git-ignored (regenerable); logs, tables and
figures are tracked.

## 3. Measured results (mean over 1000 rounds)

HHI uses the unified paper definition `Σ(lᵢ/S)²` (higher = more concentrated).

| distribution | policy | occupancy | duration | fragmentation | hhi |
|---|---|---|---|---|---|
| lognormal_train | DQN Agent | 0.9489 | 117.28 | 0.6882 | 0.1798 |
| lognormal_train | First-Fit | 0.9535 | 117.86 | 0.6701 | 0.1921 |
| lognormal_train | Best-Fit | 0.9588 | 118.59 | 0.6239 | 0.2311 |
| lognormal_train | Worst-Fit | 0.7838 | 99.22 | 0.8723 | 0.0851 |
| lognormal_large | DQN Agent | 0.9098 | 33.97 | 0.5538 | 0.3230 |
| lognormal_large | First-Fit | 0.9111 | 33.99 | 0.5504 | 0.3262 |
| lognormal_large | Best-Fit | 0.9152 | 34.14 | 0.5277 | 0.3449 |
| lognormal_large | Worst-Fit | 0.8115 | 30.84 | 0.7015 | 0.2269 |
| uniform | DQN Agent | 0.8857 | 20.52 | 0.4581 | 0.4321 |
| uniform | First-Fit | 0.8879 | 20.59 | 0.4483 | 0.4419 |
| uniform | Best-Fit | 0.8929 | 20.71 | 0.4366 | 0.4508 |
| uniform | Worst-Fit | 0.8239 | 19.22 | 0.5529 | 0.3654 |
| bimodal | DQN Agent | 0.9026 | 54.19 | 0.4558 | 0.4080 |
| bimodal | First-Fit | 0.9064 | 54.45 | 0.4237 | 0.4425 |
| bimodal | Best-Fit | 0.9176 | 55.03 | 0.3768 | 0.4921 |
| bimodal | Worst-Fit | 0.7701 | 47.55 | 0.6734 | 0.2389 |

### Reading the numbers honestly

* **Occupancy / duration:** DQN matches First-Fit and trails Best-Fit by
  ~0.003–0.015 occupancy, while beating Worst-Fit by ~0.08–0.13. This agrees
  with the paper's statement that Best-Fit remains the occupancy gold standard.
* **HHI:** we use one definition, the paper's Eq. 1 (`Σ(lᵢ/S)²`, higher =
  more concentrated). Under it DQN sits below First-Fit/Best-Fit and above
  Worst-Fit, consistent with its fragmentation. The paper's printed numbers
  are the opposite direction and are provably not `Σ(lᵢ/S)²`; compute
  `1 − HHI` to compare with its table. See `docs/DEVIATIONS.md` §1.
* **No exact paper reproduction is claimed:** learning rate, episode count,
  ε schedule and the bimodal σ are unspecified in the paper.
* **Quick profile is a smoke test only:** 300 CPU episodes yield near-zero
  return (`-0.234`) and are not a trained policy.

## 4. Provenance

* Environment: conda env `test-py312`, Python and Torch versions recorded in
  `metrics.json` → `provenance.environment`.
* Seed: `0`. Every round uses `make_rng(seed, round_idx)`; all policies in a
  round see the *same* event list (fair paired comparison).
* Eval protocol: 1000 rounds, 200 requests/round, release rate 0.3,
  max 2000 steps/round.
* Checkpoint metadata (`provenance.checkpoint_meta`): 10 000 episodes,
  568 654 env steps, final ε = 0.01.
* Originals preserved and verified byte-identical to
  `backups/pre-refactor/` via `md5sum`
  (PDF `5141a8a1…`, `result.txt` `55d30f95…`).
