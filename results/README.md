# Results

Catalog of every artifact under `results/`, plus the measured numbers and the
exact commands that produced them. All Python commands run in the isolated
environment `test-py312`:

```bash
micromamba run -n test-py312 python ...
```

## 1. Artifact index

| Artifact | Description |
|---|---|
| `metrics.json` | Consolidated machine-readable provenance + train + eval summary. |
| `metrics/train_log.csv` | Per-episode training log (10 000 rows): return, steps, duration, epsilon, loss, occupancy. |
| `metrics/train_log_quick.csv` | Per-episode log of the bounded smoke run (300 episodes). |
| `metrics/eval_per_round.csv` | Per-round × policy × distribution metrics (16 000 rows). |
| `metrics/eval_summary.csv` / `.json` | Mean/std summary over rounds (16 dist × policy rows). |
| `tables/summary.md` | Markdown results table. |
| `figures/fig3_train_hist.{png,svg}` | Training request-size histogram. |
| `figures/fig3_train_curve.{png,svg}` | Training learning curve (return / epsilon). |
| `figures/fig4_<dist>.{png,svg}` | Request-size histograms for the four evaluation workloads. |
| `figures/fig5_comparison_<dist>.{png,svg}` | 2×2 policy comparison (occupancy, duration, fragmentation, HHI) per distribution. |
| `figures/fig5_{occupancy,duration,fragmentation,hhi}_all.{png,svg}` | Cross-distribution grouped bars for each metric. |
| `checkpoints/agent.pt`, `agent_target.pt`, `agent_meta.json` | Full trained DQN (10 000 episodes) + target net + metadata. |
| `checkpoints/agent_quick*.pt`, `agent_quick_meta.json` | Bounded quick-profile checkpoint. |
| `original/result.txt` | Archived copy of an earlier output (`result.txt`). |

Generated `.pt` checkpoints are git-ignored (regenerable); logs, tables and
figures are tracked.

## 2. Reproduce

```bash
# (a) fast test suite — 22 tests
micromamba run -n test-py312 python -m pytest -q tests
# -> 22 passed in ~1s

# (b) bounded smoke training (CPU, ~2 s)
bash scripts/train_quick.sh

# (c) full training (10 000 episodes; GPU recommended)
micromamba run -n test-py312 python -m rlmalloc.train \
  --dist lognormal_train --episodes 10000 --max-steps 2000 \
  --seed 0 --device cuda --out results/checkpoints/agent \
  --log results/metrics/train_log.csv

# (d) full evaluation (4 policies x 4 distributions x 1000 rounds, ~22 s)
bash scripts/evaluate.sh

# (e) rebuild consolidated metrics.json
micromamba run -n test-py312 python scripts/make_results_index.py

# (f) extra figures
micromamba run -n test-py312 python -m rlmalloc.plotting
micromamba run -n test-py312 python scripts/plot_training_curve.py \
  --log results/metrics/train_log.csv --out results/figures/fig3_train_curve
```

Measured reference run: `episodes=10000 steps=568654 final_eps=0.0100
mean_return_last100=74.345 elapsed=1275.6s` on an RTX 5060 Laptop GPU.

## 3. Measured results (mean over 1000 rounds)

HHI uses the unified definition `Σ(lᵢ/S)²` (higher = more concentrated).

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

### Interpretation

* **Occupancy / duration / fragmentation:** Best-Fit is best on all three,
  Worst-Fit is worst, and the DQN sits between them close to (very slightly
  below) First-Fit. The learned policy does **not** beat Best-Fit.
* **HHI:** higher = more concentrated. Ordering is Best-Fit > First-Fit > DQN >
  Worst-Fit. The reason Worst-Fit is lowest is that it flattens the free list
  into many similar-sized blocks (see `docs/DESIGN.md` §3).
* **Single seed, no error bars:** the DQN↔First-Fit gaps are small and may not
  be statistically meaningful.
* **Quick profile is a smoke test only:** 300 CPU episodes yield a near-zero
  return and are not a trained policy.

## 4. Provenance

* Environment: conda env `test-py312`; Python / Torch versions recorded in
  `metrics.json` → `provenance`.
* Evaluation protocol: seed 0, 1000 rounds, 200 requests/round, release rate
  0.3, four distributions, policies `dqn`, `first_fit`, `best_fit`,
  `worst_fit`, all replayed on identical per-round workloads.
* Checkpoints are git-ignored; regenerate them with the commands above.
