# Reproducing the results

All Python commands MUST go through the isolated environment:

```bash
micromamba run -n test-py312 python ...
```

Run everything from the repository root.

## 0. Install dependencies

Runtime dependencies are already present in `test-py312`. To recreate:

```bash
micromamba run -n test-py312 pip install -r requirements.txt
micromamba run -n test-py312 pip install -r requirements-dev.txt
```

## 1. Tests (fast, < 2 s measured)

```bash
cd /home/yangsch/RLmalloc
micromamba run -n test-py312 python -m pytest -q tests
```

Measured: `21 passed in ~1s` (pytest 9.1.1).

## 2. Bounded smoke training (quick profile)

```bash
micromamba run -n test-py312 python -m rlmalloc.train \
  --dist lognormal_train --episodes 300 --max-steps 200 --learn-every 2 \
  --seed 0 --device cpu \
  --out results/checkpoints/agent_quick \
  --log results/metrics/train_log_quick.csv
```

Measured on CPU: **~2 s** (300 episodes, 439 env steps). Episodes are short
early on because invalid actions terminate the episode while `epsilon` is
high (see `docs/DEVIATIONS.md` §5).

## 3. Full paper-scale training

```bash
# 10000 episodes. The paper does not state a wall-clock budget.
# CPU estimate: ~10 ms/step; early episodes are short (invalid-action
# termination), late episodes run up to hundreds of steps.
micromamba run -n test-py312 python -m rlmalloc.train \
  --dist lognormal_train --episodes 10000 --max-steps 2000 \
  --seed 0 --device cuda \
  --out results/checkpoints/agent \
  --log results/metrics/train_log.csv
```

**Measured reference run (this repo, RTX 5060 Laptop GPU):**
`episodes=10000 steps=568654 final_eps=0.0100 mean_return_last100=74.345
elapsed=1275.6s` (~21.3 min). Final checkpoint at
`results/checkpoints/agent.pt`.

On pure CPU the same run is roughly an order of magnitude slower; budget
hours and prefer the bounded quick profile first.

## 4. Evaluation (4 policies x 4 distributions)

Full protocol (1000 rounds, paper p.4 IV.B):

```bash
micromamba run -n test-py312 python -m rlmalloc.evaluate \
  --checkpoint results/checkpoints/agent --seed 0 --device cpu \
  --rounds 1000 --requests 200 --release-rate 0.3 \
  --dists lognormal_train lognormal_large uniform bimodal \
  --out-metrics results/metrics --out-tables results/tables \
  --figures results/figures
```

Bounded preview (same code path):

```bash
micromamba run -n test-py312 python -m rlmalloc.evaluate \
  --checkpoint results/checkpoints/agent --seed 0 --device cpu \
  --rounds 50 --requests 200 --release-rate 0.3 \
  --dists lognormal_train lognormal_large uniform bimodal \
  --out-metrics results/metrics_preview --out-tables results/tables_preview \
  --figures results/figures_preview
```

**Measured reference run:** the full 1000-round x 4-distribution x 4-policy
evaluation takes **~22 s** on CPU (run-to-run variation of a few seconds).

## 5. Outputs

| Artifact | Path |
|---|---|
| Training curve CSV | `results/metrics/train_log.csv` |
| Per-round metrics | `results/metrics/eval_per_round.csv` |
| Summary CSV / JSON | `results/metrics/eval_summary.csv`, `.json` |
| Markdown table | `results/tables/summary.md` |
| Figures | `results/figures/fig3_*`, `fig4_*`, `fig5_*` |
| Checkpoints | `results/checkpoints/agent.pt`, `_target.pt`, `_meta.json` |
| Legacy output copy | `results/original/result.txt` |

## 6. Quick shell wrappers

```bash
bash scripts/train_quick.sh
bash scripts/train_full.sh      # long
bash scripts/evaluate.sh        # needs a trained checkpoint
```
