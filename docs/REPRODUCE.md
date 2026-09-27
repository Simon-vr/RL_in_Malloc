# Reproducing the results

All Python commands **must** go through the isolated environment `test-py312`,
and should be run from the repository root. This keeps the project from
touching any other Python environment.

```bash
micromamba run -n test-py312 python ...
```

## 0. Dependencies

Runtime dependencies are already present in `test-py312`. To recreate the
environment:

```bash
micromamba run -n test-py312 pip install -r requirements.txt
micromamba run -n test-py312 pip install -r requirements-dev.txt
# or, from scratch:
micromamba env create -f environment.yml      # creates env `test-py312`
```

## 1. Tests (fast, < 2 s measured)

```bash
cd /home/yangsch/RLmalloc
micromamba run -n test-py312 python -m pytest -q tests
```

Measured: **22 passed in ~1 s** (pytest 9.1.1).

## 2. Bounded smoke training

```bash
micromamba run -n test-py312 python -m rlmalloc.train \
  --dist lognormal_train --episodes 300 --max-steps 200 --learn-every 2 \
  --seed 0 --device cpu \
  --out results/checkpoints/agent_quick \
  --log results/metrics/train_log_quick.csv
```

Measured on CPU: **~2 s** (300 episodes). Episodes are short early on because
invalid actions terminate the episode while `epsilon` is high. This is a smoke
test, **not** a trained policy.

## 3. Full training

```bash
# 10 000 episodes. GPU recommended; on CPU this is hours.
# CPU estimate: ~10 ms/step; early episodes are short (invalid-action
# termination), late episodes run up to hundreds of steps.
micromamba run -n test-py312 python -m rlmalloc.train \
  --dist lognormal_train --episodes 10000 --max-steps 2000 \
  --seed 0 --device cuda \
  --out results/checkpoints/agent \
  --log results/metrics/train_log.csv
```

**Measured reference run (RTX 5060 Laptop GPU):**
`episodes=10000 steps=568654 final_eps=0.0100 mean_return_last100=74.345
elapsed=1275.6s` (**~21.3 min**). Checkpoint at `results/checkpoints/agent.pt`.

On pure CPU the same run is roughly an order of magnitude slower; budget hours
and run the bounded smoke profile first.

## 4. Evaluation (4 policies × 4 distributions × 1000 rounds)

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

**Measured reference run:** the full evaluation takes **~22 s** on CPU.

## 5. Outputs

| Artifact | Path |
|---|---|
| Training curve CSV | `results/metrics/train_log.csv` |
| Per-round metrics | `results/metrics/eval_per_round.csv` |
| Summary CSV / JSON | `results/metrics/eval_summary.csv`, `eval_summary.json` |
| Consolidated index | `results/metrics.json` |
| Markdown table | `results/tables/summary.md` |
| Figures | `results/figures/fig3_*`, `fig4_*`, `fig5_*` |
| Checkpoints | `results/checkpoints/agent.pt`, `agent_target.pt`, `agent_meta.json` |
| Legacy output copy | `results/original/result.txt` |

## 6. Shell wrappers

```bash
bash scripts/train_quick.sh     # bounded smoke run (~2 s)
bash scripts/train_full.sh      # full run (long)
bash scripts/evaluate.sh        # needs a trained checkpoint
```
