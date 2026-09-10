# RLmalloc

A faithful, runnable re-implementation of **“A DQN-Based Hybrid Decision
Framework for Dynamic Memory Allocation”** (Sicheng Yang, Jing Ma).

> English first, 中文见下方.

The framework pre-filters a small set of candidate free blocks with a
First-Fit-style rule and lets a DQN agent choose among them. The reward is the
Herfindahl–Hirschman Index (HHI) of the free space, rewarding a concentrated
(healthy) memory layout. The agent is compared against First-Fit, Best-Fit and
Worst-Fit on four request distributions.

> **Honesty note.** The paper does not specify the learning rate, episode
> count, or epsilon schedule; the bimodal σ is also unstated. Exact
> reproduction of the paper’s numbers is therefore not claimed. See
> [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md). All numbers in `results/` come
> from real runs in this repository.

---

## 1. Paper

* **Title:** A DQN-Based Hybrid Decision Framework for Dynamic Memory Allocation
* **Authors:** Sicheng Yang (Sun Yat-sen University), Jing Ma (University of Jinan)
* **PDF:** [`docs/assets/paper.pdf`](docs/assets/paper.pdf) (root copy preserved)

One-paragraph summary: dynamic allocation is cast as an MDP. Rather than
outputting an address, the DQN picks an index into a 5-block candidate set
built by sorting free blocks by start address and taking the first that fit.
The reward is `Σ(lᵢ/S_total)²`, and invalid actions terminate the episode with
reward `-1`. The network is a small `11→128→128→5` MLP trained with experience
replay and a target network.

## 2. Method at a glance

| Element | Design | Code |
|---|---|---|
| State | `2k+1 = 11` (size/addr pairs + request) | `rlmalloc/env.py` |
| Action | index in `{0..k-1}`, `k = 5` | `rlmalloc/env.py::step` |
| Candidate set | sort by start asc, first `k` that fit | `rlmalloc/candidates.py` |
| Reward | HHI `Σ(lᵢ/S_total)² ∈ [0,1]` | `rlmalloc/env.py::_reward` |
| Invalid action | reward `-1`, `done` | `rlmalloc/env.py::step` |
| Network | `11→128→128→5` ReLU | `rlmalloc/agent.py` |
| DQN hyperparams | `B=64, γ=0.99, C=10000, T=50` | `rlmalloc/config.py` |

Full spec→code mapping: [`docs/METHOD.md`](docs/METHOD.md).

## 3. Repository layout

```
RLmalloc/
├── README.md                  # this file (EN + 中文)
├── LICENSE  CITATION.cff
├── requirements.txt  requirements-dev.txt  environment.yml
├── conftest.py                # makes repo importable under pytest
├── config.py agent.py env.py utils.py main.py test.py   # legacy shims
├── docs/
│   ├── METHOD.md  REPRODUCE.md  DEVIATIONS.md
│   └── assets/{paper.pdf, architecture.txt}
├── rlmalloc/                  # the real package
│   ├── config.py candidates.py env.py workloads.py
│   ├── policies.py metrics.py agent.py
│   ├── train.py evaluate.py plotting.py utils.py
├── scripts/{train_quick.sh, train_full.sh, evaluate.sh}
├── tests/{test_candidates,test_env,test_metrics,test_workloads}.py
└── results/
    ├── checkpoints/  metrics/  tables/  figures/  original/
```

## 4. Install

```bash
# dependencies are pinned in requirements*.txt / environment.yml
micromamba run -n test-py312 pip install -r requirements.txt
micromamba run -n test-py312 pip install -r requirements-dev.txt
```

Or create the env: `micromamba env create -f environment.yml`.

## 5. Quickstart

```bash
cd /home/yangsch/RLmalloc

# tests
micromamba run -n test-py312 python -m pytest -q tests

# bounded smoke training
micromamba run -n test-py312 python -m rlmalloc.train \
  --dist lognormal_train --episodes 300 --max-steps 200 --learn-every 2 \
  --seed 0 --device cpu \
  --out results/checkpoints/agent_quick \
  --log results/metrics/train_log_quick.csv

# bounded evaluation (needs a checkpoint)
micromamba run -n test-py312 python -m rlmalloc.evaluate \
  --checkpoint results/checkpoints/agent_quick --seed 0 --device cpu \
  --rounds 20 --requests 200 --release-rate 0.3 \
  --dists lognormal_train lognormal_large uniform bimodal \
  --out-metrics results/metrics --out-tables results/tables \
  --figures results/figures
```

## 6. Reproduce (paper scale)

```bash
# LONG: 10000 episodes (GPU strongly recommended)
micromamba run -n test-py312 python -m rlmalloc.train \
  --dist lognormal_train --episodes 10000 --max-steps 2000 \
  --seed 0 --device cuda \
  --out results/checkpoints/agent --log results/metrics/train_log.csv

micromamba run -n test-py312 python -m rlmalloc.evaluate \
  --checkpoint results/checkpoints/agent --seed 0 --device cpu \
  --rounds 1000 --requests 200 --release-rate 0.3 \
  --dists lognormal_train lognormal_large uniform bimodal \
  --out-metrics results/metrics --out-tables results/tables \
  --figures results/figures
```

Detailed timings and output paths: [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## 7. Results

Produced by the full run in this repository (10 000 training episodes,
569 879 env steps, 21.3 min on an RTX 5060; 1 000 evaluation rounds per
distribution). The raw metric table is reproduced from
[`results/tables/summary.md`](results/tables/summary.md); nothing is hand
written.

| distribution | policy | occupancy | duration | fragmentation | hhi | hhi_complement |
|---|---|---|---|---|---|---|
| lognormal_train | DQN Agent | 0.9500 | 117.48 | 0.6849 | 0.1791 | 0.8209 |
| lognormal_train | First-Fit | 0.9535 | 117.86 | 0.6701 | 0.1921 | 0.8079 |
| lognormal_train | Best-Fit | 0.9588 | 118.59 | 0.6239 | 0.2311 | 0.7689 |
| lognormal_train | Worst-Fit | 0.7838 | 99.22 | 0.8723 | 0.0851 | 0.9149 |
| lognormal_large | DQN Agent | 0.9106 | 34.00 | 0.5562 | 0.3219 | 0.6781 |
| lognormal_large | First-Fit | 0.9111 | 33.99 | 0.5504 | 0.3262 | 0.6738 |
| lognormal_large | Best-Fit | 0.9152 | 34.14 | 0.5277 | 0.3449 | 0.6551 |
| lognormal_large | Worst-Fit | 0.8115 | 30.84 | 0.7015 | 0.2269 | 0.7731 |
| uniform | DQN Agent | 0.8860 | 20.54 | 0.4539 | 0.4362 | 0.5638 |
| uniform | First-Fit | 0.8879 | 20.59 | 0.4483 | 0.4419 | 0.5581 |
| uniform | Best-Fit | 0.8929 | 20.71 | 0.4366 | 0.4508 | 0.5492 |
| uniform | Worst-Fit | 0.8239 | 19.22 | 0.5529 | 0.3654 | 0.6346 |
| bimodal | DQN Agent | 0.9024 | 54.22 | 0.4574 | 0.4061 | 0.5939 |
| bimodal | First-Fit | 0.9064 | 54.45 | 0.4237 | 0.4425 | 0.5575 |
| bimodal | Best-Fit | 0.9176 | 55.03 | 0.3768 | 0.4921 | 0.5079 |
| bimodal | Worst-Fit | 0.7701 | 47.55 | 0.6734 | 0.2389 | 0.7611 |

**How to read this (and how it relates to the paper).**

* Occupancy/duration: DQN is close to but slightly below Best-Fit and
  First-Fit, and clearly above Worst-Fit on every distribution. This is
  consistent with the paper's own finding that "Best-Fit remains the gold
  standard for occupancy and fragmentation".
* Fragmentation: Best-Fit is lowest, as expected. In our run First-Fit edges
  out the DQN on the training distribution (0.6701 vs 0.6849), whereas the
  paper reports a small DQN advantage over First-Fit (0.6521 vs 0.6828). The
  gap is small and the training/epsilon schedules are unspecified by the
  paper; we report the measured values rather than tuning to match.
* Concentration: the paper is internally inconsistent about HHI. Its Eq.(1)
defines the raw sum `Σ(lᵢ/S)²`, but the experimental ordering it reports
(DQN 0.8165 > First-Fit 0.7942 > Best-Fit 0.7548) only matches the legacy
`1 − HHI` convention. Our run reproduces **both**: the raw-sum `hhi` puts the
DQN lowest (0.1791 < 0.1921 < 0.2311), while `hhi_complement` puts the DQN
highest (0.8209 > 0.8079 > 0.7689), matching the paper's claim. Compare
against the paper using the `hhi_complement` column; see
[`docs/DEVIATIONS.md`](docs/DEVIATIONS.md) §1.

Metric definitions:

* **Occupancy** `1 − S_total/M` (higher better)
* **Duration** successful allocations (higher better)
* **Fragmentation** `1 − max(lᵢ)/S_total` (lower better)
* **HHI** `Σ(lᵢ/S_total)²` (higher = more concentrated)

Figures (all generated from real runs):

* Training distribution — `results/figures/fig3_train_hist.png`
* Test distributions — `results/figures/fig4_*.png`
* Policy comparison — `results/figures/fig5_comparison_*.png` and
  `results/figures/fig5_{occupancy,duration,fragmentation,hhi}_all.png`

## 8. Known deviations

See [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md) for the full list: HHI
convention, unspecified learning rate / E / epsilon, bimodal σ assumption,
candidate-set restriction, and terminal invalid actions.

## 9. Citation

See [`CITATION.cff`](CITATION.cff).

## 10. License

MIT — see [`LICENSE`](LICENSE).

---

# 中文说明

本项目是对论文 **《A DQN-Based Hybrid Decision Framework for Dynamic Memory
Allocation》**（杨思成、马静）的可运行复现。英文部分在上方，中文摘要如下。

## 方法概览

- **状态**：`2k+1 = 11` 维（前 k 个候选空闲块的 `大小/M`、`起始地址/M`，最后是 `请求大小/M`）。
- **候选集**：按起始地址升序排序，取前 k 个满足请求的空闲块（k = 5）。状态构造与动作解码共用同一函数，避免“状态/动作顺序不一致”的历史缺陷。
- **动作**：`{0..k-1}` 的离散动作，表示选择第 i 个候选块。
- **奖励**：HHI `Σ(lᵢ/S_total)²`，范围 [0,1]，鼓励空闲空间集中。
- **非法动作**：奖励 `-1.0` 并立即结束该回合。
- **网络**：`11→128→128→5` 全连接 ReLU 网络，经验回放 + 目标网络。

## 快速开始

```bash
cd /home/yangsch/RLmalloc

# 单元测试
micromamba run -n test-py312 python -m pytest -q tests

# 快速训练（数百回合，CPU）
micromamba run -n test-py312 python -m rlmalloc.train \
  --dist lognormal_train --episodes 300 --max-steps 200 --learn-every 2 \
  --seed 0 --device cpu \
  --out results/checkpoints/agent_quick \
  --log results/metrics/train_log_quick.csv

# 评估（4 策略 × 4 分布）
micromamba run -n test-py312 python -m rlmalloc.evaluate \
  --checkpoint results/checkpoints/agent \
  --seed 0 --device cpu --rounds 1000 --requests 200 --release-rate 0.3 \
  --dists lognormal_train lognormal_large uniform bimodal \
  --out-metrics results/metrics --out-tables results/tables \
  --figures results/figures
```

## 说明

- 论文未给出学习率、训练回合数 E、ε 衰减以及双峰分布的 σ；本仓库显式给出这些取值，详见 `docs/DEVIATIONS.md`，因此**不宣称逐位复现论文数值**。
- 所有指标与图表均由本仓库真实运行生成，不手工编造。
- 根目录下的论文 PDF 与原始 `result.txt` 始终保留；其副本分别位于 `docs/assets/paper.pdf` 与 `results/original/result.txt`。
