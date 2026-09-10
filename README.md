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

Every command in this README runs inside the isolated conda/mamba
environment **`test-py312`** (Python 3.12). The reference machine already has
it; the exact package versions are pinned in `requirements.txt`,
`requirements-dev.txt`, and `environment.yml`.

```bash
# dependencies already present in test-py312; re-install / re-pin with:
micromamba run -n test-py312 pip install -r requirements.txt
micromamba run -n test-py312 pip install -r requirements-dev.txt
```

To recreate an equivalent environment from scratch on another machine:

```bash
micromamba env create -f environment.yml   # creates env `test-py312`
```

> `environment.yml` pins `torch==2.14.0`; on the reference machine this
> resolves to a CUDA build (`2.14.0+cu130`). Swap in a CPU wheel if you do not
> need GPU training.

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
# LONG: 10000 episodes. Use a GPU if available; otherwise replace
# `--device cuda` with `--device cpu` (hours instead of ~20 min).
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
A full catalog of every generated artifact is in
[`results/README.md`](results/README.md).

## 7. Results

Produced by the full run in this repository (10 000 training episodes,
568 654 env steps, 21.3 min on an RTX 5060; 1 000 evaluation rounds per
distribution). The raw metric table is reproduced from
[`results/tables/summary.md`](results/tables/summary.md); nothing is hand
written.

| distribution | policy | occupancy | duration | fragmentation | hhi | hhi_complement |
|---|---|---|---|---|---|---|
| lognormal_train | DQN Agent | 0.9489 | 117.28 | 0.6882 | 0.1798 | 0.8202 |
| lognormal_train | First-Fit | 0.9535 | 117.86 | 0.6701 | 0.1921 | 0.8079 |
| lognormal_train | Best-Fit | 0.9588 | 118.59 | 0.6239 | 0.2311 | 0.7689 |
| lognormal_train | Worst-Fit | 0.7838 | 99.22 | 0.8723 | 0.0851 | 0.9149 |
| lognormal_large | DQN Agent | 0.9098 | 33.97 | 0.5538 | 0.3230 | 0.6770 |
| lognormal_large | First-Fit | 0.9111 | 33.99 | 0.5504 | 0.3262 | 0.6738 |
| lognormal_large | Best-Fit | 0.9152 | 34.14 | 0.5277 | 0.3449 | 0.6551 |
| lognormal_large | Worst-Fit | 0.8115 | 30.84 | 0.7015 | 0.2269 | 0.7731 |
| uniform | DQN Agent | 0.8857 | 20.52 | 0.4581 | 0.4321 | 0.5679 |
| uniform | First-Fit | 0.8879 | 20.59 | 0.4483 | 0.4419 | 0.5581 |
| uniform | Best-Fit | 0.8929 | 20.71 | 0.4366 | 0.4508 | 0.5492 |
| uniform | Worst-Fit | 0.8239 | 19.22 | 0.5529 | 0.3654 | 0.6346 |
| bimodal | DQN Agent | 0.9026 | 54.19 | 0.4558 | 0.4080 | 0.5920 |
| bimodal | First-Fit | 0.9064 | 54.45 | 0.4237 | 0.4425 | 0.5575 |
| bimodal | Best-Fit | 0.9176 | 55.03 | 0.3768 | 0.4921 | 0.5079 |
| bimodal | Worst-Fit | 0.7701 | 47.55 | 0.6734 | 0.2389 | 0.7611 |

**How to read this (and how it relates to the paper).**

* Occupancy/duration: DQN is close to but slightly below Best-Fit and
  First-Fit, and clearly above Worst-Fit on every distribution. This is
  consistent with the paper's own finding that "Best-Fit remains the gold
  standard for occupancy and fragmentation".
* **Transfer distributions:** unlike the paper, our DQN does *not* exceed
  First-Fit on the three transferred distributions. For example, on Bimodal
  the paper reports DQN occupancy 0.9213 > First-Fit 0.9124, while we measure
  DQN 0.9026 < First-Fit 0.9064. The difference is small (0.4 pp) and the
  learning-rate / episode / epsilon schedule is unspecified by the paper, so
  we report the measured value rather than tuning to match. Do not read this
  table as an exact reproduction of the paper.
* Fragmentation: Best-Fit is lowest, as expected. In our run First-Fit edges
  out the DQN on the training distribution (0.6701 vs 0.6882), whereas the
  paper reports a small DQN advantage over First-Fit (0.6521 vs 0.6828). The
  gap is small and the training/epsilon schedules are unspecified by the
  paper; we report the measured values rather than tuning to match.
* Concentration: the paper is internally inconsistent about HHI. Its Eq.(1)
  defines the raw sum `Σ(lᵢ/S)²`, but the experimental ordering it reports
  (DQN 0.8165 > First-Fit 0.7942 > Best-Fit 0.7548) only matches the legacy
  `1 − HHI` convention. Our run reproduces **both**: the raw-sum `hhi` puts
  the DQN lowest (0.1798 < 0.1921 < 0.2311), while `hhi_complement` puts the
  DQN highest (0.8202 > 0.8079 > 0.7689), matching the paper's claim. Compare
  against the paper using the `hhi_complement` column; see
  [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md) §1.

Metric definitions:

* **Occupancy** `1 − S_total/M` (higher better)
* **Duration** successful allocations (higher better)
* **Fragmentation** `1 − max(lᵢ)/S_total` (lower better)
* **HHI** `Σ(lᵢ/S_total)²` (higher = more concentrated)
* **hhi_complement** `1 − HHI` (the convention used by the legacy code / printed
  paper numbers)

Figures (all generated from real runs):

* Training distribution — `results/figures/fig3_train_hist.png` and
  `results/figures/fig3_train_curve.png`
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

> 英文部分见上方。以下是中文镜像版本，内容与英文一致。

本项目是对论文 **《A DQN-Based Hybrid Decision Framework for Dynamic Memory
Allocation》**（作者：Sicheng Yang，中山大学；Jing Ma，济南大学）的忠实、可运行复现。
框架先用 First-Fit 风格规则预筛出少量候选空闲块，再由 DQN 智能体从中选择；
奖励函数基于 Herfindahl–Hirschman 指数（HHI），鼓励空闲空间保持集中（健康）。
实验在四种请求分布上对比 DQN 与 First-Fit、Best-Fit、Worst-Fit。

> **诚实声明**：论文未给出学习率、训练回合数与 ε 衰减方案，双峰分布的 σ 也未给出，
> 因此**不宣称逐位复现论文数值**。详见 [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md)。
> `results/` 中的所有数字均来自本仓库的真实运行。

## 论文

* **标题**：A DQN-Based Hybrid Decision Framework for Dynamic Memory Allocation
* **作者**：Sicheng Yang（中山大学）、Jing Ma（济南大学）
* **PDF**：[`docs/assets/paper.pdf`](docs/assets/paper.pdf)（根目录原始文件保留）
* **摘要**：将动态内存分配建模为 MDP。DQN 不直接输出地址，而是在按起始地址
  排序、满足请求的前 5 个候选块中选择一个。奖励为 `Σ(lᵢ/S_total)²`，非法动作
  以奖励 `-1` 结束回合。网络为 `11→128→128→5` 的小型 MLP，使用经验回放与目标网络。

## 方法概览

| 要素 | 设计 | 代码 |
|---|---|---|
| 状态 | `2k+1 = 11`（前 k 个候选块的大小/地址 + 请求大小） | `rlmalloc/env.py` |
| 动作 | `{0..k-1}` 的离散索引，k = 5 | `rlmalloc/env.py::step` |
| 候选集 | 按起始地址升序，取前 k 个满足请求的块 | `rlmalloc/candidates.py` |
| 奖励 | HHI `Σ(lᵢ/S_total)² ∈ [0,1]` | `rlmalloc/env.py::_reward` |
| 非法动作 | 奖励 `-1` 并立即结束回合 | `rlmalloc/env.py::step` |
| 网络 | `11→128→128→5` ReLU | `rlmalloc/agent.py` |
| DQN 超参 | `B=64, γ=0.99, C=10000, T=50` | `rlmalloc/config.py` |

其中候选集由**同一个函数**同时服务于状态构造与动作解码，从根本上避免了历史上
“状态/动作顺序不一致”的缺陷。完整的“论文规格 → 代码”映射见
[`docs/METHOD.md`](docs/METHOD.md)。

## 仓库结构

```
RLmalloc/
├── README.md                  # 本文件（英文 + 中文）
├── LICENSE  CITATION.cff
├── requirements.txt  requirements-dev.txt  environment.yml
├── docs/                      # METHOD.md / REPRODUCE.md / DEVIATIONS.md + 资源
├── rlmalloc/                  # 真实代码包
├── scripts/                   # 训练/评估 shell 包装
├── tests/                     # pytest 单元测试（21 项）
└── results/                   # 指标、表格、图、检查点、原始结果副本
```

> 根目录下的论文 PDF 与原始 `result.txt` 始终保留（不改名、不删除）；
> 其副本分别位于 `docs/assets/paper.pdf` 与 `results/original/result.txt`。

## 安装

所有命令都在隔离环境 **`test-py312`**（Python 3.12）中运行。参考机器上该环境
已就绪；确切版本固定于 `requirements.txt`、`requirements-dev.txt` 与
`environment.yml`。

```bash
# test-py312 中已安装依赖；如需重装/重新固定：
micromamba run -n test-py312 pip install -r requirements.txt
micromamba run -n test-py312 pip install -r requirements-dev.txt
```

在另一台机器上从零重建等价环境：

```bash
micromamba env create -f environment.yml   # 创建名为 test-py312 的环境
```

## 快速开始

```bash
cd /home/yangsch/RLmalloc

# 单元测试（21 项，约 1 秒）
micromamba run -n test-py312 python -m pytest -q tests

# 有界冒烟训练（300 回合，CPU，约 2 秒；仅用于验证流程）
micromamba run -n test-py312 python -m rlmalloc.train \
  --dist lognormal_train --episodes 300 --max-steps 200 --learn-every 2 \
  --seed 0 --device cpu \
  --out results/checkpoints/agent_quick \
  --log results/metrics/train_log_quick.csv

# 有界评估（20 回合，需上面的检查点）
micromamba run -n test-py312 python -m rlmalloc.evaluate \
  --checkpoint results/checkpoints/agent_quick --seed 0 --device cpu \
  --rounds 20 --requests 200 --release-rate 0.3 \
  --dists lognormal_train lognormal_large uniform bimodal \
  --out-metrics results/metrics --out-tables results/tables \
  --figures results/figures
```

## 复现（论文规模）

```bash
# 训练 10000 回合。有 GPU 则用 cuda；否则把 --device 换成 cpu（数小时而非约 20 分钟）。
micromamba run -n test-py312 python -m rlmalloc.train \
  --dist lognormal_train --episodes 10000 --max-steps 2000 \
  --seed 0 --device cuda \
  --out results/checkpoints/agent --log results/metrics/train_log.csv

# 评估：4 策略 × 4 分布 × 1000 回合
micromamba run -n test-py312 python -m rlmalloc.evaluate \
  --checkpoint results/checkpoints/agent --seed 0 --device cpu \
  --rounds 1000 --requests 200 --release-rate 0.3 \
  --dists lognormal_train lognormal_large uniform bimodal \
  --out-metrics results/metrics --out-tables results/tables \
  --figures results/figures
```

完整耗时说明与输出路径见 [`docs/REPRODUCE.md`](docs/REPRODUCE.md)；所有产物的
清单见 [`results/README.md`](results/README.md)。

## 结果摘要

以下数值来自本仓库的真实完整运行（10000 回合训练，568654 步，RTX 5060 上约
21.3 分钟；每个分布 1000 回合评估）。原始表格见
[`results/tables/summary.md`](results/tables/summary.md)，未手工编造。

| 分布 | 策略 | occupancy | duration | fragmentation | hhi | hhi_complement |
|---|---|---|---|---|---|---|
| lognormal_train | DQN Agent | 0.9489 | 117.28 | 0.6882 | 0.1798 | 0.8202 |
| lognormal_train | First-Fit | 0.9535 | 117.86 | 0.6701 | 0.1921 | 0.8079 |
| lognormal_train | Best-Fit | 0.9588 | 118.59 | 0.6239 | 0.2311 | 0.7689 |
| lognormal_train | Worst-Fit | 0.7838 | 99.22 | 0.8723 | 0.0851 | 0.9149 |
| lognormal_large | DQN Agent | 0.9098 | 33.97 | 0.5538 | 0.3230 | 0.6770 |
| lognormal_large | First-Fit | 0.9111 | 33.99 | 0.5504 | 0.3262 | 0.6738 |
| lognormal_large | Best-Fit | 0.9152 | 34.14 | 0.5277 | 0.3449 | 0.6551 |
| lognormal_large | Worst-Fit | 0.8115 | 30.84 | 0.7015 | 0.2269 | 0.7731 |
| uniform | DQN Agent | 0.8857 | 20.52 | 0.4581 | 0.4321 | 0.5679 |
| uniform | First-Fit | 0.8879 | 20.59 | 0.4483 | 0.4419 | 0.5581 |
| uniform | Best-Fit | 0.8929 | 20.71 | 0.4366 | 0.4508 | 0.5492 |
| uniform | Worst-Fit | 0.8239 | 19.22 | 0.5529 | 0.3654 | 0.6346 |
| bimodal | DQN Agent | 0.9026 | 54.19 | 0.4558 | 0.4080 | 0.5920 |
| bimodal | First-Fit | 0.9064 | 54.45 | 0.4237 | 0.4425 | 0.5575 |
| bimodal | Best-Fit | 0.9176 | 55.03 | 0.3768 | 0.4921 | 0.5079 |
| bimodal | Worst-Fit | 0.7701 | 47.55 | 0.6734 | 0.2389 | 0.7611 |

**如何解读（以及与论文的关系）**

* **占用率/服务时长**：DQN 略低于 Best-Fit 与 First-Fit，但明显高于 Worst-Fit，
  这与论文“Best-Fit 仍是占用率与碎片率的黄金标准”的结论一致。
* **迁移分布**：与论文不同，本实现中 DQN 在三个迁移分布上并未超过 First-Fit。
  例如双峰分布上论文报告 DQN 占用率 0.9213 > First-Fit 0.9124，而本仓库实测
  DQN 0.9026 < First-Fit 0.9064。差距很小（0.4 个百分点），且论文未给出学习率/
  回合数/ε 方案，因此我们如实报告实测值，而不调参去贴合论文。
* **碎片率**：Best-Fit 最低，符合预期。本实现中 First-Fit 在训练分布上略优于
  DQN（0.6701 vs 0.6882）。
* **集中度（HHI）**：论文内部存在不一致——其 Eq.(1) 定义的是原始和
  `Σ(lᵢ/S)²`，但其报告的数值（DQN 0.8165 > First-Fit 0.7942 > Best-Fit
  0.7548）只有在 `1 − HHI` 约定下才成立。本仓库同时给出两列：原始 `hhi` 下
  DQN 最低（0.1798 < 0.1921 < 0.2311），而 `hhi_complement` 下 DQN 最高
  （0.8202 > 0.8079 > 0.7689），与论文结论一致。**与论文表格对比时请使用
  `hhi_complement` 列。**

指标定义：

* **Occupancy** `1 − S_total/M`（越高越好）
* **Duration** 成功分配次数（越高越好）
* **Fragmentation** `1 − max(lᵢ)/S_total`（越低越好）
* **HHI** `Σ(lᵢ/S_total)²`（越高表示空闲空间越集中）
* **hhi_complement** `1 − HHI`（旧代码/论文印刷值使用的约定）

图表（均由真实运行生成）：

* 训练分布 — `results/figures/fig3_train_hist.png`、`fig3_train_curve.png`
* 测试分布 — `results/figures/fig4_*.png`
* 策略对比 — `results/figures/fig5_comparison_*.png` 与
  `results/figures/fig5_{occupancy,duration,fragmentation,hhi}_all.png`

## 已知偏差

完整清单见 [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md)：HHI 约定、论文未给出的
学习率/E/ε、双峰 σ 假设、DQN 候选集限制，以及非法动作终止等。

## 引用与许可

* 引用信息见 [`CITATION.cff`](CITATION.cff)。
* 许可证：MIT，见 [`LICENSE`](LICENSE)。
