# RLmalloc

[English](README.md) | **中文**

一个用**强化学习研究动态内存分配**的项目：DQN 智能体在每个分配请求到来时，
从少量候选空闲块中选择一个来分配，奖励函数鼓励空闲空间保持**集中**（健康）。
项目将它与经典的 First-Fit、Best-Fit、Worst-Fit 策略在四种请求分布上对比。

本仓库中的所有数字与图都来自本仓库代码的真实运行。

## 目录

1. [项目是什么](#1-项目是什么)
2. [结果速览](#2-结果速览)
3. [方法](#3-方法)
4. [仓库结构](#4-仓库结构)
5. [安装](#5-安装)
6. [快速开始](#6-快速开始)
7. [复现完整结果](#7-复现完整结果)
8. [局限与可能原因](#8-局限与可能原因)
9. [指标（以及 HHI 的定义）](#9-指标以及-hhi-的定义)
10. [测试](#10-测试)
11. [文档索引](#11-文档索引)
12. [许可证](#12-许可证)

---

## 1. 项目是什么

一个模拟的内存分配竞技场（4096 字节）持续处理分配请求与随机释放。每一步，
环境暴露一小组**候选空闲块**（满足当前请求、按起始地址排序的前 `k = 5` 个），
DQN 从中选择一个。奖励是空闲空间的 *Herfindahl–Hirschman 指数*（HHI）：空闲
字节集中在少数大块时高（健康），散落成小碎片时低（碎片化）。

项目想回答一个简单问题：**学习得到的非贪心放置策略，能否与手写启发式抗衡？**
本仓库实测的答案是：*部分可以——与 First-Fit 相当，但没有超过 Best-Fit。*

## 2. 结果速览

以下所有图都由 `rlmalloc.evaluate` / `rlmalloc.plotting` 生成，位于
[`results/figures/`](results/figures/)。

### 2.1 训练

训练请求大小分布（中位数 32 B，右偏），以及完整 10000 回合训练的学习曲线：

![训练请求分布](results/figures/fig3_train_hist.png)
![训练学习曲线](results/figures/fig3_train_curve.png)

智能体确实在学习：最后 100 回合的平均回报为 **74.35**（早期约为 −1，因为随机
探索会触发非法动作）。

### 2.2 评估负载

用四种请求分布来检验鲁棒性：

| 训练分布 | 大请求 | 均匀 | 双峰 |
|---|---|---|---|
| ![](results/figures/fig4_lognormal_train.png) | ![](results/figures/fig4_lognormal_large.png) | ![](results/figures/fig4_uniform.png) | ![](results/figures/fig4_bimodal.png) |

### 2.3 各分布上的策略对比

![占用率](results/figures/fig5_occupancy_all.png)
![服务时长](results/figures/fig5_duration_all.png)
![碎片率](results/figures/fig5_fragmentation_all.png)
![HHI](results/figures/fig5_hhi_all.png)

单个分布上的 2×2 对比（占用率 / 时长 / 碎片率 / HHI）：

![单分布对比](results/figures/fig5_comparison_lognormal_train.png)

### 2.4 结果表

每个分布 1000 回合的均值（单一种子；见第 8 节）。原始表格：
[`results/tables/summary.md`](results/tables/summary.md)。

| 分布 | 策略 | occupancy | duration | fragmentation | hhi |
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

**怎么读这张表**：Best-Fit 在四种分布上的占用率最高、服务时长最长、碎片率
最低；Worst-Fit 在这三项上都最差。DQN 位于两者之间，接近（且略低于）
First-Fit。在空闲空间集中度指标（HHI）上排序相同：Best-Fit > First-Fit >
DQN > Worst-Fit。总之，**智能体学到了一个合理的策略，但没有超过经典最优启发式。**

## 3. 方法

| 要素 | 设计 |
|---|---|
| 状态 | `2k+1 = 11` 个浮点：`k=5` 个候选块的 `[size/M, start/M]`，末尾拼上 `request/M` |
| 动作 | 候选集索引，`0..k-1` |
| 候选集 | 空闲块按起始地址升序，取前 `k` 个满足 `size >= request` 的块 |
| 奖励 | 空闲链表的 HHI：`Σ(lᵢ/S_total)² ∈ [0,1]` |
| 非法动作 | 奖励 `-1.0`，且**回合立即结束** |
| 网络 | `11 → 128 → ReLU → 128 → ReLU → 5` 的 MLP |
| 学习 | DQN：经验回放（`C=10000`）、目标网络（每 `T=50` 步同步一次）、SmoothL1、Adam（`lr=1e-4`） |

状态构造与动作解码共用同一个函数
（`rlmalloc/candidates.py::build_candidates`），因此智能体**看到**的候选块与
动作**选中**的候选块一致。完整细节见 [`docs/METHOD.zh-CN.md`](docs/METHOD.zh-CN.md)。

## 4. 仓库结构

```
RLmalloc/
├── README.md  README.zh-CN.md          # 英文 / 中文
├── LICENSE
├── requirements.txt  requirements-dev.txt  environment.yml
├── conftest.py
├── config.py agent.py env.py utils.py main.py test.py   # 兼容旧入口的 shim├── docs/
│   ├── METHOD.md  METHOD.zh-CN.md
│   ├── DESIGN.md  DESIGN.zh-CN.md
│   ├── REPRODUCE.md  REPRODUCE.zh-CN.md
│   └── assets/architecture.txt
├── rlmalloc/                           # 真正的代码包
│   ├── config.py candidates.py env.py workloads.py
│   ├── policies.py metrics.py agent.py
│   └── train.py evaluate.py plotting.py utils.py
├── scripts/{train_quick.sh, train_full.sh, evaluate.sh}
├── tests/{test_candidates,test_env,test_metrics,test_workloads}.py
└── results/
    ├── README.md  metrics.json
    ├── checkpoints/  metrics/  tables/  figures/  original/
```

## 5. 安装

本 README 的所有命令都在**隔离的 conda/mamba 环境 `test-py312`**（Python 3.12）
中运行，从而不污染机器上的其它 Python 环境。

```bash
# test-py312 中已装好依赖；如需重装 / 重新固定版本：
micromamba run -n test-py312 pip install -r requirements.txt
micromamba run -n test-py312 pip install -r requirements-dev.txt
```

在别的机器上从零重建等价环境：

```bash
micromamba env create -f environment.yml   # 创建名为 test-py312 的环境
```

> `environment.yml` 固定了 `torch==2.14.0`；在参考机器上解析为 CUDA 版
> （`2.14.0+cu130`）。若不需要 GPU 训练，可换成 CPU wheel。

## 6. 快速开始

```bash
cd /home/yangsch/RLmalloc

# 单元测试（很快）
micromamba run -n test-py312 python -m pytest -q tests

# 有界冒烟训练（CPU 约 2 秒；注意：这不是训练好的策略）
micromamba run -n test-py312 python -m rlmalloc.train \
  --dist lognormal_train --episodes 300 --max-steps 200 --learn-every 2 \
  --seed 0 --device cpu \
  --out results/checkpoints/agent_quick \
  --log results/metrics/train_log_quick.csv

# 用该检查点做有界评估
micromamba run -n test-py312 python -m rlmalloc.evaluate \
  --checkpoint results/checkpoints/agent_quick --seed 0 --device cpu \
  --rounds 20 --requests 200 --release-rate 0.3 \
  --dists lognormal_train lognormal_large uniform bimodal \
  --out-metrics results/metrics --out-tables results/tables \
  --figures results/figures
```

## 7. 复现完整结果

```bash
# 完整训练：10000 回合。建议用 GPU；纯 CPU 需要数小时。
micromamba run -n test-py312 python -m rlmalloc.train \
  --dist lognormal_train --episodes 10000 --max-steps 2000 \
  --seed 0 --device cuda \
  --out results/checkpoints/agent --log results/metrics/train_log.csv

# 完整评估：4 策略 × 4 分布 × 1000 回合
micromamba run -n test-py312 python -m rlmalloc.evaluate \
  --checkpoint results/checkpoints/agent --seed 0 --device cpu \
  --rounds 1000 --requests 200 --release-rate 0.3 \
  --dists lognormal_train lognormal_large uniform bimodal \
  --out-metrics results/metrics --out-tables results/tables \
  --figures results/figures
```

本机参考耗时（RTX 5060 Laptop GPU + CPU）：完整训练**约 21.3 分钟**
（`episodes=10000 steps=568654`），完整评估**约 22 秒**。确切命令、输出与产物
清单见 [`docs/REPRODUCE.zh-CN.md`](docs/REPRODUCE.zh-CN.md)。

## 8. 局限与可能原因

学习到的策略未超过 Best-Fit。主要观察：

* **Best-Fit 在每个经典指标上都更强。** 在四种分布上，占用率、时长、碎片率均
  偏向 Best-Fit。
* **相对 First-Fit 的优势很小。** 在训练分布上，First-Fit 的碎片率反而略低
  （0.6701 vs 0.6882）。
* **明显胜出的只有 Worst-Fit。**
* **单一种子。** 只跑了一个种子，因此 DQN 与 First-Fit 的小差距（占用率约
  0.003–0.005）没有方差估计。

**可能原因（按重要性从高到低）：**

1. **动作空间受限。** DQN 只能在（按地址排序的）前 `k = 5` 个可用块中选择，而
   启发式会扫描**所有**空闲块。Best-Fit 的全局最优块常常**不在候选集里**，
   于是智能体根本无法表达"让 Best-Fit 变好"的那个选择。这很可能是最主要的限制。
2. **奖励是代理指标，而非目标本身。** 学习最大化的是 HHI（集中度），而评估的是
   占用率 / 时长 / 碎片率。优化一个平滑的集中度代理，并不保证占用率最优。
3. **可观测性有限。** 状态只包含这 `k` 个候选块与当前请求——没有全局空闲链表
   统计（块数、总空闲字节、最大块），也没有历史。智能体无法像启发式那样"纵观
   全局"。
4. **分布偏移。** 训练只用一个负载，另外三个是分布外数据，策略只能部分迁移。
5. **优化未调参。** 学习率、ε 方案、回合数都是手工设定而非搜索得到；单一种子
   无法给出方差估计，因此小差距可能是噪声。
6. **探索代价高。** 非法动作会以奖励 `-1` 立即结束回合，因此早期随机探索代价大，
   可能对学习产生偏置。

这些是当前建模方式的设计局限，而非 bug；具体改进方向见
[`docs/DESIGN.zh-CN.md`](docs/DESIGN.zh-CN.md)。

## 9. 指标（以及 HHI 的定义）

所有指标都在轮末空闲链表上计算。

| 指标 | 公式 | 方向 |
|---|---|---|
| Occupancy | `1 − S_total/M` | 越高越好 |
| Duration | 成功分配次数 | 越高越好 |
| Fragmentation | `1 − max(lᵢ)/S_total` | 越低越好 |
| HHI | `Σ(lᵢ/S_total)²` | 越高表示越集中 |

**HHI。** `HHI = Σ(lᵢ/S_total)²` 是把 Herfindahl–Hirschman 指数用于空闲块大小
份额，全项目**只定义一次**（`rlmalloc/metrics.py`）。它回答"空闲空间有多集中？"：

* **取值范围**：`N` 个空闲块时 `1/N ≤ HHI ≤ 1`。单个空闲块 → `HHI = 1`；
  `N` 个等大块 → `HHI = 1/N`；没有空闲 → `HHI = 0`。
* **方向**：越高越健康（空闲空间集中在少数大块）。它被最大空闲份额从上方限制，
  而最大份额等于 `1 − Fragmentation`，故 `HHI ≤ 1 − Fragmentation`。
* **为什么 Worst-Fit 的 HHI 最低**：Worst-Fit 总是从最大空闲块分配，会逐渐把
  空闲链表"摊平"成许多大小相近的中等块（最大份额约 0.13，而 First/Best-Fit
  约 0.34–0.37）。份额越接近相等，`Σsᵢ²` 越小，因此 HHI 最低。

## 10. 测试

```bash
micromamba run -n test-py312 python -m pytest -q tests
# -> 22 passed
```

覆盖：候选块顺序不变量（状态与动作一致）、非法动作终止、服务时长计数、
workload/free-ordinal 有效性，以及指标定义（含 `HHI ≤ 1 − Fragmentation` 边界）。

## 11. 文档索引

| 文档 | 内容 |
|---|---|
| [`docs/METHOD.zh-CN.md`](docs/METHOD.zh-CN.md) | 环境、MDP、网络、训练、负载 |
| [`docs/DESIGN.zh-CN.md`](docs/DESIGN.zh-CN.md) | 设计选择、假设、局限、未来工作 |
| [`docs/REPRODUCE.zh-CN.md`](docs/REPRODUCE.zh-CN.md) | 确切命令、耗时、产物清单 |
| [`results/README.md`](results/README.md) | 所有生成产物的目录（英文） |
| English | [`README.md`](README.md), `docs/*.md` |

## 12. 许可证

MIT —— 见 [`LICENSE`](LICENSE)。
