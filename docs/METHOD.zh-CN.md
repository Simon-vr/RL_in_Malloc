# 方法

本文描述环境、MDP、网络、训练、负载与评估协议。文中的文件/行号指向
`rlmalloc/` 包。

## 1. 问题设定

一个 `M = 4096` 字节的模拟竞技场（`rlmalloc/config.py: MEMORY_SIZE`）按如下
方式维护：

* **空闲块链表** `F = {(aᵢ, lᵢ)}`，按起始地址排序；`aᵢ` 为起始地址，`lᵢ` 为大小；
* **已分配字典**：分配序号 → `(start, size)`。

负载是一串请求：

* **分配** `s` 字节的连续块：选一个 `size >= s` 的空闲块；若比 `s` 大，剩余部分
  作为新的（更小）空闲块放回；若没有足够大的块，分配失败；
* **释放**：释放某个先前分配的块。

空闲链表每次变化时都会合并相邻空闲块（`MemoryEnv._merge_free_blocks`）。

评估指标在轮末空闲链表上计算（`rlmalloc/metrics.py`）：

| 指标 | 公式 | 方向 |
|---|---|---|
| Occupancy | `1 − S_total / M` | 越高越好 |
| Duration | 成功分配次数 | 越高越好 |
| Fragmentation | `1 − max(lᵢ) / S_total` | 越低越好 |
| HHI | `Σ(lᵢ / S_total)²` | 越高表示越集中 |

其中 `S_total = Σᵢ lᵢ` 为总空闲空间。

## 2. MDP 形式化

### 状态（`2k + 1 = 11`）—— `MemoryEnv._get_state`

针对当前请求，用 `rlmalloc/candidates.py::build_candidates` 构造候选集（空闲块
按起始地址升序，取前 `k = 5` 个满足 `size >= request` 的块）。每个候选 `i` 贡献
两个归一化值 `[size_i / M, start_i / M]`，最后一项为 `request / M`。未使用的
候选槽用 0 补齐。

由于状态编码与动作解码**共用同一个** `build_candidates` 函数，智能体在槽 `i`
看到的块，恰好就是动作 `i` 所分配的块。（早期版本两者排序不同，曾静默地使学到的
策略失效；这正是把候选构造收敛为单一函数、并用回归测试保护它的原因。）

### 动作（`0 .. k−1`）—— `MemoryEnv.step`

动作 `i` 选择候选 `i`。`rlmalloc/policies.py::dqn_action` 只是返回智能体贪心索引
的薄适配层，**不会**重新排序。

### 奖励 —— `MemoryEnv._reward`

`R = Σ(lᵢ / S_total)²`，对空闲块集合求和；`S_total = 0` 时 `R = 0`。这就是空闲
份额的 HHI，取值在 `[0, 1]`：空闲空间是一个连续块时 `R = 1`，`N` 个等分碎片时
`R = 1/N`。它为何只是一个**代理**目标，见 [`DESIGN.zh-CN.md`](DESIGN.zh-CN.md)。

### 非法动作 —— `MemoryEnv.step`

若动作索引超出候选集范围，环境返回奖励 `-1.0`，且 `done = True`，回合立即结束。

### 终止 —— `MemoryEnv.is_done`

事件列表耗尽（评估）或没有空闲块满足当前请求（训练与评估）时为真。

## 3. 网络 —— `rlmalloc/agent.py::QNetwork`

```
11 → Linear(128) → ReLU → Linear(128) → ReLU → Linear(5)
```

按标准 value-based DQN 训练：经验回放（`REPLAY_BUFFER_SIZE = 10000`）、每
`TARGET_UPDATE_FREQ = 50` 个环境步同步一次的目标网络、SmoothL1 损失、Adam
（`LEARNING_RATE = 1e-4`）、ε-贪心（`1.0 → 0.01`，每回合乘 `0.999`）。

所有超参数在 `rlmalloc/config.py`。

## 4. 训练 —— `rlmalloc/train.py`

* ε-贪心选动作；ε 每回合衰减一次；
* 每 `LEARN_EVERY` 个环境步做一次梯度更新（默认 1）；
* 每 `TARGET_UPDATE_FREQ = 50` 个环境步复制一次目标网络；
* 每回合写 CSV：`episode, return, steps, duration, epsilon, mean_loss,
  occupancy_end`；
* `MAX_STEPS_PER_EPISODE` 安全上限，防止回合失控。

运行：

```bash
micromamba run -n test-py312 python -m rlmalloc.train \
  --dist lognormal_train --episodes 10000 --max-steps 2000 \
  --seed 0 --device cuda --out results/checkpoints/agent \
  --log results/metrics/train_log.csv
```

## 5. 负载 —— `rlmalloc/workloads.py`

`DISTRIBUTIONS`（在 `config.py`）定义四种请求大小分布，大小都裁剪到 `[1, 512]`。

| 名称 | 规格 |
|---|---|
| `lognormal_train` | `clip(Lognormal(ln 32, σ=0.9), 1, 512)` —— 默认训练负载 |
| `lognormal_large` | `clip(Lognormal(ln 128, σ=0.7), 1, 512)` |
| `uniform` | `Uniform{1..512}` |
| `bimodal` | 70% `Lognormal(ln 16, σ=0.9)` + 30% `Lognormal(ln 256, σ=0.9)` |

`generate_workload` 生成显式事件列表，基于**乐观参考时间线**：假定每次分配都
成功，因此每个 `free` 事件引用的分配序号在参考路径上必然存活。真实策略若提前
失败，会在到达后续 free 事件前停下，因此永远不会遇到不存在的序号。同一轮中所有
策略回放完全相同的事件列表。

## 6. 基线 —— `rlmalloc/policies.py`

`first_fit`、`best_fit`、`worst_fit` 会遍历**所有**空闲块：

* **First-Fit** —— 地址最低的可用块；
* **Best-Fit** —— 剩余最小的可用块；
* **Worst-Fit** —— 最大的可用块。

因此它们比 DQN 的"前 `k` 个候选"拥有更大的有效动作集；这一不对称是当前建模的
固有特性，见 [`DESIGN.zh-CN.md`](DESIGN.zh-CN.md)。

## 7. 评估协议 —— `rlmalloc/evaluate.py`

对每个分布、每一轮 `r`：`rng = make_rng(seed, r)`，生成一个负载，然后在**同一个**
负载上运行所有策略。输出：

* 逐轮长表 CSV（`results/metrics/eval_per_round.csv`）；
* 汇总 CSV/JSON（`eval_summary.csv`、`eval_summary.json`）；
* Markdown 表格（`results/tables/summary.md`）；
* `results/figures/` 下的图。

## 8. 代码地图

| 关注点 | 模块 |
|---|---|
| 候选规则 | `rlmalloc/candidates.py` |
| 环境 / MDP | `rlmalloc/env.py` |
| 指标 | `rlmalloc/metrics.py` |
| 网络 / 智能体 | `rlmalloc/agent.py` |
| 启发式 | `rlmalloc/policies.py` |
| 负载 | `rlmalloc/workloads.py` |
| 训练 | `rlmalloc/train.py` |
| 评估 | `rlmalloc/evaluate.py` |
| 绘图 | `rlmalloc/plotting.py` |
| 配置 | `rlmalloc/config.py` |
