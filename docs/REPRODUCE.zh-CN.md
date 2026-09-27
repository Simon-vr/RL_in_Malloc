# 复现结果

所有 Python 命令**必须**在隔离环境 `test-py312` 中运行，且应从仓库根目录执行。
这样项目不会污染机器上的其它 Python 环境。

```bash
micromamba run -n test-py312 python ...
```

## 0. 依赖

运行时依赖已安装在 `test-py312`。如需重建环境：

```bash
micromamba run -n test-py312 pip install -r requirements.txt
micromamba run -n test-py312 pip install -r requirements-dev.txt
# 或从零创建：
micromamba env create -f environment.yml      # 创建名为 test-py312 的环境
```

## 1. 测试（很快，实测 < 2 秒）

```bash
cd /home/yangsch/RLmalloc
micromamba run -n test-py312 python -m pytest -q tests
```

实测：**22 passed in ~1 s**（pytest 9.1.1）。

## 2. 有界冒烟训练

```bash
micromamba run -n test-py312 python -m rlmalloc.train \
  --dist lognormal_train --episodes 300 --max-steps 200 --learn-every 2 \
  --seed 0 --device cpu \
  --out results/checkpoints/agent_quick \
  --log results/metrics/train_log_quick.csv
```

CPU 实测：**约 2 秒**（300 回合）。早期回合很短，因为 `epsilon` 高时非法动作会
终止回合。这只是冒烟测试，**不是**训练好的策略。

## 3. 完整训练

```bash
# 10000 回合。建议 GPU；纯 CPU 需要数小时。
# CPU 估算：约 10 ms/步；早期回合短（非法动作终止），后期回合可达数百步。
micromamba run -n test-py312 python -m rlmalloc.train \
  --dist lognormal_train --episodes 10000 --max-steps 2000 \
  --seed 0 --device cuda \
  --out results/checkpoints/agent \
  --log results/metrics/train_log.csv
```

**参考运行实测（RTX 5060 Laptop GPU）：**
`episodes=10000 steps=568654 final_eps=0.0100 mean_return_last100=74.345
elapsed=1275.6s`（**约 21.3 分钟**）。检查点位于
`results/checkpoints/agent.pt`。

纯 CPU 下同一运行大约慢一个数量级；请预留数小时，并先跑有界冒烟配置。

## 4. 评估（4 策略 × 4 分布 × 1000 回合）

```bash
micromamba run -n test-py312 python -m rlmalloc.evaluate \
  --checkpoint results/checkpoints/agent --seed 0 --device cpu \
  --rounds 1000 --requests 200 --release-rate 0.3 \
  --dists lognormal_train lognormal_large uniform bimodal \
  --out-metrics results/metrics --out-tables results/tables \
  --figures results/figures
```

有界预览（同一代码路径）：

```bash
micromamba run -n test-py312 python -m rlmalloc.evaluate \
  --checkpoint results/checkpoints/agent --seed 0 --device cpu \
  --rounds 50 --requests 200 --release-rate 0.3 \
  --dists lognormal_train lognormal_large uniform bimodal \
  --out-metrics results/metrics_preview --out-tables results/tables_preview \
  --figures results/figures_preview
```

**参考运行实测：**完整评估在 CPU 上约 **22 秒**。

## 5. 产物

| 产物 | 路径 |
|---|---|
| 训练曲线 CSV | `results/metrics/train_log.csv` |
| 逐轮指标 | `results/metrics/eval_per_round.csv` |
| 汇总 CSV / JSON | `results/metrics/eval_summary.csv`, `eval_summary.json` |
| 合并索引 | `results/metrics.json` |
| Markdown 表格 | `results/tables/summary.md` |
| 图 | `results/figures/fig3_*`, `fig4_*`, `fig5_*` |
| 检查点 | `results/checkpoints/agent.pt`, `agent_target.pt`, `agent_meta.json` |
| 归档输出副本 | `results/original/result.txt` |

## 6. Shell 包装脚本

```bash
bash scripts/train_quick.sh     # 有界冒烟运行（约 2 秒）
bash scripts/train_full.sh      # 完整运行（耗时长）
bash scripts/evaluate.sh        # 需要已训练的检查点
```
