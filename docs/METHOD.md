# Method

This document describes the environment, the MDP, the network, training, the
workloads and the evaluation protocol. File/line pointers refer to the
`rlmalloc/` package.

## 1. Problem setup

A simulated arena of `M = 4096` bytes (`rlmalloc/config.py: MEMORY_SIZE`) is
managed as:

* a **free-block list** `F = {(aᵢ, lᵢ)}` kept sorted by start address, where
  `aᵢ` is the start address and `lᵢ` the size;
* an **allocated dictionary** mapping allocation ordinal → `(start, size)`.

A workload is a sequence of requests:

* **allocation** of a contiguous block of `s` bytes: a free block with
  `size >= s` is chosen; if it is larger, the remainder is returned to the
  free list as a new (smaller) block; if no block fits, the allocation fails;
* **deallocation** of a previously allocated block.

Adjacent free blocks are coalesced whenever the free list changes
(`MemoryEnv._merge_free_blocks`).

Four evaluation metrics are computed on the end-of-round free list
(`rlmalloc/metrics.py`):

| Metric | Formula | Direction |
|---|---|---|
| Occupancy | `1 − S_total / M` | higher better |
| Duration | number of successful allocations | higher better |
| Fragmentation | `1 − max(lᵢ) / S_total` | lower better |
| HHI | `Σ(lᵢ / S_total)²` | higher = more concentrated |

`S_total = Σᵢ lᵢ` is the total free space.

## 2. MDP formulation

### State (`2k + 1 = 11`) — `MemoryEnv._get_state`

For the current request, build the candidate set with
`rlmalloc/candidates.py::build_candidates` (sort free blocks by start address
ascending, keep the first `k = 5` with `size >= request`). Each candidate `i`
contributes two normalised values `[size_i / M, start_i / M]`; the last element
is `request / M`. Unused candidate slots are zero-padded.

Because the **same** `build_candidates` function feeds both the state encoder
and the action decoder, the block the agent sees at slot `i` is exactly the
block that action `i` allocates. (An earlier version sorted the two
differently, which silently invalidated the learned policy; that is why the
candidate builder is a single shared function and is covered by a regression
test.)

### Action (`0 .. k−1`) — `MemoryEnv.step`

Action `i` selects candidate `i`. `rlmalloc/policies.py::dqn_action` is a thin
adapter that returns the agent's greedy index; it does **not** re-sort blocks.

### Reward — `MemoryEnv._reward`

`R = Σ(lᵢ / S_total)²` over the free-block set, and `R = 0` when `S_total = 0`.
This is the HHI of the free-space shares, bounded in `[0, 1]`: `R = 1` when the
free space is one contiguous block, `R = 1/N` for `N` equal fragments. See
[`DESIGN.md`](DESIGN.md) for why this is a *proxy* objective.

### Invalid action — `MemoryEnv.step`

If the action index is out of range for the candidate set, the environment
returns reward `-1.0` **and** `done = True`, so the episode ends immediately.

### Termination — `MemoryEnv.is_done`

True when the event list is exhausted (evaluation) or no free block fits the
current request (both training and evaluation).

## 3. Network — `rlmalloc/agent.py::QNetwork`

```
11 → Linear(128) → ReLU → Linear(128) → ReLU → Linear(5)
```

Trained as a standard value-based DQN: experience replay
(`REPLAY_BUFFER_SIZE = 10000`), a target network copied every
`TARGET_UPDATE_FREQ = 50` environment steps, SmoothL1 loss, Adam
(`LEARNING_RATE = 1e-4`), epsilon-greedy behaviour with
`1.0 → 0.01` multiplicative decay (`0.999` per episode).

All hyperparameters live in `rlmalloc/config.py`.

## 4. Training — `rlmalloc/train.py`

* epsilon-greedy action selection; epsilon decayed once per episode;
* one gradient step every `LEARN_EVERY` environment steps (default 1);
* target-network copy every `TARGET_UPDATE_FREQ = 50` environment steps;
* per-episode CSV log: `episode, return, steps, duration, epsilon, mean_loss,
  occupancy_end`;
* a bounded `MAX_STEPS_PER_EPISODE` safety cap prevents runaway episodes.

Run:

```bash
micromamba run -n test-py312 python -m rlmalloc.train \
  --dist lognormal_train --episodes 10000 --max-steps 2000 \
  --seed 0 --device cuda --out results/checkpoints/agent \
  --log results/metrics/train_log.csv
```

## 5. Workloads — `rlmalloc/workloads.py`

`DISTRIBUTIONS` (in `config.py`) defines four request-size distributions. In all
of them sizes are clipped to `[1, 512]`.

| Name | Spec |
|---|---|
| `lognormal_train` | `clip(Lognormal(ln 32, σ=0.9), 1, 512)` — default training workload |
| `lognormal_large` | `clip(Lognormal(ln 128, σ=0.7), 1, 512)` |
| `uniform` | `Uniform{1..512}` |
| `bimodal` | 70% `Lognormal(ln 16, σ=0.9)` + 30% `Lognormal(ln 256, σ=0.9)` |

`generate_workload` builds an explicit event list. It uses an **optimistic
reference timeline**: every allocation is assumed to succeed, so every `free`
event refers to an allocation ordinal that is guaranteed to be live on the
reference path. A real policy that fails early simply stops before reaching the
later free events and therefore never has to resolve an absent ordinal. All
policies in a round replay the exact same event list.

## 6. Baselines — `rlmalloc/policies.py`

`first_fit`, `best_fit` and `worst_fit` traverse **all** free blocks:

* **First-Fit** — the lowest-address block that fits.
* **Best-Fit** — the fitting block with the smallest leftover.
* **Worst-Fit** — the largest fitting block.

They therefore have a wider effective action set than the DQN's first-`k`
candidate set; this asymmetry is intrinsic to the current framing and is
discussed in [`DESIGN.md`](DESIGN.md).

## 7. Evaluation protocol — `rlmalloc/evaluate.py`

For each distribution and each round `r`: `rng = make_rng(seed, r)`, generate
one workload, then run **all** policies on that identical workload. Outputs:

* per-round long CSV (`results/metrics/eval_per_round.csv`);
* summary CSV/JSON (`eval_summary.csv`, `eval_summary.json`);
* a Markdown table (`results/tables/summary.md`);
* figures under `results/figures/`.

## 8. Code map

| Concern | Module |
|---|---|
| Candidate rule | `rlmalloc/candidates.py` |
| Environment / MDP | `rlmalloc/env.py` |
| Metrics | `rlmalloc/metrics.py` |
| Network / agent | `rlmalloc/agent.py` |
| Heuristics | `rlmalloc/policies.py` |
| Workloads | `rlmalloc/workloads.py` |
| Training | `rlmalloc/train.py` |
| Evaluation | `rlmalloc/evaluate.py` |
| Figures | `rlmalloc/plotting.py` |
| Config | `rlmalloc/config.py` |
