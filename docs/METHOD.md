# Method — spec to code mapping

Paper: *A DQN-Based Hybrid Decision Framework for Dynamic Memory Allocation*
(Sicheng Yang, Jing Ma). The PDF is kept at the repository root and copied to
`docs/assets/paper.pdf`.

## 1. Problem setup (paper Section II.A)

Memory of size `M = 4096` (`rlmalloc/config.py: MEMORY_SIZE`). A free-block
set `F = {(a_i, l_i)}` is maintained as a sorted list; an allocated-block
dictionary maps allocation ordinal -> `(start, size)`.

Metrics (`rlmalloc/metrics.py`):

| Metric | Formula | Direction |
|---|---|---|
| Occupancy | `1 - S_total / M` | higher better |
| Duration | number of successful allocations | higher better |
| Fragmentation | `1 - max(l_i) / S_total` | lower better |
| HHI | `sum((l_i / S_total)^2)` | higher = more concentrated |

## 2. MDP (paper Section III.B)

### State (`2k+1 = 11`) — `rlmalloc/env.py::_get_state` (line 143)
For the current request, build the candidate set with
`rlmalloc/candidates.py::build_candidates` (line 20) (sort by start address ascending,
keep the first `k = 5` blocks with `size >= request`). Each candidate `i`
contributes `[size/M, start/M]`; the final element is `request/M`. Empty
slots are zero-padded. Because state building and action decoding share the
same function, the historical state/action mismatch cannot recur.

### Action (`A = {0..k-1}`) — `rlmalloc/env.py::step` (line 206)
Action `i` selects candidate `i`. `policies.dqn_action` is a thin adapter that
just returns the agent's greedy index; it does **not** re-sort blocks.

### Reward (Eq.1) — `rlmalloc/env.py::_reward` (line 159)
`R = sum((l_i / S_total)^2)` over the free-block set, `0` when `S_total = 0`.
Bounded in `[0, 1]`. See `docs/DEVIATIONS.md` for the HHI convention note.

### Invalid action — `rlmalloc/env.py::step` (line 206)
If the action index is out of range for the candidate set, return reward
`-1.0` and `done = True` (paper p.3 III.B.4).

### Termination — `rlmalloc/env.py::is_done` (line 165)
True when the event list is exhausted (evaluation) or no free block fits the
current request (both modes).

## 3. Network (paper Section III.C.1) — `rlmalloc/agent.py::QNetwork` (line 27)

`11 -> 128 -> ReLU -> 128 -> ReLU -> 5`, trained with the standard DQN
recipe: experience replay (`C = 10000`), target network (`T = 50` **steps**),
SmoothL1 loss, Adam. Hyperparameters live in `rlmalloc/config.py` and are
tagged `[PAPER]` or `[CODE]`.

## 4. Training (Algorithm 1) — `rlmalloc/train.py`

* epsilon-greedy action selection, epsilon decayed once per episode;
* one gradient step every `LEARN_EVERY` env steps (default 1);
* target-network copy every `TARGET_UPDATE_FREQ = 50` env steps;
* per-episode CSV log: `episode, return, steps, duration, epsilon,
  mean_loss, occupancy_end`.

## 5. Workloads (paper Section IV.B) — `rlmalloc/workloads.py`

`DISTRIBUTIONS` (in `config.py`):

| Name | Spec |
|---|---|
| `lognormal_train` | `clip(Lognormal(ln32, 0.9), 1, 512)` — Eq.(2) |
| `lognormal_large` | `clip(Lognormal(ln128, 0.7), 1, 512)` |
| `uniform` | `Uniform{1..512}` |
| `bimodal` | 70% `Lognormal(ln16, 0.9)` + 30% `Lognormal(ln256, 0.9)` |

Evaluation event lists are generated from an *optimistic reference timeline*
(all allocations succeed), so every `free` event references a live allocation
ordinal. All policies in a round replay the exact same list.

## 6. Baselines (paper Section IV.A) — `rlmalloc/policies.py`

`first_fit`, `best_fit`, `worst_fit` traverse **all** free blocks, exactly as
the paper defines them. This is intentionally a wider search than the DQN's
first-`k` candidate set.

## 7. Evaluation — `rlmalloc/evaluate.py`

For each distribution and round `r`: `rng = make_rng(seed, r)`, build one
workload, run all policies on it. Writes per-round and summary CSV/JSON, a
Markdown table (`results/tables/summary.md`), and figures under
`results/figures/`.
