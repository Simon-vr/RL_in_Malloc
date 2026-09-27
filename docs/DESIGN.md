# Design, assumptions and limitations

This document records the deliberate design choices, the assumptions that are
not otherwise visible in the code, and the known limitations of the current
approach. It is meant to be read together with [`METHOD.md`](METHOD.md).

## 1. Why a candidate set — and the asymmetry it creates

At each step the environment hands the agent only the first `k = 5` fitting
free blocks (sorted by start address); the agent picks among them. This keeps
the action space small and fixed (`k` discrete actions), which makes a small
network viable.

The deliberate cost: the **heuristic baselines scan all free blocks**, so they
have a strictly larger effective action space than the DQN. In particular,
Best-Fit's globally best block is frequently **not** in the DQN's candidate
set, so the DQN *cannot represent* the choice that makes Best-Fit strong. This
is the most likely reason the learned policy does not beat Best-Fit (see
[Section 8](#8-known-limitations)).

## 2. Reward design: HHI is a proxy objective

The reward is the HHI of the free-space shares, i.e. the extent to which free
bytes are concentrated in a few large blocks. The rationale is that a
concentrated free list is easier to satisfy future (including large) requests,
so it correlates with low external fragmentation.

It is important to be clear that this is a **proxy**, not the evaluation
objective. We *evaluate* occupancy, duration and fragmentation. A policy that
maximises a smooth concentration surrogate is not guaranteed to maximise
occupancy, and in practice it does not (the DQN trails Best-Fit on occupancy).

## 3. HHI: a single definition

HHI is defined once, in `rlmalloc/metrics.py`:

```
HHI = Σ(lᵢ / S_total)²      over the free blocks at the end of a round
```

* **Range:** for `N` free blocks, `1/N ≤ HHI ≤ 1`. One single free block →
  `HHI = 1`; `N` equal blocks → `HHI = 1/N`; no free memory → `HHI = 0`.
* **Direction:** higher is healthier (fewer, larger free blocks).
* **Bound:** since `Σ sᵢ² ≤ maxᵢ sᵢ · Σ sᵢ = maxᵢ sᵢ` for shares `sᵢ = lᵢ/S_total`,
  and `maxᵢ sᵢ = 1 − Fragmentation`, we always have
  `HHI ≤ 1 − Fragmentation`. A high HHI therefore implies low fragmentation.
* **Why Worst-Fit has the *lowest* HHI:** Worst-Fit always allocates from the
  largest block, progressively flattening the free list into many similarly
  sized medium blocks (largest share ≈ 0.13 vs ≈ 0.34–0.37 for First-/Best-Fit).
  Near-equal shares minimise `Σ sᵢ²`, so Worst-Fit scores lowest.

The reward uses this same quantity; there is no second "complement" metric.

## 4. Invalid actions terminate the episode

If the agent picks an index outside the candidate set, the episode ends
immediately with reward `-1.0`. This discourages invalid picks, but during
early training (high `epsilon`) many random actions are invalid, so episodes
are short and learning starts from a mostly-negative signal. This is expected
behaviour under the current design, not a bug.

## 5. Evaluation workload semantics

Evaluation workloads are generated on an **optimistic reference timeline** in
which every allocation succeeds, so each `free` event refers to an allocation
ordinal that is guaranteed to be live on that reference path. A policy that
fails early simply stops before reaching later free events. All policies in a
round replay the exact same event list, so comparisons are paired.

Free events reference **allocation ordinals** (the n-th successful allocation),
not raw addresses; this avoids the earlier failure mode in which a random free
id could reference an allocation that never happened and be silently ignored.

## 6. Hyperparameters that are choices, not derivations

The following were chosen by hand and are **not** tuned systematically:

| Parameter | Value | Note |
|---|---|---|
| learning rate | `1e-4` | Adam |
| epsilon schedule | `1.0 → 0.01`, decay `0.999` / episode | |
| episodes | `10000` | |
| `LEARN_EVERY` | `1` | gradient-step cadence (speed knob) |
| `MAX_STEPS_PER_EPISODE` | `2000` | safety cap |
| bimodal σ | `0.9` | 70% / 30% mix |

Because these are not searched and only a single seed is used, small metric
differences (e.g. DQN vs First-Fit) may not be statistically meaningful.

## 7. Device and determinism

Training defaults to CPU (`--device cpu` or `RLMALLOC_DEVICE`) so that runs are
reproducible for a fixed seed. GPU training is opt-in. The full reference run
used an NVIDIA RTX 5060 Laptop GPU.

## 8. Known limitations

1. **Restricted action space** — first-`k` candidates only; cannot represent
   Best-Fit's global choice ([Section 1](#1-why-a-candidate-set--and-the-asymmetry-it-creates)).
2. **Proxy reward** — optimises HHI, not occupancy/duration/fragmentation.
3. **Limited observability** — no global free-list statistics or history.
4. **Distribution shift** — trained on one workload; the other three are
   out-of-distribution.
5. **Single seed, no error bars** — small gaps may be noise.
6. **Expensive early exploration** — invalid actions end the episode.

Net effect: the agent beats Worst-Fit clearly but does not beat Best-Fit, and
only roughly matches First-Fit.

## 9. Future work

* Widen or restructure the action space (e.g. candidates that include the
  best-fit block, or a pointer-style / factored action representation) so the
  agent can express strong heuristic choices.
* Shape or replace the reward toward the evaluated objectives (occupancy,
  duration) while keeping stability.
* Add richer state features (free-block count, total free bytes, largest free
  block, recent request statistics, temporal context).
* Train on a mixture of distributions and measure transfer.
* Run multiple seeds and report mean ± std; sweep learning rate and epsilon
  schedule.
* Compare against a learned "best-fit oracle" to isolate the cost of the
  candidate restriction.
