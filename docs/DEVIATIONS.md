# Known deviations, ambiguities, and assumptions

This document lists every place where the paper is ambiguous or silent and
states exactly what this implementation does, so results can be interpreted
honestly.

## 1. HHI convention (real ambiguity)

* **Paper** Eq.(1) and Section II.A define
  `HHI = sum((l_i / S_total)^2)` (raw sum, higher = more concentrated).
* **Legacy code** reported `1 - sum((l_i / S_total)^2)`.

**Decision:** the reported metric is the paper-faithful **raw sum** `hhi`.
The summary additionally exposes `hhi_complement = 1 - hhi`, so readers used
to the legacy convention can map values directly. The environment *reward*
always uses the raw sum (it always did).

The paper is internally inconsistent here: while Eq.(1)/Section II.A define
the raw sum, the experimental numbers it reports (e.g. DQN 0.8165,
First-Fit 0.7942, Best-Fit 0.7548) are only consistent with `1 -` raw sum.
Our full run reproduces that ordering and magnitude under the complement
convention (DQN 0.8202 > First-Fit 0.8079 > Best-Fit 0.7689), while the raw
sum is correspondingly small (~0.18). Both columns are emitted; use
`hhi_complement` to compare against the paper's printed tables.

> Consequence: a raw `hhi` of ~0.20 corresponds to a legacy/paper-style value
> of ~0.80. Do not compare raw `hhi` to the numbers printed in the legacy
> `result.txt`.

## 2. Hyperparameters the paper does not specify

Only `B = 64`, `gamma = 0.99`, `C = 10000`, and `T = 50` are pinned. These
are **code choices** (tagged `[CODE]` in `rlmalloc/config.py`):

| Parameter | Value | Note |
|---|---|---|
| learning rate | `1e-4` | Adam |
| epsilon schedule | `1.0 -> 0.01`, decay `0.999` per episode | |
| episodes `E` | `10000` | paper's `E` in Algorithm 1 is unspecified |
| `LEARN_EVERY` | `1` | pure speed knob; default matches one step per transition |
| `MAX_STEPS_PER_EPISODE` | `2000` | safety cap |

Because `LR`, `E`, and the epsilon schedule are unspecified, **exact
reproduction of the paper's numbers is not claimed**.

## 3. Bimodal distribution sigma (assumption)

The paper gives only the two centres (16 and 256) for the bimodal workload.
This implementation assumes `sigma = 0.9` for both components (matching the
training distribution's skew) and mixes `70% / 30%`.

## 4. Candidate-set restriction

The DQN can only select among the first `k = 5` free blocks that fit, sorted
by start address. Heuristic baselines search **all** free blocks, so they have
a larger effective action set. This asymmetry is intrinsic to the framework
and is documented rather than "fixed".

## 5. Invalid action is terminal

When the agent picks an index outside the candidate set, the episode ends
immediately with reward `-1.0` (paper p.3 III.B.4). During early training,
when `epsilon` is high and the candidate set is small, many random actions are
invalid, so episodes are short. This is expected behaviour, not a bug.

## 6. Evaluation free-event IDs

Evaluation workloads reference frees by **successful-allocation ordinal**,
generated on an optimistic all-succeed reference timeline
(`rlmalloc/workloads.py::generate_workload`). The legacy code chose random
allocation IDs that could already be freed, causing silent no-op frees. The
new model guarantees every free target is live on the reference path; a policy
that fails early simply never reaches later free events.

## 7. Device / determinism

Training defaults to `cpu` (`RLMALLOC_DEVICE` env var or `--device`). GPU is
opt-in. The full paper-scale run was executed on an NVIDIA RTX 5060 Laptop
GPU; CPU timings are documented in `docs/REPRODUCE.md`.
