# Known deviations, ambiguities, and assumptions

This document lists every place where the paper is ambiguous or silent and
states exactly what this implementation does, so results can be interpreted
honestly.

## 1. HHI: one unified definition (the paper's printed numbers contradict it)

This project uses exactly **one** HHI definition, the paper's Eq.(1) /
Section II.A formula:

    HHI = sum_i (l_i / S_total)^2 ,   S_total = sum_i l_i

applied to the end-of-round free blocks `l_i`. Higher = free space more
concentrated (fewer / larger free blocks); range `1/N <= HHI <= 1`. The
environment *reward* uses this same quantity. The legacy code instead reported
`1 - sum(...)`; that value is not an HHI and is **no longer reported** (there
is no `hhi_complement` column anywhere).

**The paper's printed numbers cannot be `sum(s_i^2)`.** For free-block shares
`s_i = l_i/S_total` (so `sum s_i = 1`, `s_i >= 0`):

    sum_i s_i^2  <=  max_i(s_i) * sum_i s_i  =  max_i(s_i)  =  1 - Fragmentation.

The paper reports DQN fragmentation `0.6521`, i.e. `1 - Fragmentation =
0.3479`, together with HHI `0.8165`. But `0.8165 > 0.3479`, which is
impossible for `sum(s_i^2)`. Its numbers instead behave like `1 - HHI`, a
*scatter / diversity* index that is **larger when free space is more
fragmented** (it ranks Worst-Fit highest, and our run reproduces that under
`1 - HHI`). So the paper's metric label/interpretation is inverted relative to
its own equation.

**Decision and consequence.** Only the definition-faithful `hhi` is reported.
Under it, DQN sits *below* First-Fit and Best-Fit (training distribution:
DQN 0.1798 < First-Fit 0.1921 < Best-Fit 0.2311), in line with DQN's slightly
higher fragmentation; the paper's claim that DQN has the *highest* HHI is an
artifact of the mislabeled (complement) metric. To compare against the paper's
printed table, compute `1 - HHI` yourself.

> Why Worst-Fit has the lowest `hhi`: Worst-Fit always allocates from the
> largest block, flattening the free list into many similar-sized blocks
> (largest share ~0.13 vs ~0.34-0.37 for First-/Best-Fit). Near-equal shares
> minimise `sum(s_i^2)` -- the paper's own "N equal fragments -> 1/N".

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
