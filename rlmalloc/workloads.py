"""Request-distribution sampling and event-sequence generation.

Training requests are drawn i.i.d. from one of the four distributions in
:data:`rlmalloc.config.DISTRIBUTIONS`.  Evaluation uses an explicit event
list generated from an *optimistic reference timeline*: every allocation
succeeds, so every ``free`` event refers to an allocation ordinal that is
guaranteed to be live on the reference path.  A real policy that fails early
simply stops before it ever reaches the later free events.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .config import DISTRIBUTIONS, MAX_REQUEST_SIZE, MIN_REQUEST_SIZE

# ('alloc', ordinal, size) or ('free', ordinal)
Event = Tuple[str, int, int]

DEFAULT_RELEASE_RATE = 0.3


def sample_request(dist_cfg: dict, rng: np.random.Generator) -> int:
    """Sample a single allocation size (int in [1, 512]) for a distribution."""
    kind = dist_cfg["kind"]
    if kind == "lognormal":
        value = rng.lognormal(mean=dist_cfg["mu"], sigma=dist_cfg["sigma"])
    elif kind == "uniform":
        value = rng.integers(dist_cfg["lo"], dist_cfg["hi"] + 1)
    elif kind == "bimodal":
        if rng.random() < dist_cfg["w"]:
            value = rng.lognormal(mean=dist_cfg["mu_small"],
                                  sigma=dist_cfg["sigma"])
        else:
            value = rng.lognormal(mean=dist_cfg["mu_large"],
                                  sigma=dist_cfg["sigma"])
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown distribution kind: {kind!r}")

    value = int(round(float(value)))
    return int(np.clip(value, MIN_REQUEST_SIZE, MAX_REQUEST_SIZE))


def sample_request_by_name(dist: str, rng: np.random.Generator) -> int:
    return sample_request(DISTRIBUTIONS[dist], rng)


def generate_workload(
    dist: str,
    n_allocs: int = 200,
    release_rate: float = DEFAULT_RELEASE_RATE,
    rng: Optional[np.random.Generator] = None,
) -> List[Event]:
    """Generate an event list shared by every policy in a test round.

    After each (optimistic, always-successful) allocation we emit a ``free``
    event with probability ``release_rate`` targeting a uniformly chosen
    currently-live ordinal.  Because ordinals are only freed while live on
    this reference path, every free target is valid.
    """
    if dist not in DISTRIBUTIONS:
        raise KeyError(f"unknown distribution {dist!r}; "
                       f"available: {sorted(DISTRIBUTIONS)}")
    if rng is None:
        rng = np.random.default_rng()

    cfg = DISTRIBUTIONS[dist]
    events: List[Event] = []
    live: List[int] = []
    ordinal = 0

    for _ in range(n_allocs):
        size = sample_request(cfg, rng)
        events.append(("alloc", ordinal, size))
        live.append(ordinal)
        ordinal += 1

        if live and rng.random() < release_rate:
            idx = int(rng.integers(0, len(live)))
            freed = live.pop(idx)
            events.append(("free", freed, 0))

    return events
