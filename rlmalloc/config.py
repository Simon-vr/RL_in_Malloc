"""Central configuration for RLmalloc.

Every constant is tagged:

* ``[PAPER]``  value is explicitly pinned by the paper.
* ``[CODE]``   value is *not* given by the paper; this repository chooses it.

The two choice categories are explained in ``docs/DEVIATIONS.md``.
"""

from __future__ import annotations

import os

import numpy as np

# --------------------------------------------------------------------------- #
# Environment / MDP                                                           #
# --------------------------------------------------------------------------- #
MEMORY_SIZE = 4096              # [PAPER] p.4 IV.A: M = 4096
N_CANDIDATE_BLOCKS = 5          # [PAPER] p.2 III.B.1: k = 5
STATE_SIZE = N_CANDIDATE_BLOCKS * 2 + 1   # [PAPER] 2k + 1 = 11
ACTION_SIZE = N_CANDIDATE_BLOCKS          # [PAPER] A = {0, ..., k-1}
RELEASE_RATE = 0.3              # [PAPER] p.4 IV.A: 30% release probability
MIN_REQUEST_SIZE = 1            # [PAPER] Eq.(2): clip to [1, 512]
MAX_REQUEST_SIZE = 512          # [PAPER] Eq.(2): clip to [1, 512]
INVALID_ACTION_REWARD = -1.0    # [PAPER] p.3 III.B.4

# --------------------------------------------------------------------------- #
# DQN / RL hyper-parameters (Algorithm 1)                                     #
# --------------------------------------------------------------------------- #
GAMMA = 0.99                    # [PAPER] Alg.1: gamma = 0.99
BATCH_SIZE = 64                 # [PAPER] Alg.1: B = 64
REPLAY_BUFFER_SIZE = 10000      # [PAPER] Alg.1: C = 10000
TARGET_UPDATE_FREQ = 50         # [PAPER] Alg.1: target update interval T = 50
HIDDEN_SIZE = 128               # [PAPER] p.3 III.C.1: two 128-dim ReLU layers

LEARNING_RATE = 1e-4            # [CODE] paper is silent; Adam default-ish
EPSILON_START = 1.0             # [CODE] paper is silent
EPSILON_END = 0.01              # [CODE] paper is silent
EPSILON_DECAY = 0.999           # [CODE] per episode, paper is silent
NUM_EPISODES = 10000            # [CODE] paper is silent; "E" in Alg.1
MAX_STEPS_PER_EPISODE = 2000    # [CODE] safety cap for one episode
LEARN_EVERY = 1                 # [CODE] gradient step cadence (speed knob)

# --------------------------------------------------------------------------- #
# Evaluation protocol                                                         #
# --------------------------------------------------------------------------- #
N_TEST_ROUNDS = 1000            # [PAPER] p.4 IV.B: 1000 rounds
N_REQUESTS_PER_ROUND = 200      # [PAPER] p.4 IV.A: 200 allocation requests

# --------------------------------------------------------------------------- #
# Reproducibility / execution                                                 #
# --------------------------------------------------------------------------- #
SEED = 0                        # [CODE]
DEVICE = os.environ.get("RLMALLOC_DEVICE", "cpu")   # [CODE] cpu for determinism

# --------------------------------------------------------------------------- #
# Request-distribution registry                                               #
# --------------------------------------------------------------------------- #
# Four workloads from the paper (p.4 IV.B).  ``mu`` is the mean of the
# underlying normal in log-space (i.e. np.log(centre)).
#
# NOTE: only the *centres* (16/32/128/256) are given by the paper; the
# bimodal sigma = 0.9 is an explicit assumption documented in
# docs/DEVIATIONS.md.
DISTRIBUTIONS = {
    # (1) training distribution, Eq.(2)
    "lognormal_train": dict(kind="lognormal", mu=float(np.log(32)), sigma=0.9),
    # (2) large-request workload
    "lognormal_large": dict(kind="lognormal", mu=float(np.log(128)), sigma=0.7),
    # (3) unbiased uniform
    "uniform": dict(kind="uniform", lo=1, hi=512),
    # (4) bimodal shock workload (70% small mu=16, 30% large mu=256)
    "bimodal": dict(kind="bimodal", w=0.7,
                    mu_small=float(np.log(16)), mu_large=float(np.log(256)),
                    sigma=0.9),
}

DEFAULT_TRAIN_DIST = "lognormal_train"
EVAL_DISTS = ["lognormal_train", "lognormal_large", "uniform", "bimodal"]
