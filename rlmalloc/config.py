"""Central configuration for RLmalloc.

Values are grouped by concern.  Two tags are used:

* ``[FIXED]``  structural constants (arena size, state/action sizing).
* ``[TUNED]``  empirically chosen values (learning rate, schedules, limits).

See ``docs/DESIGN.md`` for the rationale and known limitations.
"""

from __future__ import annotations

import os

import numpy as np

# --------------------------------------------------------------------------- #
# Environment / MDP                                                           #
# --------------------------------------------------------------------------- #
MEMORY_SIZE = 4096              # simulated arena size (bytes)
N_CANDIDATE_BLOCKS = 5          # number of candidate blocks the agent sees
STATE_SIZE = N_CANDIDATE_BLOCKS * 2 + 1   # 2k + 1 = 11
ACTION_SIZE = N_CANDIDATE_BLOCKS          # action = candidate index in {0..k-1}
RELEASE_RATE = 0.3              # random-free probability after each allocation
MIN_REQUEST_SIZE = 1            # request sizes are clipped to [1, 512]
MAX_REQUEST_SIZE = 512
INVALID_ACTION_REWARD = -1.0    # reward for an out-of-range candidate index

# --------------------------------------------------------------------------- #
# DQN / RL hyper-parameters                                                   #
# --------------------------------------------------------------------------- #
GAMMA = 0.99                    # discount factor
BATCH_SIZE = 64                 # replay minibatch size
REPLAY_BUFFER_SIZE = 10000      # replay capacity
TARGET_UPDATE_FREQ = 50         # target-network copy interval (env steps)
HIDDEN_SIZE = 128               # width of the two hidden ReLU layers

LEARNING_RATE = 1e-4            # Adam learning rate
EPSILON_START = 1.0             # epsilon-greedy start
EPSILON_END = 0.01              # epsilon-greedy floor
EPSILON_DECAY = 0.999           # per-episode multiplicative decay
NUM_EPISODES = 10000            # default training episodes
MAX_STEPS_PER_EPISODE = 2000    # safety cap for one episode
LEARN_EVERY = 1                 # gradient-step cadence in env steps

# --------------------------------------------------------------------------- #
# Evaluation protocol                                                         #
# --------------------------------------------------------------------------- #
N_TEST_ROUNDS = 1000            # evaluation rounds per distribution
N_REQUESTS_PER_ROUND = 200      # allocation requests per round

# --------------------------------------------------------------------------- #
# Reproducibility / execution                                                 #
# --------------------------------------------------------------------------- #
SEED = 0
DEVICE = os.environ.get("RLMALLOC_DEVICE", "cpu")   # cpu for determinism

# --------------------------------------------------------------------------- #
# Request-distribution registry                                               #
# --------------------------------------------------------------------------- #
# ``mu`` is the mean of the underlying normal in log-space (np.log(centre)).
DISTRIBUTIONS = {
    # (1) default training workload: median 32 B, right-skewed
    "lognormal_train": dict(kind="lognormal", mu=float(np.log(32)), sigma=0.9),
    # (2) large-request workload
    "lognormal_large": dict(kind="lognormal", mu=float(np.log(128)), sigma=0.7),
    # (3) unbiased uniform
    "uniform": dict(kind="uniform", lo=1, hi=512),
    # (4) bimodal workload: 70% small (median 16), 30% large (median 256)
    "bimodal": dict(kind="bimodal", w=0.7,
                    mu_small=float(np.log(16)), mu_large=float(np.log(256)),
                    sigma=0.9),
}

DEFAULT_TRAIN_DIST = "lognormal_train"
EVAL_DISTS = ["lognormal_train", "lognormal_large", "uniform", "bimodal"]
