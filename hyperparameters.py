"""Commonly used hyperparameters and utility functions"""

import numpy as np
from gym_environment import SUPPORTED_EDGE_FEATURES, SUPPORTED_NODE_FEATURES

# reproducibility
STATE = np.random.RandomState(42)

DEFAULT = {
    "learnsteps": 100000,
    "train_freq": 1,
    "batch_size": 32,
    "early_exit_factor": np.infty,
    "seedgen": lambda: STATE.randint(0, 2 ** 32),
    "experiment_name": "default",
    "prioritized": True,
    "node_feat_whitelist": SUPPORTED_NODE_FEATURES,
    "node_feat_blacklist": frozenset(),
    "edge_feat_whitelist": SUPPORTED_EDGE_FEATURES,
    "edge_feat_blacklist": frozenset(),
}
