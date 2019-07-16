"""Commonly used hyperparameters and utility functions"""

import math
import numpy as np
import generator as g
from features import features_by_name

# reproducibility
STATE = np.random.RandomState(42)

GENERATOR_DEFAULTS = {
    "interm_nodes_dist": lambda r: round(g.truncnorm(r, mean=5, sd=3, low=0)),
    "pos_dist": lambda r: r.uniform(low=(0, 0), high=(25, 25)),
    "capacity_dist": lambda r: g.truncnorm(r, mean=10, sd=5, low=0),
    "power_dist": lambda r: r.normal(30, 2),
    "interm_blocks_dist": lambda r: round(g.truncnorm(r, mean=3, sd=2, low=0)),
    "pairwise_connection": lambda r: r.rand() < 0.01,
    "block_weight_dist": lambda r: g.truncnorm(r, mean=5, low=0, sd=2),
    # mean equivalent to a linear SINRth of 20, which is what marvelo uses
    "requirement_dist": lambda r: g.truncnorm(
        r, mean=math.log(1 + 20, 2), low=0, sd=1
    ),
    "num_sources_dist": lambda r: round(g.truncnorm(r, mean=2, sd=1, low=1)),
    "connection_choice": lambda r, a: r.choice(a),
}

DEFAULT_FEATURES = [
    features_by_name()[name]
    for name in ["node_relay", "edge_additional_timeslot"]
]

DEFAULT = {
    "learnsteps": 20000,
    "train_freq": 1,
    "batch_size": 32,
    "early_exit_factor": 1.5,
    "num_processing_steps": 20,
    "latent_size": 16,
    "num_layers": 2,
    "seedgen": lambda: STATE.randint(0, 2 ** 32),
    "experiment_name": "default",
    "prioritized": True,
    "features": DEFAULT_FEATURES,
    "generator_args": GENERATOR_DEFAULTS,
    "exploration_fraction": 0.5,
    "rl_seed": STATE.randint(0, 2 ** 32),
}
