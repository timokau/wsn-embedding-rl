"""Run an experiment on the early exit factor"""
import sys
from common import run_training, DEFAULT

if len(sys.argv) < 3:
    print("Please specify the amount of nodes and blocks to use")
    sys.exit(1)

ARGS = DEFAULT
ARGS["experiment_name"] = f"early_exit_{sys.argv[1]}_{sys.argv[2]}"
ARGS["generator_args"]["num_sources_dist"] = lambda _: 1
ARGS["generator_args"]["interm_nodes_dist"] = lambda _: int(sys.argv[1]) - 1
ARGS["generator_args"]["interm_blocks_dist"] = lambda _: int(sys.argv[1]) - 1
run_training(**ARGS)
