"""Run an experiment on the early exit factor"""
import sys
from common import run_training, DEFAULT

if len(sys.argv) < 2:
    print("Please specify the number of processing steps to use")
    sys.exit(1)

ARGS = DEFAULT
ARGS["experiment_name"] = f"num_processing_steps_{sys.argv[1]}"
ARGS["num_processing_steps"] = int(sys.argv[1])

# use an easy problem to begin with
ARGS["generator_args"]["num_sources_dist"] = lambda _: 1
ARGS["generator_args"]["interm_nodes_dist"] = lambda _: 5
ARGS["generator_args"]["interm_blocks_dist"] = lambda _: 3

run_training(**ARGS)
