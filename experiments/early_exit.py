"""Run an experiment on the early exit factor"""
import sys
from common import run_training, DEFAULT

if len(sys.argv) < 2:
    print("Please specify the early exit factor to use")
    sys.exit(1)

ARGS = DEFAULT
ARGS["experiment_name"] = f"early_exit_{sys.argv[1]}"
ARGS["early_exit_factor"] = float(sys.argv[1])
run_training(**ARGS)
