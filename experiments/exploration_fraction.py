"""Run an experiment on the early exit factor"""
import sys
from common import run_training, DEFAULT

if len(sys.argv) < 2:
    print("Please specify the fraction over which exploration should decay")
    sys.exit(1)

ARGS = DEFAULT
ARGS["experiment_name"] = f"exploration_fraction_{sys.argv[1]}"
ARGS["exploration_fraction"] = float(sys.argv[1])
ARGS["learning_starts"] = (
    ARGS["learnsteps"] * ARGS["train_freq"] * float(sys.argv[1]) * 0.2
)
run_training(**ARGS)
