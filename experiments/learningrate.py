"""Run an experiment on the learningrate"""
import sys
from common import run_training, DEFAULT

if len(sys.argv) < 2:
    print("Please specify the learningrate")
    sys.exit(1)

ARGS = DEFAULT
ARGS["experiment_name"] = f"lr_{sys.argv[1]}"
ARGS["lr"] = float(sys.argv[1])
run_training(**ARGS)
