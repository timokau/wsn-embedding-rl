"""Run an experiment on the value of features"""
import sys
from common import run_training, DEFAULT
from features import features_by_name

if len(sys.argv) < 2:
    print("Please specify which feature to add")
    sys.exit(1)

ARGS = DEFAULT
ARGS["experiment_name"] = f"extra_features_{'+'.join(sys.argv[1:])}"
ARGS["features"] = ARGS["features"] + [
    features_by_name()[name] for name in sys.argv[1:]
]
run_training(**ARGS)
