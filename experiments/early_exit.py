"""Run training with default options"""
import sys
from common import run_training

if len(sys.argv) < 2:
    print("Please specify the early exit factor to use")
    sys.exit(1)

run_training(
    experiment_name=f"early_exit_{sys.argv[1]}",
    early_exit_factor=float(sys.argv[1]),
)
