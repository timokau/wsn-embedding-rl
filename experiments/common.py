"""A hack to make experiments directly executable"""

# import file module relative to current file
import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
# pylint: disable=wrong-import-position,unused-import
from dqn_agent import run_training
from hyperparameters import DEFAULT
