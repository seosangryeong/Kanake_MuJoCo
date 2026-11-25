"""Make source-tree entry points independent of ROS and system RSL-RL."""

from pathlib import Path
import sys

RL_ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(RL_ROOT.parent), str(RL_ROOT / "third_party" / "rsl_rl")]
