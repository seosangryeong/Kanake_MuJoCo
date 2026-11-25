"""Open a Kanake robot and terrain in the native MuJoCo viewer."""

from __future__ import annotations

import argparse

import mujoco
from mujoco import viewer

from .config import TERRAINS
from .model import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain", choices=TERRAINS, default="flat")
    args = parser.parse_args()
    model = build_model(args.terrain)
    viewer.launch(model, mujoco.MjData(model))


if __name__ == "__main__":
    main()
