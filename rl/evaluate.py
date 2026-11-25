"""Score a Kanake checkpoint (or the zero-action serpenoid) on fixed command scenarios.

Each scenario starts straight along +X, holds one world-frame command and averages the metrics
after a settling time. Columns: speed along the command, velocity error, head yaw error and
yaw rate (gaze stability), body heading error and the sideways share of link motion (slip).
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import _bootstrap  # Initialize source-tree imports.

import numpy as np

from kanake_mujoco.config import ACTION_SIZE, EnvConfig
from kanake_mujoco.env import KanakeEnv

SCENARIOS = (
    ("stop", (0.0, 0.0, 0.0)),
    ("0.05 straight", (0.05, 0.0, 0.0)),
    ("0.10 straight", (0.10, 0.0, 0.0)),
    ("0.15 straight", (0.15, 0.0, 0.0)),
    ("0.10 turn 90°", (0.0, 0.10, 0.0)),
)
HEAD_SCENARIOS = (
    ("0.10 head +30°", (0.10, 0.0, 0.52)),
    ("0.10 head −30°", (0.10, 0.0, -0.52)),
    ("0.05 head +40°", (0.05, 0.0, 0.70)),
    ("stop head −40°", (0.0, 0.0, -0.70)),
)
LIFT_SCENARIOS = (
    ("stop h8", (0.0, 0.0, 0.0, 0.0, 0.08)),
    ("0.05 h8", (0.05, 0.0, 0.0, 0.0, 0.08)),
    ("0.10 h8", (0.10, 0.0, 0.0, 0.0, 0.08)),
    ("0.05 h10", (0.05, 0.0, 0.0, 0.0, 0.10)),
    ("0.05 h6 turn90", (0.0, 0.05, 0.0, 0.0, 0.06)),
    ("0.05 h8 pitch+20", (0.05, 0.0, 0.0, 0.35, 0.08)),
    ("0.05 h8 pitch−20", (0.05, 0.0, 0.0, -0.35, 0.08)),
    ("0.05 h8 yaw+40", (0.05, 0.0, 0.70, 0.0, 0.08)),
)
LIFT_COLUMNS = ("head_height_cm", "head_pitch_error_deg", "head_roll_deg")
COLUMNS = ("speed_mps", "velocity_error_mps", "head_yaw_error_deg", "head_spin_dps", "head_sway_mps",
           "head_tilt_deg", "body_heading_error_deg", "slip_ratio", "frequency_hz", "amplitude_deg")


def run(cfg, policy, command, seconds, settle, seeds):
    rows = []
    for seed in seeds:
        env = KanakeEnv(replace(cfg, fixed_command=command, seed=seed, randomize=False))
        obs = env.reset(neutral=True)
        totals, count, falls = {}, 0, 0
        direction = np.array(command[:2]) / max(np.linalg.norm(command[:2]), 1e-9)
        for step in range(int(seconds / 0.025)):
            obs, _, terminated, truncated, info = env.step(policy(obs))
            if terminated:
                falls += 1
                obs = env.reset(neutral=True)
            if step * 0.025 < settle:
                continue
            info = dict(info, speed_mps=float(env.filtered_velocity @ direction) if command[:2] != (0.0, 0.0)
                        else float(np.linalg.norm(env.filtered_velocity)))
            for key in COLUMNS + (LIFT_COLUMNS if cfg.head_lift else ()):
                totals[key] = totals.get(key, 0.0) + info[key]
            count += 1
        rows.append({key: value / count for key, value in totals.items()} | {"falls": falls})
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--settle", type=float, default=5.0)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--head", action="store_true", help="Add head yaw offset scenarios")
    parser.add_argument("--friction-axial", type=float)
    parser.add_argument("--friction-lateral", type=float)
    args = parser.parse_args()
    cfg = EnvConfig()

    def policy(_obs):
        return np.zeros(ACTION_SIZE)

    if args.checkpoint:
        from play import policy_for_checkpoint

        cfg, policy = policy_for_checkpoint(args.checkpoint, "cpu")
    overrides = {k: getattr(args, k) for k in ("friction_axial", "friction_lateral") if getattr(args, k)}
    cfg = replace(cfg, terrain="flat", park_zone=None, task="locomotion", **overrides)
    scenarios = SCENARIOS + (HEAD_SCENARIOS if args.head else ())
    columns = COLUMNS
    if cfg.head_lift:
        scenarios, columns = LIFT_SCENARIOS, COLUMNS + LIFT_COLUMNS
    header = f"{'scenario':<18}" + "".join(f"{c.split('_')[0] + '_' + c.split('_')[1][:4]:>13}" for c in columns) + "  falls"
    print(header)
    for name, command in scenarios:
        result = run(cfg, policy, command, args.seconds, args.settle, range(args.seeds))
        print(f"{name:<18}" + "".join(f"{result[c]:>13.3f}" for c in columns) + f"  {result['falls']:.0f}",
              flush=True)


if __name__ == "__main__":
    main()
