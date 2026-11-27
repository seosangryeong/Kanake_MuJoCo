"""Train or resume Kanake PPO in MuJoCo."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

from _bootstrap import RL_ROOT

import mujoco
import torch

from kanake_mujoco.config import TASKS, TERRAINS, EnvConfig
from kanake_rl.ppo import KanakeRunner, load_checkpoint, resolve_device, runner_config, warm_start
from kanake_rl.vector import KanakeVecEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--terrain", choices=TERRAINS)
    parser.add_argument("--park-zone", type=int)
    parser.add_argument("--episode-steps", type=int)
    parser.add_argument("--speed-scale", type=float, help="Maximum commanded speed (m/s)")
    parser.add_argument("--head-yaw-range", type=float,
                        help="Sample head yaw targets within ±this angle of the travel direction (rad)")
    parser.add_argument("--head-lift", action=argparse.BooleanOptionalAction, default=None,
                        help="Raise the front (j1–j7) and track head height and pitch commands")
    parser.add_argument("--head-pitch-range", type=float, help="Head pitch commands within ±this (rad)")
    parser.add_argument("--head-height-min", type=float, help="Lowest head height command (m)")
    parser.add_argument("--head-height-max", type=float, help="Highest head height command (m)")
    parser.add_argument("--lift-curriculum-steps", type=int,
                        help="Per-world steps over which the head height range widens")
    parser.add_argument("--collision", choices=("capsule", "mesh"))
    parser.add_argument("--friction-axial", type=float, help="Ground friction along each link")
    parser.add_argument("--friction-lateral", type=float, help="Ground friction across each link")
    parser.add_argument("--lateral", choices=("pair", "force"),
                        help="Sideways friction as contact pairs (planes only) or applied forces (any terrain)")
    parser.add_argument("--kp", type=float)
    parser.add_argument("--kv", type=float)
    commands = parser.add_mutually_exclusive_group()
    commands.add_argument("--command", type=float, nargs="+", metavar="X Y [HEAD_YAW [PITCH HEIGHT]]",
                          help="Fixed world velocity (m/s) or navigation goal (m), and head yaw offset (rad)")
    commands.add_argument("--random-commands", action="store_true", help="Resample commands when resuming")
    parser.add_argument("--randomize", action=argparse.BooleanOptionalAction, default=None,
                        help="Randomize per-world link masses and contact friction")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--workers", type=int, default=16, help="CPU simulation processes")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="PPO device")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--rollout", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-from", type=Path,
                        help="Warm-start weights (appended observations / rescaled residuals) from a checkpoint")
    parser.add_argument("--climb-pitch", type=float, help="Body pitch residual range when climbing (rad)")
    parser.add_argument("--climb-yaw", type=float, help="Body yaw residual range when climbing (rad)")
    parser.add_argument("--upright-weight", type=float,
                        help="Upright reward weight; above 0.5 tracking rewards also scale with uprightness")
    parser.add_argument("--amplitude-per-joint", action=argparse.BooleanOptionalAction, default=None,
                        help="One amplitude action per wave joint (j4–j16) instead of a shared amplitude")
    parser.add_argument("--support-reward", action=argparse.BooleanOptionalAction, default=None,
                        help="Climbing: penalize links touching nothing and link roll rate")
    parser.add_argument("--action-bound-penalty", type=float, help="Penalty weight on raw actions beyond ±1")
    parser.add_argument("--entropy-coef", type=float, help="PPO entropy bonus")
    parser.add_argument("--flip-probability", type=float, help="Share of episodes that start rolled over")
    parser.add_argument("--push-interval", type=float, help="Mean seconds between random pushes (0 off)")
    parser.add_argument("--obs-noise", type=float, help="Gaussian noise added to policy observations")
    parser.add_argument("--climb", action=argparse.BooleanOptionalAction, default=None,
                        help="Wide body pitch residuals, proprioceptive history and a terrain-aware critic")
    args = parser.parse_args()
    if min(args.num_envs, args.workers, args.iterations) < 1:
        parser.error("Environment, worker and iteration counts must be positive")
    try:
        device = resolve_device(args.device)
        cfg = EnvConfig()
        ppo = runner_config(24)
        if args.resume and args.init_from:
            raise ValueError("Use either --resume or --init-from")
        if args.resume or args.init_from:
            state, metadata = load_checkpoint(args.resume or args.init_from)
            init_env = metadata["environment"]
            cfg = EnvConfig(**metadata["environment"])
            ppo = metadata["ppo"]
            if args.resume and args.task is not None and args.task != cfg.task:
                raise ValueError("Resume requires the checkpoint's task")
        if args.entropy_coef is not None:
            ppo["algorithm"]["entropy_coef"] = args.entropy_coef
        if args.rollout is not None:
            ppo["num_steps_per_env"] = args.rollout
        fields = ("task", "terrain", "park_zone", "episode_steps", "speed_scale", "head_yaw_range",
                  "head_lift", "head_pitch_range", "head_height_min", "head_height_max", "lift_curriculum_steps",
                  "randomize", "seed", "collision", "friction_axial", "friction_lateral", "kp", "kv", "lateral", "climb", "climb_pitch", "climb_yaw", "upright_weight",
                  "amplitude_per_joint", "support_reward", "action_bound_penalty",
                  "flip_probability", "push_interval", "obs_noise")
        overrides = {key: getattr(args, key) for key in fields if getattr(args, key) is not None}
        if args.terrain is not None and args.terrain != cfg.terrain:
            overrides.setdefault("park_zone", None)
        if args.command is not None:
            if len(args.command) not in (2, 3, 5):
                raise ValueError("--command takes X Y [HEAD_YAW [HEAD_PITCH HEAD_HEIGHT]]")
            overrides["fixed_command"] = tuple(args.command) if len(args.command) == 5 else (*args.command, 0.0)[:3]
        elif args.random_commands:
            overrides["fixed_command"] = None
        cfg = replace(cfg, **overrides)
        batch = args.num_envs * ppo["num_steps_per_env"]
        if batch < 4 or batch % 4:
            raise ValueError("num-envs × rollout must be a positive multiple of 4")
    except (ValueError, OSError) as error:
        parser.error(str(error))

    torch.manual_seed(cfg.seed)
    torch.set_num_threads(1)
    run_dir = args.log_dir or RL_ROOT / "runs" / f"{cfg.task}_{cfg.terrain}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "environment": asdict(cfg), "ppo": ppo, "num_envs": args.num_envs, "workers": args.workers,
        "device": device, "mujoco_version": mujoco.__version__,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Run directory: {run_dir}", flush=True)
    env = KanakeVecEnv(cfg, args.num_envs, min(args.workers, args.num_envs))
    runner = None
    try:
        runner = KanakeRunner(env, ppo, cfg, str(run_dir), device)
        if args.resume:
            runner.restore(state, resume=True)
        elif args.init_from:
            warm_start(runner, state, init_env, cfg)
        runner.learn(args.iterations)
    finally:
        env.close()
        if runner is not None and runner.writer is not None:
            runner.writer.close()


if __name__ == "__main__":
    main()
