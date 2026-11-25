"""Replay a Kanake MuJoCo checkpoint or preview the built-in sine gait."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import math
import time

import _bootstrap  # Initialize source-tree imports.

import mujoco
import numpy as np

from kanake_mujoco.config import ACTION_SIZE, CONTROL_DT, TASKS, TERRAINS, EnvConfig
from kanake_mujoco.env import KanakeEnv
from kanake_mujoco.model import surface_height


def _arrow(geom, start, direction, length, color):
    end = start + length * np.array([math.cos(direction), math.sin(direction), 0.0])
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                        np.eye(3).ravel(), np.array(color, dtype=np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.012, start, end)


def draw_command(handle, env):
    """Green: commanded velocity (length ∝ speed). Blue: head gaze target from the head."""
    scene = handle.user_scn
    if scene is None:
        return
    count = 0
    if env.cfg.task == "navigation":
        xy = env.goal
        height = surface_height(env.model, env.data, xy)
        mujoco.mjv_initGeom(scene.geoms[count], mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, 0.04),
                            np.array([*xy, height + 0.05]), np.eye(3).ravel(),
                            np.array([0.2, 0.85, 0.35, 0.8], dtype=np.float32))
        count += 1
    speed = float(np.linalg.norm(env.command[:2]))
    if speed > 1e-6:
        start = env.com.copy()
        start[2] += 0.1
        _arrow(scene.geoms[count], start, env.travel_yaw, 2.0 * speed, (0.2, 0.85, 0.35, 1))
        count += 1
    head = env.data.xpos[env.head_id].copy()
    head[2] += 0.04
    _arrow(scene.geoms[count], head, env.head_target, 0.15, (0.2, 0.45, 0.95, 1))
    scene.ngeom = count + 1


def policy_for_checkpoint(path, device):
    import torch

    from kanake_rl.ppo import KanakeRunner, load_checkpoint, resolve_device
    from kanake_rl.vector import KanakeVecEnv

    device = resolve_device(device)
    torch.set_num_threads(1)
    state, metadata = load_checkpoint(path)
    cfg = EnvConfig(**metadata["environment"])
    with_env = KanakeVecEnv(cfg, 1)
    try:
        runner = KanakeRunner(with_env, metadata["ppo"], cfg, device=device)
        runner.restore(state)
        model = runner.get_inference_policy()
        def policy(obs):
            with torch.inference_mode():
                tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
                return model(tensor)[0].cpu().numpy()
        return cfg, policy
    finally:
        with_env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--task", choices=TASKS, help="Task for the demo without a checkpoint")
    parser.add_argument("--terrain", choices=TERRAINS)
    parser.add_argument("--park-zone", type=int)
    commands = parser.add_mutually_exclusive_group()
    commands.add_argument("--command", type=float, nargs="+", metavar="X Y [HEAD_YAW [PITCH HEIGHT]]",
                          help="Fixed world velocity (m/s) or navigation goal (m), and head yaw offset (rad)")
    commands.add_argument("--random-commands", action="store_true", help="Replace a saved fixed command")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--terrain-level", type=int, choices=range(5),
                        help="Random terrain: pin the curriculum level (0 easiest … 4 Teraffe's hardest)")
    parser.add_argument("--collision", choices=("off", "only", "both"), default="off",
                        help="Show the capsule collision geoms (group 3) instead of or with the CAD meshes")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error("Steps must be positive")
    if args.checkpoint and args.task:
        parser.error("A checkpoint already specifies the task")
    try:
        cfg = EnvConfig(task=args.task or "locomotion")
        # Zero actions run the default serpenoid wave without residuals.
        demo = np.zeros(ACTION_SIZE)
        def policy(_observation):
            return demo
        if args.checkpoint:
            cfg, policy = policy_for_checkpoint(args.checkpoint, args.device)
        overrides = {}
        if args.terrain is not None and args.terrain != cfg.terrain:
            overrides.update(terrain=args.terrain, park_zone=None)
        if args.park_zone is not None:
            overrides["park_zone"] = args.park_zone
        if args.seed is not None:
            overrides["seed"] = args.seed
        if args.command is not None:
            if len(args.command) not in (2, 3, 5):
                raise ValueError("--command takes X Y [HEAD_YAW [HEAD_PITCH HEAD_HEIGHT]]")
            overrides["fixed_command"] = tuple(args.command) if len(args.command) == 5 else (*args.command, 0.0)[:3]
        elif args.random_commands:
            overrides["fixed_command"] = None
        cfg = replace(cfg, **overrides)
    except (ValueError, OSError) as error:
        parser.error(str(error))

    if cfg.terrain == "random" and cfg.lateral != "force":
        cfg = replace(cfg, lateral="force")
    env = KanakeEnv(cfg)
    if cfg.terrain == "random":
        env.fixed_level = args.terrain_level if args.terrain_level is not None else 4
    obs = env.reset(neutral=not args.checkpoint)
    refresh = bool(env.model.nhfield)
    context = nullcontext()
    if not args.headless:
        from mujoco import viewer
        context = viewer.launch_passive(env.model, env.data)
    limit = args.steps or (800 if args.headless else float("inf"))
    step = resets = 0
    totals = {}
    with context as handle:
        if handle is not None:
            with handle.lock():
                handle.cam.distance = 1.8
                handle.cam.azimuth = 135
                handle.cam.elevation = -30
                if args.collision != "off":
                    handle.opt.geomgroup[3] = 1
                    handle.opt.geomgroup[1] = int(args.collision == "both")
        physics_lock = handle.lock if handle is not None else nullcontext
        while step < limit and (handle is None or handle.is_running()):
            start = time.monotonic()
            if refresh and handle is not None:  # Upload a new heightfield outside the viewer lock.
                handle.update_hfield(0)
                print(f"terrain: level {env.terrain_level} {env.terrain_kind}", flush=True)
            refresh = False
            action = policy(obs)
            with physics_lock():
                obs, _, terminated, truncated, info = env.step(action)
                step += 1
                for key, value in info.items():
                    if not key.startswith("reward_"):
                        totals[key] = totals.get(key, 0.0) + value
                if terminated or truncated:
                    obs = env.reset(neutral=not args.checkpoint)
                    resets += 1
                    refresh = bool(env.model.nhfield)
                if handle is not None:
                    handle.cam.lookat[:] = env.data.xipos[env.body_ids].mean(axis=0)
                    draw_command(handle, env)
            if handle is not None:
                handle.sync()
                time.sleep(max(0.0, CONTROL_DT - (time.monotonic() - start)))
    if step:
        means = " ".join(f"{key}={value / step:.3f}" for key, value in totals.items())
        print(f"steps={step} resets={resets} {means}")


if __name__ == "__main__":
    main()
