"""Isolated MuJoCo worlds with RSL-RL auto-reset and timeout flags.

With more than one worker, each worker process owns a slice of the worlds, so physics and reward
code run in parallel without the GIL.
"""

import multiprocessing as mp
from dataclasses import asdict

import numpy as np
import torch
from rsl_rl.env import VecEnv

from kanake_mujoco.config import action_size
from kanake_mujoco.env import KanakeEnv
from kanake_mujoco.model import build_model

TRACKING_KEYS = ("velocity_error_mps", "speed_mps", "command_speed_mps", "head_yaw_error_deg",
                 "head_rate_dps", "head_spin_dps", "head_sway_mps", "head_tilt_deg", "head_height_error_cm", "head_height_cm", "head_pitch_error_deg", "head_roll_deg", "body_heading_error_deg", "slip_ratio", "frequency_hz",
                 "amplitude_deg", "goal_distance_m", "terrain_level", "head_slope_error_deg", "air_fraction", "unsupported_fraction", "steps_climbed", "top_reached", "stair_height_cm")


class _Worlds:
    """Observations come back as (policy, critic); critic is None without climb mode."""

    def __init__(self, cfg, seeds):
        model = build_model(cfg.terrain, cfg.physics)
        self.worlds = [KanakeEnv(cfg, model, seed) for seed in seeds]
        self.critic = cfg.climb

    def _pack(self, observations):
        policy = np.stack(observations)
        if not self.critic:
            return policy, None
        return policy, np.stack([w.privileged(o) for w, o in zip(self.worlds, observations)])

    def observe(self):
        return self._pack([world.observe() for world in self.worlds])

    def reset(self):
        return self._pack([world.reset() for world in self.worlds])

    def step(self, actions):
        out = []
        for world, action in zip(self.worlds, actions, strict=True):
            obs, reward, terminated, truncated, info = world.step(action)
            done = terminated or truncated
            if done:
                obs = world.reset()
            critic = world.privileged(obs) if self.critic else None
            out.append((obs, reward, done, truncated, info, world.elapsed_steps, critic))
        return out


def _worker(pipe, cfg, seeds):
    worlds = _Worlds(cfg, seeds)
    while True:
        command, payload = pipe.recv()
        if command == "close":
            break
        pipe.send(getattr(worlds, command)(*payload))
    pipe.close()


class KanakeVecEnv(VecEnv):
    def __init__(self, cfg, num_envs, workers=1):
        self.cfg = asdict(cfg)
        self.device = torch.device("cpu")
        self.num_envs = num_envs
        self.num_actions = action_size(cfg)
        self.max_episode_length = cfg.episode_steps
        seeds = np.random.SeedSequence(cfg.seed).spawn(num_envs)
        workers = max(1, min(workers, num_envs))
        self.slices = np.array_split(np.arange(num_envs), workers)
        self.local = None
        self.pipes, self.processes = [], []
        if workers == 1:
            self.local = _Worlds(cfg, seeds)
        else:
            context = mp.get_context("spawn")
            for part in self.slices:
                parent, child = context.Pipe()
                process = context.Process(target=_worker, args=(child, cfg, [seeds[i] for i in part]),
                                          daemon=True)
                process.start()
                child.close()
                self.pipes.append(parent)
                self.processes.append(process)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)

    def _call(self, command, per_worker=None):
        if self.local is not None:
            return [getattr(self.local, command)(*(per_worker[0] if per_worker else ()))]
        for i, pipe in enumerate(self.pipes):
            pipe.send((command, per_worker[i] if per_worker else ()))
        return [pipe.recv() for pipe in self.pipes]

    def _join(self, parts):
        policy = torch.from_numpy(np.concatenate([p for p, _ in parts]))
        extras = {}
        if parts[0][1] is not None:
            extras["critic"] = torch.from_numpy(np.concatenate([c for _, c in parts]))
        return policy, {"observations": extras}

    def get_observations(self):
        return self._join(self._call("observe"))

    def reset(self):
        self.episode_length_buf.zero_()
        return self._join(self._call("reset"))

    def step(self, actions):
        actions = actions.detach().cpu().numpy().astype(np.float64)
        results = [r for part in self._call("step", [(actions[s],) for s in self.slices]) for r in part]
        observations, rewards, dones, timeouts, infos, steps, critics = zip(*results)
        self.episode_length_buf[:] = torch.as_tensor(steps)
        log = {}
        for key in {k for info in infos for k in info}:
            values = [info[key] for info in infos if key in info]
            group = "Tracking" if key in TRACKING_KEYS else "Reward"
            log[f"{group}/{key}"] = torch.tensor(float(np.mean(values)))
        return (
            torch.from_numpy(np.stack(observations)),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.bool),
            {"observations": {"critic": torch.from_numpy(np.stack(critics))} if critics[0] is not None else {},
             "time_outs": torch.tensor(timeouts, dtype=torch.bool), "log": log},
        )

    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send(("close", ()))
            except (BrokenPipeError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        self.pipes, self.processes = [], []
