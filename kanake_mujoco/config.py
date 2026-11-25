"""Simulation and task settings shared by training and playback."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

JOINT_NAMES = tuple(f"j{i}" for i in range(1, 17))
BODY_NAMES = ("tail", *(f"Link{i}" for i in range(1, 16)), "head")
# j1 is next to the head. Even joints bend in the ground plane, odd joints pitch.
YAW_INDEX = tuple(range(1, 16, 2))
PITCH_INDEX = tuple(range(0, 16, 2))
TERRAINS = ("flat", "ramps", "park", "random")
TASKS = ("locomotion", "navigation")
CONTROL_DT = 0.025
PHYSICS_SUBSTEPS = 10
# Serpenoid amplitude, frequency and steering offset, then a residual for every joint.
GAIT_SIZE = 3
ACTION_SIZE = GAIT_SIZE + len(JOINT_NAMES)
# Command 2, head 9, body 6, tail 6, joints 3×16, CPG 5, filtered residuals 16, previous action.
OBS_SIZE = 2 + 9 + 6 + 6 + 4 * len(JOINT_NAMES) + 5 + ACTION_SIZE


@dataclass(frozen=True)
class EnvConfig:
    task: str = "locomotion"
    terrain: str = "flat"
    park_zone: int | None = None
    episode_steps: int = 800
    speed_scale: float = 0.15
    # (vx, vy, head yaw) or, with head_lift, (vx, vy, head yaw, head pitch, head height).
    fixed_command: tuple[float, ...] | None = None
    # Sample head yaw targets relative to the travel direction (radians, 0 disables).
    head_yaw_range: float = 0.0
    # j2 is a dedicated neck outside the serpenoid wave (False reproduces version-2 checkpoints).
    neck_joint: bool = True
    # Raise the front (j1–j7) like a cobra and track head height and gaze pitch commands.
    head_lift: bool = False
    head_pitch_range: float = 0.35
    head_height_min: float = 0.05
    head_height_max: float = 0.10
    lift_curriculum_steps: int = 24000
    randomize: bool = False
    seed: int = 42
    collision: str = "capsule"
    friction_axial: float = 0.2
    friction_lateral: float = 1.0
    kp: float = 3.0
    kv: float = 1.0
    # "pair" (planes only) or "force" (any surface); the random terrain requires "force".
    lateral: str = "pair"
    # Wide body pitch residuals, 0.5 s proprioceptive history and a terrain-aware critic.
    climb: bool = False
    climb_pitch: float = 0.8  # rad residual range of the body pitch joints j3–j15 when climbing.
    climb_yaw: float = 0.15  # rad residual range of the body yaw joints j4–j16 when climbing.
    upright_weight: float = 0.5  # With flips or pushes; >0.5 also scales tracking rewards by uprightness.
    # v9: every wave joint (j4–j16) gets its own amplitude action; frequency and steering stay shared.
    amplitude_per_joint: bool = False
    # Contact-based support reward (links touching nothing are penalized) and a link roll-rate penalty.
    support_reward: bool = False
    # Robustness randomization (Teraffe-style): start rolled over, random pushes, observation noise.
    flip_probability: float = 0.0
    push_interval: float = 0.0  # Mean seconds between pushes; 0 disables.
    obs_noise: float = 0.0

    def __post_init__(self):
        if self.task not in TASKS or self.terrain not in TERRAINS:
            raise ValueError("Unknown task or terrain")
        if self.park_zone is not None and (self.terrain != "park" or not 1 <= self.park_zone <= 15):
            raise ValueError("Park zones 1–15 are available on the park terrain")
        if self.episode_steps < 1 or not isfinite(self.speed_scale) or self.speed_scale <= 0 or self.seed < 0:
            raise ValueError("Episode length and speed scale must be positive and the seed must be nonnegative")
        if self.terrain == "random" and self.lateral != "force":
            raise ValueError("Random terrain needs --lateral force")
        if not 0 <= self.head_yaw_range <= 1.5:
            raise ValueError("Head yaw range must be within 0–1.5 rad")
        if self.amplitude_per_joint and (not self.neck_joint or self.head_lift):
            raise ValueError("Per-joint amplitudes need the neck-joint layout without head lift")
        if self.fixed_command is not None:
            if len(self.fixed_command) not in (3, 5) or not all(map(isfinite, self.fixed_command)):
                raise ValueError("A fixed command must contain three or five finite numbers")
        if not 0.02 <= self.head_height_min <= self.head_height_max <= 0.25:
            raise ValueError("Head heights must satisfy 0.02 ≤ min ≤ max ≤ 0.25 m")

    @property
    def physics(self):
        from .model import PhysicsConfig

        return PhysicsConfig(self.collision, self.friction_axial, self.friction_lateral, self.kp, self.kv,
                             self.lateral)


WAVE_JOINTS = len(YAW_INDEX) - 1  # j4–j16 with the neck-joint layout.


def action_size(cfg):
    """Shared frequency and steering, one amplitude per wave joint, then 16 residuals (v9);
    otherwise amplitude, frequency, steering and 16 residuals."""
    return 2 + WAVE_JOINTS + len(JOINT_NAMES) if cfg.amplitude_per_joint else ACTION_SIZE
