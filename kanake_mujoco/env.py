"""Single-world Kanake serpentine control and reward on CPU MuJoCo.

The policy modulates a serpenoid travelling wave on the eight yaw joints (amplitude, frequency,
steering offset) and adds a residual to every joint. The wave runs from head to tail so the body
slides along its own S-curve; the head joints' large residuals let the policy hold the gaze steady.
All directional observations are expressed relative to the commanded travel direction.
"""

from __future__ import annotations

import copy
import math

import mujoco
import numpy as np

from .config import (ACTION_SIZE, BODY_NAMES, CONTROL_DT, GAIT_SIZE, JOINT_NAMES, PHYSICS_SUBSTEPS,
                     PITCH_INDEX, YAW_INDEX, EnvConfig, action_size)
from . import procedural
from .friction import LateralFriction
from .model import build_model, surface_height
from .terrain import PARK_ZONES

YAW = np.array(YAW_INDEX)
PITCH = np.array(PITCH_INDEX)
WAVE_LAG = math.pi / 4  # One full wave over the eight yaw joints.
GAIT_FILTER = CONTROL_DT / 0.2
VELOCITY_FILTER = CONTROL_DT / 0.4
# Low-passed residuals cannot drive the high-frequency vibration that also slides on
# anisotropic friction; the serpenoid wave must do the propulsion.
BODY_FILTER = CONTROL_DT / 0.08
HEAD_FILTER = CONTROL_DT / 0.03


NECK_LIFT = 7  # j1–j7 hold the raised front (four pitch, three yaw joints).
# Level, stable neck poses for j1, j3, j5, j7 found by CEM (kp=10, kv=0.5) while the rear body
# crawls, keyed by the head height they reach (m). The policy adds residuals around the pose for
# the commanded height, which also absorbs the pitch the poses leave (up to ~14°).
LIFT_HEIGHTS = np.array([0.055, 0.072, 0.092])
LIFT_POSES = np.array([[0.109, -0.356, -0.056, 0.056],
                       [0.451, -0.389, -0.243, 0.060],
                       [0.514, -0.477, -0.513, 0.248]])
LIFT_WAVES = 0.75  # Wavelengths over the grounded j8–j16; better than one in the gait sweep.


def lift_pose(height):
    """Neck pitch targets (j1, j3, j5, j7) interpolated from the CEM poses."""
    return np.array([np.interp(height, LIFT_HEIGHTS, column) for column in LIFT_POSES.T])


HISTORY_FRAMES, HISTORY_STRIDE = 10, 2  # 0.5 s of proprioception for climbing.
# Critic-only height samples around the body centre, in the travel frame.
SCAN_ALONG = np.linspace(-0.4, 0.6, 11)
SCAN_SIDE = np.linspace(-0.2, 0.2, 5)


def residual_layout(neck_joint, head_lift=False, climb=False, climb_pitch=0.8, climb_yaw=0.15):
    """Per-joint residual scale and filter gain, the wave joints and their phase lags.

    With ``neck_joint`` j2 leaves the wave and becomes a dedicated neck: the policy owns its full
    range and both head joints react within ~30 ms. Without it (checkpoint version 2) j2 rides the
    wave and every residual uses the slow body filter. ``head_lift`` makes j1–j7 a fast, wide neck
    that raises the front like a cobra; one full wave then runs over the grounded j8–j16.
    Returns scale, filter gain, wave joint indices, wave phase lags and the neck joint count.
    """
    scale = np.full(len(JOINT_NAMES), 0.15)
    gain = np.full(len(JOINT_NAMES), BODY_FILTER)
    scale[:2] = 1.0  # j1 pitches and j2 yaws the head.
    wave, neck = YAW, 2
    if head_lift:
        scale[:NECK_LIFT] = 0.5  # Around the neck pose.
        scale[[3, 5]] = 0.3  # j4, j6 keep the neck in its vertical plane.
        scale[1] = 1.2  # j2 aims the gaze.
        gain[:NECK_LIFT] = HEAD_FILTER
        wave, neck = YAW[YAW >= NECK_LIFT], NECK_LIFT
        return scale, gain, wave, LIFT_WAVES * math.tau / len(wave) * np.arange(len(wave)), neck
    if neck_joint:
        scale[1] = 1.2
        gain[:2] = HEAD_FILTER
        wave = YAW[1:]
    if climb:
        scale[PITCH[1:]] = climb_pitch
        scale[YAW[1:]] = climb_yaw
    return scale, gain, wave, WAVE_LAG * np.searchsorted(YAW, wave), neck


def wrap_angle(angle):
    return (angle + math.pi) % math.tau - math.pi


def gait_parameters(action):
    """Map the first three actions to amplitude (rad), frequency (Hz) and steering offset (rad)."""
    a, f, g = np.tanh(action[:GAIT_SIZE])
    return np.array([0.525 * (1 + a), 0.75 + 0.45 * f, 0.5 * g])


def serpenoid_targets(gait, phase, residual, wave=YAW, lags=None, amplitudes=None):
    """Travelling wave from head to tail on the wave joints plus the filtered residuals."""
    amplitude, _, offset = gait
    if amplitudes is not None:
        amplitude = amplitudes
    target = residual.copy()
    lags = WAVE_LAG * np.searchsorted(YAW, wave) if lags is None else lags
    target[wave] += amplitude * np.sin(phase - lags) + offset
    return target


def yaw_of(rotation):
    return math.atan2(rotation[3], rotation[0])


def rotate(xy, angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * xy[0] + s * xy[1], -s * xy[0] + c * xy[1]])


class KanakeEnv:
    def __init__(self, cfg: EnvConfig, model=None, seed=None):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed if seed is None else seed)
        template = build_model(cfg.terrain, cfg.physics) if model is None else model
        # Random terrain rewrites the heightfield per world, so each world owns its model.
        self.model = copy.copy(template) if cfg.randomize or cfg.terrain == "random" else template
        self.data = mujoco.MjData(self.model)
        self.joint_qpos = np.array([self.model.joint(name).qposadr[0] for name in JOINT_NAMES])
        self.joint_dofs = np.array([self.model.joint(name).dofadr[0] for name in JOINT_NAMES])
        self.body_ids = np.array([self.model.body(name).id for name in BODY_NAMES])
        self.body_mass = self.model.body_mass[self.body_ids] / self.model.body_mass[self.body_ids].sum()
        self.root_id = self.model.body("tail").id
        self.head_id = self.model.body("head").id
        self.ctrl_low, self.ctrl_high = self.model.actuator_ctrlrange.T
        (self.residual_scale, self.residual_gain, self.wave, self.wave_lags,
         self.neck) = residual_layout(cfg.neck_joint, cfg.head_lift, cfg.climb, cfg.climb_pitch,
                                       cfg.climb_yaw)
        self.history = np.zeros((HISTORY_FRAMES * HISTORY_STRIDE, 38))
        self.grid = None  # Current procedural heightfield (m), for critic height samples.
        self.total_steps = 0  # Survives resets; drives the head height curriculum.
        mujoco.mj_forward(self.model, self.data)
        # The rolled link frames differ; remember which local axis points up when straight.
        self.head_up = self.data.xmat[self.head_id].reshape(3, 3).T @ np.array([0.0, 0.0, 1.0])
        # Each link's local axis that points up when the robot lies straight and upright.
        self.body_up = np.einsum("bji,j->bi", self.data.xmat[self.body_ids].reshape(-1, 3, 3), [0.0, 0.0, 1.0])
        self.base_gains = (self.model.actuator_gainprm[:, 0].copy(), self.model.actuator_biasprm[:, 1:3].copy())
        self.push_time = math.inf
        self.base_pair_friction = self.model.pair_friction[:, :2].copy()
        self.lateral = None
        if cfg.lateral == "force":
            capsules = np.array([g for g in range(self.model.ngeom)
                                 if (self.model.geom(g).name or "").endswith("_capsule")])
            self.capsules = capsules
            self.base_axial = self.model.geom_friction[capsules, 0].copy()
            self.lateral = LateralFriction(self.model, capsules, cfg.friction_lateral - cfg.friction_axial)
        # Per-world terrain curriculum (random terrain only); survives resets.
        self.terrain_level = 0
        self.fixed_level = None  # Playback can pin a level and skip promotion.
        self.terrain_override = None  # Playback can pin an exact (grid, kind).
        self.terrain_kind = "flat"
        self.episode_error, self.episode_samples, self.fell = 0.0, 0, False
        self.action_size = action_size(cfg)
        self.per_joint = cfg.amplitude_per_joint
        self.last_action = np.zeros(self.action_size)
        self.amplitudes = np.zeros(len(self.wave))
        # Velocity x, y, head yaw offset, head pitch and head height above the ground.
        self.command = np.zeros(5)
        self.reset()

    # --- kinematics ---------------------------------------------------------------------------

    @property
    def root_position(self):
        return self.data.xpos[self.root_id]

    @property
    def com(self):
        return self.body_mass @ self.data.xipos[self.body_ids]

    @property
    def head_yaw(self):
        return yaw_of(self.data.xmat[self.head_id])

    @property
    def body_heading(self):
        axes = self.data.xmat[self.body_ids][:, [0, 3]]
        axes = axes / np.maximum(np.linalg.norm(axes, axis=1, keepdims=True), 1e-9)
        return math.atan2(axes[:, 1].sum(), axes[:, 0].sum())

    @property
    def heading(self):
        return self.body_heading

    def _randomize_model(self):
        masses = self.model.body_mass[self.body_ids]
        scale = self.rng.uniform(0.9, 1.1, size=len(masses))
        self.model.body_mass[self.body_ids] = masses * scale
        self.model.body_inertia[self.body_ids] *= scale[:, None]
        if self.lateral is not None:
            axial = self.rng.uniform(0.7, 1.4) * self.cfg.friction_axial
            lateral = self.rng.uniform(0.8, 1.2) * self.cfg.friction_lateral
            self.model.geom_friction[self.capsules, 0] = axial
            self.lateral.extra_mu = max(lateral - axial, 0.0)
        elif self.model.npair:
            friction = self.base_pair_friction * [self.rng.uniform(0.7, 1.4), self.rng.uniform(0.8, 1.2)]
            self.model.pair_friction[:, :2] = friction
        else:
            robot = self.model.geom_contype == 2
            self.model.geom_friction[robot, 0] *= self.rng.uniform(0.7, 1.3)
        if self.cfg.flip_probability > 0 or self.cfg.push_interval > 0:
            # Actuator stiffness and damping ±10 % (position actuator: gain kp, bias −kp, −kv).
            gain, bias = self.base_gains
            k, d = self.rng.uniform(0.9, 1.1, 2)
            self.model.actuator_gainprm[:, 0] = gain * k
            self.model.actuator_biasprm[:, 1] = bias[:, 0] * k
            self.model.actuator_biasprm[:, 2] = bias[:, 1] * d
        mujoco.mj_setConst(self.model, self.data)

    # --- commands -----------------------------------------------------------------------------

    def _sample_command(self):
        """command = world velocity (x, y) and head yaw offset from the travel direction."""
        self.command_time_left = self.rng.uniform(6.0, 10.0)
        fixed = self.cfg.fixed_command
        if self.cfg.task == "navigation":
            # Navigation steers the same velocity command toward a goal position.
            if fixed is not None:
                self.goal = np.array(fixed[:2], dtype=float)
                self.command_time_left = math.inf
            else:
                self.goal = self.root_position[:2] + self.rng.uniform(-2.0, 2.0, size=2)
                self.command_time_left = self.rng.uniform(15.0, 20.0)
            self.command[2] = 0.0
        elif fixed is not None:
            self.command[:3] = fixed[:3]
            self.command[3:] = fixed[3:] if len(fixed) == 5 else (0.0, self._height_range()[1])
            self.command_time_left = math.inf
        else:
            speed = 0.0 if self.rng.random() < 0.1 else self.rng.uniform(0.03, self.cfg.speed_scale)
            direction = self.body_heading + self.rng.uniform(-1.0, 1.0)
            self.command[:2] = speed * np.array([math.cos(direction), math.sin(direction)])
            self._sample_head()
        self.travel_yaw = self.body_heading
        self._update_travel()

    def _sample_head(self):
        # The gaze target changes more often than the travel command.
        self.head_time_left = self.rng.uniform(2.0, 5.0)
        if self.cfg.task == "locomotion" and self.cfg.fixed_command is None:
            self.command[2] = self.rng.uniform(-self.cfg.head_yaw_range, self.cfg.head_yaw_range)
            if self.cfg.head_lift:
                self.command[3] = self.rng.uniform(-self.cfg.head_pitch_range, self.cfg.head_pitch_range)
                self.command[4] = self.rng.uniform(*self._height_range())
        if self.cfg.task == "navigation" and self.cfg.head_lift:
            self.command[3:] = (0.0, self.cfg.head_height_max)

    def _height_range(self):
        """Head height targets widen from the minimum to the maximum over the curriculum."""
        low, high = self.cfg.head_height_min, self.cfg.head_height_max
        progress = min(1.0, self.total_steps / max(self.cfg.lift_curriculum_steps, 1))
        return low, low + progress * (high - low)

    def _update_travel(self):
        if self.cfg.task == "navigation":
            offset = self.goal - self.com[:2]
            distance = float(np.linalg.norm(offset))
            speed = min(self.cfg.speed_scale, 0.5 * distance) if distance > 0.1 else 0.0
            self.command[:2] = speed * offset / max(distance, 1e-9)
        if np.linalg.norm(self.command[:2]) > 1e-6:
            self.travel_yaw = math.atan2(self.command[1], self.command[0])

    @property
    def head_target(self):
        return wrap_angle(self.travel_yaw + self.command[2])

    def head_pose(self):
        """Head height above the ground, gaze pitch (positive up) and roll about the gaze axis."""
        rotation = self.data.xmat[self.head_id].reshape(3, 3)
        forward, up = rotation[:, 0], rotation @ self.head_up
        side = np.cross([0.0, 0.0, 1.0], forward)
        side /= max(np.linalg.norm(side), 1e-9)
        normal = np.cross(forward, side)
        centre = self.data.xipos[self.head_id]
        height = centre[2] - surface_height(self.model, self.data, centre[:2])
        pitch = math.asin(min(1.0, max(-1.0, forward[2])))
        return float(height), pitch, math.atan2(-float(up @ side), float(up @ normal))

    # --- episode ------------------------------------------------------------------------------

    def reset(self, *, neutral=False):
        mujoco.mj_resetData(self.model, self.data)
        if self.cfg.randomize:
            self._randomize_model()
        xy, yaw = (0.0, 0.0), self.rng.uniform(-math.pi, math.pi)
        if self.cfg.terrain == "ramps":
            xy, yaw = (1.1, 0.0), self.rng.uniform(-0.3, 0.3)
        elif self.cfg.terrain == "park":
            zone = self.cfg.park_zone or int(self.rng.integers(1, len(PARK_ZONES) + 1))
            _, _, x, y = PARK_ZONES[zone - 1]
            xy, yaw = (x - 4.9, y), self.rng.uniform(-0.3, 0.3)
        if self.cfg.terrain == "random":
            xy = tuple(self.rng.uniform(-1.0, 1.0, 2))
            self._next_terrain()
        if neutral:
            yaw = 0.0
        # The tail is the root; place it so the body centre lands on xy.
        centre = 0.38 * np.array([math.cos(yaw), math.sin(yaw)])
        mujoco.mj_forward(self.model, self.data)
        start = np.asarray(xy) - centre
        # Rest the straight body just above the highest ground point along it.
        line = start + np.outer(np.linspace(0, 1, 12), 2 * centre)
        ground = max(surface_height(self.model, self.data, p) for p in line)
        # Joint noise can bend the body a little below the line; uneven ground gets more margin.
        self.data.qpos[:3] = (*start, ground + (0.022 if self.cfg.terrain != "random" else 0.04))
        self.data.qpos[3:7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        if not neutral and self.rng.random() < self.cfg.flip_probability:
            # Roll the straight body about its long axis (the tail's x axis).
            roll = self.rng.uniform(-math.pi, math.pi)
            spin = np.array([math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0])
            mujoco.mju_mulQuat(self.data.qpos[3:7], self.data.qpos[3:7].copy(), spin)
        if not neutral:
            self.data.qpos[self.joint_qpos] = self.rng.uniform(-0.1, 0.1, len(JOINT_NAMES))
        self.data.ctrl[:] = self.data.qpos[self.joint_qpos]
        self.elapsed_steps = 0
        self.last_action.fill(0)
        self.gait, amplitudes, _ = self._split(self.last_action)
        if amplitudes is not None:
            self.amplitudes = amplitudes
        self.residual = np.zeros(len(JOINT_NAMES))
        self.neck_pose = np.zeros(4)  # Eases from flat to the commanded height's pose.
        self.phase = 0.0 if neutral else self.rng.uniform(0, math.tau)
        self.warning_count = 0
        mujoco.mj_forward(self.model, self.data)
        self.velocity = np.zeros(2)
        self.filtered_velocity = np.zeros(2)
        self.body_velocity = np.zeros((len(BODY_NAMES), 2))
        self.head_rate = 0.0
        self.head_velocity = np.zeros(2)
        self.head_time_left = math.inf
        self.command[:] = 0
        self.episode_error, self.episode_samples, self.fell = 0.0, 0, False
        self.history[:] = 0.0
        self.push_time = self._next_push()
        self._sample_command()
        self.info = {}
        return self.observe()

    def _next_terrain(self):
        """Promote or demote this world on its last episode, then load a terrain of that level."""
        if self.terrain_override is not None:
            grid, self.terrain_kind = self.terrain_override
            self.model.hfield_data[:] = grid.ravel() / procedural.HEIGHT_SCALE
            self.grid = grid
            return
        if self.fixed_level is not None:
            self.terrain_level = self.fixed_level
        elif self.episode_samples >= 80:  # At least 2 s of scored motion.
            error = self.episode_error / self.episode_samples
            if not self.fell and error < 0.035:
                self.terrain_level = min(self.terrain_level + 1, len(procedural.LEVEL_SCALES) - 1)
            elif self.fell or error > 0.07:
                self.terrain_level = max(self.terrain_level - 1, 0)
        grids, kinds = procedural.bank(self.cfg.seed)[self.terrain_level]
        if self.rng.random() < procedural.FLAT_FRACTIONS[self.terrain_level]:
            grid, self.terrain_kind = np.zeros_like(grids[0]), "flat"
        else:
            index = int(self.rng.integers(len(grids)))
            grid, self.terrain_kind = grids[index], kinds[index]
        self.model.hfield_data[:] = grid.ravel() / procedural.HEIGHT_SCALE
        self.grid = grid

    def clean_observation(self):
        ref = self.travel_yaw
        head = self.data.xmat[self.head_id].reshape(3, 3)
        tail = self.data.xmat[self.root_id].reshape(3, 3)
        head_error = wrap_angle(self.head_yaw - self.head_target)
        body_error = wrap_angle(self.body_heading - ref)
        head_velocity = np.empty(6)
        tail_velocity = np.empty(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.head_id, head_velocity, 1)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.root_id, tail_velocity, 1)
        return np.concatenate((
            (np.linalg.norm(self.command[:2]), self.command[2]),
            (math.sin(head_error), math.cos(head_error), self.head_rate), head[2], 0.25 * head_velocity[:3],
            (math.sin(body_error), math.cos(body_error)),
            rotate(self.filtered_velocity, ref), rotate(self.velocity, ref),
            tail[2], 0.25 * tail_velocity[:3],
            self.data.qfrc_actuator[self.joint_dofs],
            0.1 * self.data.qvel[self.joint_dofs],
            self.data.qpos[self.joint_qpos],
            (math.sin(self.phase), math.cos(self.phase)), self.gait,
            self.residual,
            self._compat_action(self.last_action),
            self._lift_observation(),
            self.history[HISTORY_STRIDE - 1::HISTORY_STRIDE].ravel() if self.cfg.climb else (),
            # v9 extras go last so older checkpoints' columns keep their positions for warm starts.
            (*self.last_action[2:2 + len(self.wave)], *self.amplitudes) if self.per_joint else (),
        )).astype(np.float32)

    def observe(self):
        obs = self.clean_observation()
        if self.cfg.obs_noise > 0:
            obs = obs + self.rng.normal(0.0, self.cfg.obs_noise, obs.shape).astype(np.float32)
        return obs

    def _split(self, action):
        """Gait (mean amplitude, frequency, steering), per-joint amplitudes or None, residual actions."""
        if not self.per_joint:
            return gait_parameters(action), None, action[GAIT_SIZE:]
        n = len(self.wave)
        f, g = np.tanh(action[:2])
        amplitudes = 0.525 * (1 + np.tanh(action[2:2 + n]))
        return np.array([amplitudes.mean(), 0.75 + 0.45 * f, 0.5 * g]), amplitudes, action[2 + n:]

    def _compat_action(self, action):
        """The previous action in the 19-value layout (mean amplitude action first)."""
        if not self.per_joint:
            return action
        n = len(self.wave)
        return np.concatenate(((action[2:2 + n].mean(), action[0], action[1]), action[2 + n:]))

    def unsupported_fraction(self):
        """Share of links touching nothing; contact with ground, a riser or an edge all count."""
        n = self.data.ncon
        supported = np.zeros(self.model.nbody, dtype=bool)
        supported[self.model.geom_bodyid[self.data.contact.geom1[:n]]] = True
        supported[self.model.geom_bodyid[self.data.contact.geom2[:n]]] = True
        return float(1.0 - supported[self.body_ids].mean())

    def roll_rate(self):
        """Mean squared spin of the links about their own long axes (rad²/s²)."""
        spin = self.data.cvel[self.body_ids, :3]
        axes = self.data.xmat[self.body_ids][:, [0, 3, 6]]
        return float(np.mean(np.sum(spin * axes, axis=1) ** 2))

    def _next_push(self):
        if self.cfg.push_interval <= 0:
            return math.inf
        return self.rng.uniform(0.5, 1.5) * self.cfg.push_interval

    def _push(self):
        """Kick the free base sideways and roll it about the body axis."""
        self.data.qvel[:2] += self.rng.normal(0.0, 0.15, 2)
        axis = self.data.xmat[self.root_id].reshape(3, 3)[:, 0]
        self.data.qvel[3:6] += self.data.xmat[self.root_id].reshape(3, 3).T @ (axis * self.rng.normal(0.0, 3.0))

    def terrain_head_error(self):
        """Head pitch relative to the ground slope beneath it, and head roll (rad).

        Climbing rewards a head that lies parallel to the local ground instead of level with
        gravity, so it rises along a bump rather than ploughing its nose into it.
        """
        rotation = self.data.xmat[self.head_id].reshape(3, 3)
        forward, up = rotation[:, 0], rotation @ self.head_up
        flat = np.array([forward[0], forward[1]])
        flat /= max(np.linalg.norm(flat), 1e-9)
        centre = self.data.xipos[self.head_id, :2]
        ahead, behind = self.ground_height(np.array([centre + 0.05 * flat, centre - 0.05 * flat]))
        slope = math.atan2(ahead - behind, 0.10)
        pitch = math.asin(min(1.0, max(-1.0, forward[2])))
        side = np.cross([0.0, 0.0, 1.0], forward)
        side /= max(np.linalg.norm(side), 1e-9)
        roll = math.atan2(-float(up @ side), float(up @ np.cross(forward, side)))
        return pitch - slope, roll

    def air_fraction(self):
        """Share of links more than 1 cm above the ground (radius 1.9 cm)."""
        centres = self.data.xipos[self.body_ids]
        clearance = centres[:, 2] - self.ground_height(centres[:, :2]) - 0.019
        return float(np.mean(clearance > 0.01))

    def upright(self):
        """Mean over links of how well each link's rest-up axis points up (1 upright, −1 upside down)."""
        rotation = self.data.xmat[self.body_ids].reshape(-1, 3, 3)
        return float(np.mean(np.einsum("bij,bj->bi", rotation, self.body_up)[:, 2]))

    def _proprioception(self):
        """One history frame: joint tracking error, actuator torque, head and tail up axes."""
        error = self.data.ctrl - self.data.qpos[self.joint_qpos]
        return np.concatenate((error, self.data.qfrc_actuator[self.joint_dofs],
                               self.data.xmat[self.head_id][6:], self.data.xmat[self.root_id][6:]))

    def ground_height(self, xy):
        """Heightfield height at points (N, 2), bilinear like the grid; zero off random terrain."""
        if self.grid is None:
            return np.zeros(len(xy))
        n = procedural.GRID
        p = np.clip((xy + procedural.HALF_EXTENT) / (2 * procedural.HALF_EXTENT) * (n - 1), 0, n - 1.001)
        i = p.astype(int)
        f = p - i
        g = self.grid
        top = g[i[:, 1], i[:, 0]] * (1 - f[:, 0]) + g[i[:, 1], i[:, 0] + 1] * f[:, 0]
        bottom = g[i[:, 1] + 1, i[:, 0]] * (1 - f[:, 0]) + g[i[:, 1] + 1, i[:, 0] + 1] * f[:, 0]
        return top * (1 - f[:, 1]) + bottom * f[:, 1]

    def privileged(self, obs):
        """Critic input: the policy observation plus terrain heights around the body and physics."""
        c, s = math.cos(self.travel_yaw), math.sin(self.travel_yaw)
        along, side = np.meshgrid(SCAN_ALONG, SCAN_SIDE, indexing="ij")
        points = self.com[:2] + np.stack((c * along - s * side, s * along + c * side), -1).reshape(-1, 2)
        heights = self.ground_height(points) - self.com[2]
        extra = self.lateral.extra_mu if self.lateral is not None else 0.0
        axial = self.model.geom_friction[self.capsules, 0].mean() if self.lateral is not None else 0.0
        if self.cfg.obs_noise > 0:
            obs = self.clean_observation()  # The critic sees noise-free values.
        return np.concatenate((obs, heights, (axial, extra, self.terrain_level / 4),
                               (self.upright(),))).astype(np.float32)

    def _lift_observation(self):
        if not self.cfg.head_lift:
            return ()
        height, pitch, roll = self.head_pose()
        return (self.command[3], 10.0 * self.command[4], 10.0 * height, pitch, roll)

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.action_size,) or not np.isfinite(action).all():
            raise ValueError(f"Expected {self.action_size} finite actions")
        gait, amplitudes, residual_action = self._split(action)
        self.gait += GAIT_FILTER * (gait - self.gait)
        if amplitudes is not None:
            self.amplitudes += GAIT_FILTER * (amplitudes - self.amplitudes)
        self.phase = (self.phase + math.tau * self.gait[1] * CONTROL_DT) % math.tau
        command = self.residual_scale * np.clip(residual_action, -1.0, 1.0)
        self.residual += self.residual_gain * (command - self.residual)
        target = serpenoid_targets(self.gait, self.phase, self.residual, self.wave, self.wave_lags,
                                   self.amplitudes if self.per_joint else None)
        if self.cfg.head_lift:
            self.neck_pose += HEAD_FILTER * (lift_pose(self.command[4]) - self.neck_pose)
            target[[0, 2, 4, 6]] += self.neck_pose
        self.data.ctrl[:] = np.clip(target, self.ctrl_low, self.ctrl_high)

        if self.elapsed_steps * CONTROL_DT >= self.push_time:
            self._push()
            self.push_time = self.elapsed_steps * CONTROL_DT + self._next_push()
        com, bodies, head_yaw = self.com[:2].copy(), self.data.xipos[self.body_ids, :2].copy(), self.head_yaw
        head_xy = self.data.xpos[self.head_id, :2].copy()
        if self.lateral is not None:
            self.lateral.step(self.model, self.data, PHYSICS_SUBSTEPS)
        else:
            mujoco.mj_step(self.model, self.data, nstep=PHYSICS_SUBSTEPS)
        mujoco.mj_forward(self.model, self.data)

        warning_count = sum(int(self.data.warning[i].number) for i in (
            mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
            mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADCTRL,
        ))
        unstable = warning_count > self.warning_count or not np.isfinite(self.data.qpos).all()
        self.warning_count = warning_count
        if self.cfg.climb:
            self.history = np.roll(self.history, -1, axis=0)
            self.history[-1] = self._proprioception()
        self.velocity = (self.com[:2] - com) / CONTROL_DT
        self.filtered_velocity += VELOCITY_FILTER * (self.velocity - self.filtered_velocity)
        self.body_velocity = (self.data.xipos[self.body_ids, :2] - bodies) / CONTROL_DT
        self.head_rate = wrap_angle(self.head_yaw - head_yaw) / CONTROL_DT
        self.head_velocity = (self.data.xpos[self.head_id, :2] - head_xy) / CONTROL_DT
        reward = -1.0 if unstable else self._reward(action)
        self.elapsed_steps += 1
        self.total_steps += 1
        self.last_action[:] = action

        clearance = self.root_position[2] - surface_height(self.model, self.data, self.root_position[:2])
        terminated = unstable or not -0.15 <= clearance <= 0.3
        # Reaching the heightfield edge ends the episode without counting as a failure.
        edge = self.cfg.terrain == "random" and np.abs(self.com[:2]).max() > procedural.HALF_EXTENT - 0.5
        truncated = (self.elapsed_steps >= self.cfg.episode_steps or edge) and not terminated
        self.fell |= bool(terminated)
        if self.elapsed_steps * CONTROL_DT > 2.0 and "velocity_error_mps" in self.info:
            self.episode_error += self.info["velocity_error_mps"]
            self.episode_samples += 1
        self.info["terrain_level"] = float(self.terrain_level)

        self.command_time_left -= CONTROL_DT
        self.head_time_left -= CONTROL_DT
        if not (terminated or truncated):
            if self.command_time_left <= 0:
                self._sample_command()
            else:
                self._update_travel()
                if self.head_time_left <= 0:
                    self._sample_head()
        return self.observe(), reward, terminated, truncated, self.info

    def _slip(self):
        """Mean squared sideways link speed and the share of link motion that is sideways."""
        # BODY_NAMES runs tail to head; a raised neck (head, Link1–Link6) is off the ground.
        grounded = slice(0, len(BODY_NAMES) - self.neck) if self.cfg.head_lift else slice(None)
        axes = self.data.xmat[self.body_ids[grounded]][:, [0, 3]]
        axes = axes / np.maximum(np.linalg.norm(axes, axis=1, keepdims=True), 1e-9)
        velocity = self.body_velocity[grounded]
        along = np.sum(velocity * axes, axis=1)
        lateral = axes[:, 0] * velocity[:, 1] - axes[:, 1] * velocity[:, 0]
        ratio = np.abs(lateral).mean() / max(np.abs(along).mean() + np.abs(lateral).mean(), 1e-6)
        return float(np.mean(lateral ** 2)), float(ratio)

    def _reward(self, action):
        target = self.command[:2]
        velocity_error = float(np.linalg.norm(self.filtered_velocity - target))
        head_error = wrap_angle(self.head_yaw - self.head_target)
        body_error = wrap_angle(self.body_heading - self.travel_yaw)
        slip, slip_ratio = self._slip()
        head_up = float(self.data.xmat[self.head_id].reshape(3, 3)[2] @ self.head_up)
        head_tilt = math.acos(min(1.0, max(-1.0, head_up)))
        # The neck's pitch joints and residuals are free; the grounded body stays flat and on the wave.
        pitch = self.data.qpos[self.joint_qpos[PITCH[PITCH >= (self.neck if self.cfg.head_lift else 0)]]]
        residual = self.residual[self.neck:]
        if self.cfg.climb:
            # Climbing needs the body pitch joints; only the yaw residuals keep their full penalty.
            residual = np.concatenate((self.residual[YAW[YAW >= self.neck]], 0.25 * self.residual[PITCH[1:]]))
        power = np.abs(self.data.qfrc_actuator[self.joint_dofs] * self.data.qvel[self.joint_dofs]).sum()
        spin = np.empty(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.head_id, spin, 0)
        head_spin = float(np.linalg.norm(spin[:3]))
        # Sideways head sway relative to the travel direction; a steady gaze also keeps the head on track.
        sway = float(rotate(self.head_velocity, self.travel_yaw)[1])
        terms = {
            "velocity": 2.0 * math.exp(-(velocity_error / (0.015 if self.cfg.head_lift else 0.04)) ** 2),
            "head_yaw": 1.5 * math.exp(-(head_error / 0.15) ** 2),
            "head_still": 1.0 * math.exp(-(head_spin / 0.3) ** 2),
            "head_rate": -0.1 * head_spin ** 2,
            "head_sway": -10.0 * sway ** 2,
            "heading": 0.5 * math.exp(-(body_error / 0.5) ** 2),
            "head_level": 0.5 * math.exp(-(head_tilt / 0.15) ** 2),
            "slip": -30.0 * slip,
            "pitch": (-0.02 if self.cfg.climb else -0.2) * float(np.sum(pitch ** 2)),
            "residual": -1.0 * float(np.sum(residual ** 2)),
            "action_rate": -0.01 * float(np.sum(np.square(action - self.last_action))),
            "power": -0.002 * float(power),
        }
        climb_info = {}
        if self.cfg.flip_probability > 0 or self.cfg.push_interval > 0:
            upright = self.upright()
            terms["upright"] = self.cfg.upright_weight * upright
            if self.cfg.upright_weight > 0.5:
                # While rolled over, tracking pays in proportion to uprightness: recover first.
                gate = max(upright, 0.0)
                for key in ("velocity", "head_yaw", "head_still", "heading", "head_level"):
                    if key in terms:
                        terms[key] *= gate
        if self.cfg.climb:
            # Terrain-relative head attitude replaces gravity-level, and grounded links are rewarded.
            pitch_error, roll = self.terrain_head_error()
            terms["head_level"] = 0.5 * math.exp(-(pitch_error ** 2 + roll ** 2) / 0.15 ** 2)
            air = self.air_fraction()
            if self.cfg.support_reward:
                unsupported = self.unsupported_fraction()
                terms["supported"] = -0.5 * unsupported
                # Rolling is how the snake rights itself, so only penalize it while upright.
                terms["roll_rate"] = -0.005 * self.roll_rate() * max(self.upright(), 0.0)
                climb_info = dict(head_slope_error_deg=abs(math.degrees(pitch_error)), air_fraction=air,
                                  unsupported_fraction=unsupported)
            else:
                terms["grounded"] = -0.5 * air
                climb_info = dict(head_slope_error_deg=abs(math.degrees(pitch_error)), air_fraction=air)
        self.info = {
            "velocity_error_mps": velocity_error,
            "speed_mps": float(np.linalg.norm(self.filtered_velocity)),
            "command_speed_mps": float(np.linalg.norm(target)),
            "head_yaw_error_deg": abs(math.degrees(head_error)),
            "head_rate_dps": abs(math.degrees(self.head_rate)),
            "head_spin_dps": math.degrees(head_spin),
            "head_sway_mps": abs(sway),
            "head_tilt_deg": math.degrees(head_tilt),
            "body_heading_error_deg": abs(math.degrees(body_error)),
            "slip_ratio": slip_ratio,
            "frequency_hz": float(self.gait[1]),
            "amplitude_deg": math.degrees(self.gait[0]),
        }
        self.info.update(climb_info)
        if self.cfg.head_lift:
            height, pitch_angle, roll = self.head_pose()
            del terms["head_level"]
            terms["head_height"] = 1.5 * math.exp(-((height - self.command[4]) / 0.03) ** 2)
            # Narrow and wide kernels: precise near the target, still a gradient when far off.
            pitch_error = pitch_angle - self.command[3]
            terms["head_pitch"] = 0.5 * math.exp(-(pitch_error / 0.15) ** 2) + 0.5 * math.exp(-(pitch_error / 0.5) ** 2)
            terms["head_roll"] = 0.5 * math.exp(-(roll / 0.15) ** 2) + 0.5 * math.exp(-(roll / 0.5) ** 2)
            terms["head_yaw_wide"] = 0.5 * math.exp(-(head_error / 0.5) ** 2)
            self.info.update(head_height_error_cm=100 * abs(height - self.command[4]),
                             head_height_cm=100 * height,
                             head_pitch_error_deg=abs(math.degrees(pitch_angle - self.command[3])),
                             head_roll_deg=abs(math.degrees(roll)))
        if self.cfg.task == "navigation":
            self.info["goal_distance_m"] = float(np.linalg.norm(self.goal - self.com[:2]))
        self.info.update({f"reward_{k}": v for k, v in terms.items()})
        return float(sum(terms.values()) * CONTROL_DT)
