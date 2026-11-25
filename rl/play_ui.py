"""Play Kanake checkpoints with a control window, like Teraffe's play_loco UI.

The control window picks the terrain kind and curriculum level (Teraffe's seven heightfield kinds,
level 4 = Teraffe's final difficulty), switches between keyboard and fixed-command driving, sets
speeds and loads other checkpoints. The scene always uses the random-terrain heightfield with
force-based anisotropic friction, so terrain changes happen inside the open viewer. Loading a
checkpoint reopens the viewer because policies differ in observations and gains.

    rl/.venv/bin/python rl/play_ui.py
    rl/.venv/bin/python rl/play_ui.py --checkpoint rl/runs/serp_v7_terrain/model_5999.pt --kind stairs --level 2
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import replace

import _bootstrap  # Initialize source-tree imports.

import numpy as np
from mujoco import viewer

from kanake_mujoco import procedural
from kanake_mujoco.config import ACTION_SIZE, CONTROL_DT, EnvConfig
from kanake_mujoco.env import KanakeEnv, wrap_angle
from kanake_rl.panel import ControlPanel
from kanake_rl.policies import discover, label_of
from play import draw_command, policy_for_checkpoint
from teleop import TURN_LIMIT, Operator, ViewerKeyboard


def load(checkpoint):
    cfg = EnvConfig()

    def policy(_obs):
        return np.zeros(ACTION_SIZE)

    if checkpoint:
        cfg, policy = policy_for_checkpoint(checkpoint, "cpu")
    cfg = replace(cfg, terrain="random", park_zone=None, lateral="force", randomize=False, task="locomotion",
                  fixed_command=(0.0, 0.0, 0.0), episode_steps=10**9)
    return cfg, policy


def make_grid(kind, level, rng):
    if kind == "flat":
        return np.zeros((procedural.GRID, procedural.GRID), dtype=np.float32)
    return procedural.generate(rng, kind, procedural.LEVEL_SCALES[level])


def default_checkpoint(policies):
    for key in ("serp_v8_climb", "serp_v7_terrain", "serp_v4_neck"):
        for p in policies:
            if p["key"] == key:
                return str(p["checkpoint"])
    return str(policies[-1]["checkpoint"]) if policies else None


def run(checkpoint, panel, state, keyboard, rng):
    """Play one checkpoint until the viewer closes or another checkpoint is requested."""
    cfg, policy = load(checkpoint)
    panel.limits = {"speed": cfg.speed_scale, "head": cfg.head_yaw_range}
    panel.set_current(checkpoint)
    panel.root.title(f"Kanake 제어판 — {label_of(checkpoint) if checkpoint else '기본 serpenoid (액션 0)'}")
    env = KanakeEnv(cfg)

    def set_friction(axial, lateral):
        env.model.geom_friction[env.capsules, 0] = axial
        env.lateral.extra_mu = max(lateral - axial, 0.0)

    set_friction(*state["friction"])
    operator = Operator(state["k_speed"] or cfg.speed_scale, cfg.head_yaw_range, state["k_turn"], state["k_ramp"])

    def restart():
        env.terrain_override = (make_grid(state["kind"], state["level"], rng), state["kind"])
        env.terrain_level = state["level"]
        observation = env.reset(neutral=True)
        return observation, env.body_heading, env.com[:2].copy()

    obs, travel_yaw, origin = restart()
    auto_yaw = travel_yaw
    refresh, next_status = True, 0.0
    with viewer.launch_passive(env.model, env.data) as handle:
        with handle.lock():
            handle.cam.distance, handle.cam.azimuth, handle.cam.elevation = 2.0, 135, -35
            handle.opt.geomgroup[3] = int(state["view"] != "off")
            handle.opt.geomgroup[1] = int(state["view"] != "only")
        while handle.is_running() and not panel.closed:
            start = time.monotonic()
            if refresh:  # Upload a new heightfield outside the viewer lock.
                handle.update_hfield(0)
                refresh = False
            panel.pump()
            for request in panel.take_requests():
                name = request[0]
                if name == "terrain":
                    state["kind"], state["level"] = request[1], request[2]
                    obs, travel_yaw, origin = restart()
                    auto_yaw, refresh = travel_yaw, True
                    operator.speed = operator.head = 0.0
                    panel.message.set(f"지형: {state['kind']} 레벨 {state['level']} — 처음 위치에서 다시 시작")
                elif name == "mode":
                    state["mode"] = request[1]
                    auto_yaw = env.body_heading + math.radians(state["a_dir"])
                elif name == "speeds":
                    (state["k_speed"], state["k_ramp"], state["k_turn"],
                     state["a_speed"], state["a_dir"], state["a_head"]) = request[1:]
                    operator.max_speed = state["k_speed"]
                    operator.accel = state["k_speed"] / state["k_ramp"]
                    operator.turn_rate = state["k_turn"]
                    auto_yaw = env.body_heading + math.radians(state["a_dir"])
                elif name == "stop":
                    operator.speed = 0.0
                    state["a_speed"] = 0.0
                    panel.a_speed.set("0")
                    panel.message.set("급정지: 자동 주행을 다시 하려면 속도를 넣고 '속도·명령 적용'")
                elif name == "view":
                    state["view"] = request[1]
                    with handle.lock():
                        handle.opt.geomgroup[3] = int(state["view"] != "off")
                        handle.opt.geomgroup[1] = int(state["view"] != "only")
                elif name == "friction":
                    state["friction"] = request[1:]
                    set_friction(*state["friction"])
                elif name == "follow":
                    state["follow"] = request[1]
                elif name == "policy":
                    return request[1]
            keys = keyboard.held() if state["mode"] == "keyboard" else set()
            with handle.lock():
                if state["mode"] == "keyboard":
                    travel_yaw, speed, head, _, _ = operator.update(keys, travel_yaw, env.body_heading, CONTROL_DT)
                else:
                    offset = float(np.clip(wrap_angle(auto_yaw - env.body_heading), -TURN_LIMIT, TURN_LIMIT))
                    travel_yaw = wrap_angle(env.body_heading + offset)
                    speed = state["a_speed"]
                    head = float(np.clip(math.radians(state["a_head"]), -cfg.head_yaw_range, cfg.head_yaw_range))
                env.travel_yaw = travel_yaw
                env.command[:3] = (speed * math.cos(travel_yaw), speed * math.sin(travel_yaw), head)
                obs, _, terminated, _, info = env.step(policy(obs))
                if terminated:
                    obs, travel_yaw, origin = restart()
                    auto_yaw, refresh = travel_yaw, True
                    panel.message.set("넘어짐/이탈: 같은 지형에서 다시 시작")
                if state["follow"]:
                    handle.cam.lookat[:] = env.com
                draw_command(handle, env)
            handle.sync()
            if time.monotonic() >= next_status:
                next_status = time.monotonic() + 0.2
                panel.set_status(
                    f"지형      {state['kind']} 레벨 {state['level']} (높이 ×{procedural.LEVEL_SCALES[state['level']]:g})"
                    f", 마찰 축 {state['friction'][0]:g} / 옆 {state['friction'][1]:g}\n"
                    f"명령      {speed:.3f} m/s, 머리 {math.degrees(head):+.0f}°\n"
                    f"실제 속도 {np.linalg.norm(env.filtered_velocity):.3f} m/s\n"
                    f"머리 오차 yaw {info.get('head_yaw_error_deg', 0):.1f}°, 각속도 {info.get('head_spin_dps', 0):.0f}°/s\n"
                    f"파형      {info.get('frequency_hz', 0):.2f} Hz, 진폭 {info.get('amplitude_deg', 0):.0f}°\n"
                    f"이동 거리 {np.linalg.norm(env.com[:2] - origin):.2f} m")
            time.sleep(max(0.0, CONTROL_DT - (time.monotonic() - start)))
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", help="Defaults to the newest serpentine run")
    parser.add_argument("--kind", choices=procedural.KINDS, default="flat")
    parser.add_argument("--level", type=int, choices=range(len(procedural.LEVEL_SCALES)), default=0)
    parser.add_argument("--friction-axial", type=float, default=0.2, help="Ground friction along each link")
    parser.add_argument("--friction-lateral", type=float, default=1.0, help="Ground friction across each link")
    args = parser.parse_args()

    policies = discover()
    checkpoint = args.checkpoint or default_checkpoint(policies)
    state = dict(kind=args.kind, level=args.level, mode="keyboard", view="off", follow=True,
                 k_speed=None, k_ramp=1.0, k_turn=0.6, a_speed=0.0, a_dir=0.0, a_head=0.0,
                 friction=(args.friction_axial, args.friction_lateral))
    cfg, _ = load(checkpoint) if checkpoint else (EnvConfig(), None)
    state["a_speed"] = round(0.66 * cfg.speed_scale, 3)
    panel = ControlPanel(procedural.KINDS, len(procedural.LEVEL_SCALES), state,
                         {"speed": cfg.speed_scale, "head": cfg.head_yaw_range}, policies, checkpoint)
    keyboard = ViewerKeyboard()
    rng = np.random.default_rng()
    try:
        while checkpoint is not None or not panel.closed:
            requested = run(checkpoint, panel, state, keyboard, rng)
            if requested is None:
                break
            checkpoint = requested
    finally:
        keyboard.close()
        if not panel.closed:
            panel.root.destroy()


if __name__ == "__main__":
    main()
