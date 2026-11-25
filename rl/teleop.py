"""Drive a Kanake checkpoint by keyboard (MuJoCo viewer) or a ROS 2 joystick (/joy).

Inputs set a forward speed, turn the travel direction and (for checkpoints trained with
--head-yaw-range) aim the head. They become the world-frame velocity command the policy was
trained on. The travel direction stays within ±1 rad of the body heading seen in training.

Keyboard (hold keys while the viewer window has focus; X11 key polling like Teraffe):
  Up             speed rises linearly to the maximum; release and it falls linearly to zero
  Down           brake twice as fast
  Left / Right   turn the travel direction while held
  Shift+Left/Right   look left/right while held; release and the gaze returns to the front
  Shift+Up/Down      look up/down while held (raised-head checkpoints); release to level
  Ctrl+Up/Down       raise/lower the head height (raised-head checkpoints); it stays where left
  Avoid Backspace: the viewer resets its data behind the policy's back.

Joystick (ros2 run joy joy_node, same axes as Teraffe):
  left stick up    forward speed      left stick left/right   turn
  right stick left/right   head yaw
"""

from __future__ import annotations

import argparse
import math
import os
import threading
import time
from dataclasses import replace

import _bootstrap  # Initialize source-tree imports.

import numpy as np
from mujoco import viewer

from kanake_mujoco.config import ACTION_SIZE, CONTROL_DT, TERRAINS, EnvConfig
from kanake_mujoco.env import KanakeEnv, wrap_angle
from play import draw_command, policy_for_checkpoint

TURN_LIMIT = 1.0


class ViewerKeyboard:
    """Report held keys only while a window of this process has X11 keyboard focus."""

    SYMBOLS = {"up": "Up", "down": "Down", "left": "Left", "right": "Right",
               "shift_l": "Shift_L", "shift_r": "Shift_R", "ctrl_l": "Control_L", "ctrl_r": "Control_R"}

    def __init__(self):
        from Xlib import XK, display

        if not os.environ.get("DISPLAY"):
            raise RuntimeError("Keyboard driving needs an X11/XWayland display")
        self.display = display.Display()
        self.pid_atom = self.display.intern_atom("_NET_WM_PID")
        self.codes = {name: self.display.keysym_to_keycode(XK.string_to_keysym(symbol))
                      for name, symbol in self.SYMBOLS.items()}

    def _focused(self):
        from Xlib import X, error

        try:
            window = self.display.get_input_focus().focus
            for _ in range(8):
                if not hasattr(window, "get_full_property"):
                    return False
                pid = window.get_full_property(self.pid_atom, X.AnyPropertyType)
                if pid is not None and len(pid.value) and int(pid.value[0]) == os.getpid():
                    return True
                parent = window.query_tree().parent
                if not hasattr(parent, "id") or parent.id == window.id:
                    return False
                window = parent
        except error.XError:
            return False
        return False

    def held(self):
        if not self._focused():
            return set()
        bitmap = self.display.query_keymap()
        return {name for name, code in self.codes.items() if code and bitmap[code // 8] & (1 << (code % 8))}

    def close(self):
        self.display.close()


class Operator:
    """Turns held keys or joystick axes into speed, travel direction and head yaw."""

    def __init__(self, max_speed, head_range, turn_rate, ramp_time=1.0, head_rate=0.8,
                 pitch_range=0.0, heights=None):
        self.lock = threading.Lock()
        self.max_speed, self.head_range, self.turn_rate = max_speed, head_range, turn_rate
        self.accel = max_speed / ramp_time
        self.head_rate = head_rate
        self.speed = self.head = self.pitch = 0.0
        self.pitch_range = pitch_range
        self.heights = heights  # (low, high) for raised-head checkpoints, else None.
        self.height = heights[1] if heights else 0.0
        self.axes = None
        self.joy_time = None

    def joy(self, axes):
        with self.lock:
            self.axes = [float(a) for a in axes]
            self.joy_time = time.monotonic()

    def update(self, keys, travel_yaw, body_heading, dt):
        with self.lock:
            joystick = self.joy_time is not None and time.monotonic() - self.joy_time < 0.5
            if joystick and not keys:
                axis = (lambda i: self.axes[i] if len(self.axes) > i else 0.0)
                self.speed = max(0.0, axis(1)) * self.max_speed
                turn = axis(0)
                self.head = axis(3) * self.head_range
            else:
                shift = bool(keys & {"shift_l", "shift_r"})
                ctrl = bool(keys & {"ctrl_l", "ctrl_r"})
                vertical = int("up" in keys) - int("down" in keys)
                pitch_target = vertical * self.pitch_range if shift else 0.0
                step = self.head_rate * dt
                self.pitch = float(np.clip(pitch_target, self.pitch - step, self.pitch + step))
                if ctrl and self.heights:
                    self.height = float(np.clip(self.height + 0.05 * vertical * dt, *self.heights))
                if shift or ctrl:
                    keys = keys - {"up", "down"}
                if "up" in keys:
                    self.speed = min(self.max_speed, self.speed + self.accel * dt)
                else:
                    brake = 2.0 if "down" in keys else 1.0
                    self.speed = max(0.0, self.speed - brake * self.accel * dt)
                side = int("left" in keys) - int("right" in keys)
                turn = 0 if shift else side
                if shift and side:
                    target = side * self.head_range
                else:
                    target = 0.0
                step = self.head_rate * dt
                self.head = float(np.clip(target, self.head - step, self.head + step))
            travel_yaw += turn * self.turn_rate * dt
            offset = float(np.clip(wrap_angle(travel_yaw - body_heading), -TURN_LIMIT, TURN_LIMIT))
            return wrap_angle(body_heading + offset), self.speed, self.head, self.pitch, self.height


def start_joystick(topic, operator):
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Joy

    rclpy.init()
    node = Node("kanake_teleop")
    node.create_subscription(Joy, topic, lambda msg: operator.joy(msg.axes), 10)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    print(f"Listening for sensor_msgs/Joy on {topic}", flush=True)
    return node


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", help="Without one, the zero-action serpenoid runs")
    parser.add_argument("--terrain", choices=TERRAINS)
    parser.add_argument("--joy-topic", help="Also read a ROS 2 Joy topic, e.g. /joy")
    parser.add_argument("--max-speed", type=float, help="Defaults to the checkpoint's speed scale")
    parser.add_argument("--ramp-time", type=float, default=1.0, help="Seconds from zero to maximum speed")
    parser.add_argument("--turn-rate", type=float, default=0.6, help="Turn rate while held (rad/s)")
    parser.add_argument("--terrain-level", type=int, choices=range(5),
                        help="Random terrain: pin the curriculum level (0 easiest … 4 Teraffe's hardest)")
    parser.add_argument("--collision", choices=("off", "only", "both"), default="off",
                        help="Show the capsule collision geoms (group 3) instead of or with the CAD meshes")
    args = parser.parse_args()

    cfg = EnvConfig()

    def policy(_obs):
        return np.zeros(ACTION_SIZE)

    if args.checkpoint:
        cfg, policy = policy_for_checkpoint(args.checkpoint, "cpu")
    overrides = dict(fixed_command=(0.0, 0.0, 0.0), task="locomotion", randomize=False, episode_steps=10**9)
    if args.terrain is not None and args.terrain != cfg.terrain:
        overrides.update(terrain=args.terrain, park_zone=None)
    head_range = cfg.head_yaw_range
    cfg = replace(cfg, **overrides)
    lift = cfg.head_lift
    operator = Operator(args.max_speed or cfg.speed_scale, head_range, args.turn_rate, args.ramp_time,
                        pitch_range=cfg.head_pitch_range if lift else 0.0,
                        heights=(cfg.head_height_min, cfg.head_height_max) if lift else None)
    if head_range == 0:
        print("This checkpoint was trained without head commands; head input is disabled.")
    keyboard = ViewerKeyboard()
    node = start_joystick(args.joy_topic, operator) if args.joy_topic else None

    if cfg.terrain == "random" and cfg.lateral != "force":
        cfg = replace(cfg, lateral="force")
    env = KanakeEnv(cfg)
    env.fixed_level = args.terrain_level if args.terrain_level is not None else (4 if cfg.terrain == "random" else None)
    obs = env.reset(neutral=True)
    travel_yaw = env.body_heading
    origin, next_report = env.com[:2].copy(), time.monotonic() + 1.0
    print(__doc__.split("Keyboard")[1].split("Joystick")[0].strip(), flush=True)
    try:
        with viewer.launch_passive(env.model, env.data) as handle:
            with handle.lock():
                handle.cam.distance, handle.cam.azimuth, handle.cam.elevation = 1.8, 135, -35
                if args.collision != "off":
                    handle.opt.geomgroup[3] = 1
                    handle.opt.geomgroup[1] = int(args.collision == "both")
            refresh = bool(env.model.nhfield)
            while handle.is_running():
                start = time.monotonic()
                if refresh:  # Upload a new heightfield outside the viewer lock.
                    handle.update_hfield(0)
                    print(f"terrain: level {env.terrain_level} {env.terrain_kind}", flush=True)
                    refresh = False
                keys = keyboard.held()
                with handle.lock():
                    travel_yaw, speed, head, pitch, height = operator.update(
                        keys, travel_yaw, env.body_heading, CONTROL_DT)
                    env.travel_yaw = travel_yaw
                    env.command[:3] = (speed * math.cos(travel_yaw), speed * math.sin(travel_yaw), head)
                    if lift:
                        env.command[3:] = (pitch, height)
                    obs, _, terminated, _, _ = env.step(policy(obs))
                    if terminated:
                        obs = env.reset(neutral=True)
                        travel_yaw = env.body_heading
                        origin = env.com[:2].copy()
                        refresh = bool(env.model.nhfield)
                    handle.cam.lookat[:] = env.com
                    draw_command(handle, env)
                if time.monotonic() >= next_report:
                    next_report += 1.0
                    print(f"command {speed:.3f} m/s | actual {np.linalg.norm(env.filtered_velocity):.3f} m/s"
                          f" | head {math.degrees(head):+.0f}°"
                          + (f" pitch {math.degrees(pitch):+.0f}° height {100 * height:.0f}"
                             f"→{100 * env.head_pose()[0]:.0f} cm" if lift else "")
                          + f" | travelled {np.linalg.norm(env.com[:2] - origin):.2f} m",
                          flush=True)
                handle.sync()
                time.sleep(max(0.0, CONTROL_DT - (time.monotonic() - start)))
    finally:
        keyboard.close()
        if node is not None:
            import rclpy
            node.destroy_node()
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
