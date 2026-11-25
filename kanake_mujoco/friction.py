"""Anisotropic ground friction for capsule links on any surface.

MuJoCo aligns the contact tangent with the capsule axis only for plane contacts; on boxes, meshes
and heightfields the tangent frame is fixed in the world, so a contact pair cannot express
"slippery along the link, grippy across it". Instead every robot contact uses the axial friction
coefficient, and after each physics step this adds the missing sideways Coulomb friction

    F = -sign(v_side) * min((mu_lateral - mu_axial) * N, STOP_GAIN * m * |v_side| / dt) * side

at each contact point, where ``side`` is perpendicular to the link axis in the contact plane and
N is the solved normal force. The force acts during the next step (2.5 ms later). The second term
keeps the explicit force from reversing the slip within one step, which would otherwise make it
unstable near zero slip or during contact impacts.
"""

from __future__ import annotations

import mujoco
import numpy as np

STOP_GAIN = 0.6  # Share of the link momentum the force may remove per step; two contacts per link stay below the explicit limit of 2.


class LateralFriction:
    def __init__(self, model, robot_geoms, extra_mu):
        self.model = model
        self.is_robot = np.zeros(model.ngeom, dtype=bool)
        self.is_robot[robot_geoms] = True
        self.extra_mu = float(extra_mu)
        self.robot_bodies = np.unique(model.geom_bodyid[robot_geoms])
        self.dt = model.opt.timestep

    def apply(self, data):
        data.xfrc_applied[self.robot_bodies] = 0.0
        n = data.ncon
        if n == 0 or self.extra_mu <= 0:
            return
        contact = data.contact
        geom1, geom2 = contact.geom1[:n], contact.geom2[:n]
        robot1, robot2 = self.is_robot[geom1], self.is_robot[geom2]
        use = (robot1 ^ robot2) & (contact.efc_address[:n] >= 0)
        if not use.any():
            return
        geom = np.where(robot1, geom1, geom2)[use]
        frame = contact.frame[:n][use]
        normal = frame[:, :3]
        point = contact.pos[:n][use]
        force_n = np.abs(data.efc_force[contact.efc_address[:n][use]])
        axis = data.geom_xmat[geom][:, [2, 5, 8]]  # Capsule axis is the geom z axis.
        side = np.cross(normal, axis)
        length = np.linalg.norm(side, axis=1, keepdims=True)
        valid = length[:, 0] > 1e-6
        side = side / np.maximum(length, 1e-9)
        body = self.model.geom_bodyid[geom]
        # cvel is the body's spatial velocity about its tree's centre of mass.
        centre = data.subtree_com[self.model.body_rootid[body]]
        velocity = data.cvel[body]
        point_velocity = velocity[:, 3:] + np.cross(velocity[:, :3], point - centre)
        slip = np.sum(point_velocity * side, axis=1)
        limit = STOP_GAIN * self.model.body_mass[body] * np.abs(slip) / self.dt
        magnitude = np.minimum(self.extra_mu * force_n, limit)
        force = (-np.sign(slip) * magnitude * valid)[:, None] * side
        torque = np.cross(point - data.xipos[body], force)
        np.add.at(data.xfrc_applied[:, :3], body, force)
        np.add.at(data.xfrc_applied[:, 3:], body, torque)

    def step(self, model, data, substeps):
        for _ in range(substeps):
            self.apply(data)
            mujoco.mj_step(model, data)
