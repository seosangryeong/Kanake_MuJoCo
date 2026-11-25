"""Compile the vendored robot URDF into a free-base MuJoCo scene."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .config import BODY_NAMES, CONTROL_DT, JOINT_NAMES, PHYSICS_SUBSTEPS, TERRAINS
from .procedural import GRID, HALF_EXTENT, HEIGHT_SCALE
from .terrain import add_ramp_course, add_terrain_park

ASSET_DIR = Path(__file__).resolve().parent / "assets"
URDF = ASSET_DIR / "urdf" / "kanake6_sim.urdf"
ROBOT_GROUP = 1
COLLISION_GROUP = 3
GROUND_GROUPS = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)


@dataclass(frozen=True)
class PhysicsConfig:
    """Contact and actuator settings.

    ``collision="capsule"`` replaces the CAD convex hulls with one capsule per link that fits the
    round shell (r = 19 mm). MuJoCo aligns the first tangent of a capsule–plane contact with the
    capsule axis, so ``friction_axial`` acts along the body and ``friction_lateral`` across it.
    Serpentine propulsion needs lateral > axial, like snake belly scales or passive wheels.

    ``lateral="pair"`` expresses this with contact pairs, which only works on planes.
    ``lateral="force"`` gives every contact the axial coefficient and adds the sideways part as an
    applied force (see friction.py); it works on heightfields, boxes and meshes too.
    """

    collision: str = "capsule"
    friction_axial: float = 0.3
    friction_lateral: float = 1.0
    kp: float = 3.0
    kv: float = 1.0
    lateral: str = "pair"

    def __post_init__(self):
        if self.collision not in ("capsule", "mesh"):
            raise ValueError("Collision must be 'capsule' or 'mesh'")
        if self.lateral not in ("pair", "force"):
            raise ValueError("Lateral friction must be 'pair' or 'force'")
        if self.lateral == "force" and self.collision != "capsule":
            raise ValueError("Force-based lateral friction needs capsule collision")
        if min(self.friction_axial, self.friction_lateral, self.kp) <= 0 or self.kv < 0:
            raise ValueError("Friction and gains must be positive")


def _robot_xml():
    urdf = ET.parse(URDF).getroot()
    ET.SubElement(ET.SubElement(urdf, "mujoco"), "compiler",
                  discardvisual="true", fusestatic="false", strippath="false")
    for link in urdf.findall("link"):
        for index, collision in enumerate(link.findall("collision")):
            collision.set("name", f"{link.attrib['name']}_collision_{index}")
    for mesh in urdf.findall(".//mesh"):
        filename = Path(mesh.attrib["filename"]).name
        mesh.set("filename", str(ASSET_DIR / "meshes" / filename))
    # One CAD-exported cover inertial origin contains a split scientific literal.
    for origin in urdf.findall(".//inertial/origin"):
        xyz = " ".join(origin.attrib["xyz"].split())
        origin.set("xyz", xyz.replace("9.56 57E-07", "9.5657E-07"))
    for joint in urdf.findall("joint"):
        if joint.attrib["name"] not in JOINT_NAMES:
            joint.set("type", "fixed")
            continue
        # The reference Isaac USD sets all driven hinges to ±85 degrees.
        limit = joint.find("limit")
        limit.set("lower", str(-math.radians(85)))
        limit.set("upper", str(math.radians(85)))
    spec = mujoco.MjSpec.from_string(ET.tostring(urdf, encoding="unicode"))
    model = spec.compile()
    return ET.fromstring(spec.to_xml()), model


def _link_capsules(model):
    """Fit one capsule per driven link around its shell mesh and cover, in the link frame."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    capsules = {}
    for name in BODY_NAMES:
        body = model.body(name).id
        points = []
        for geom in range(model.ngeom):
            owner = model.body(model.geom_bodyid[geom]).name
            if owner not in (name, f"{name}_cover"):
                continue
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh = model.geom_dataid[geom]
            start = model.mesh_vertadr[mesh]
            vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
            world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            points.append((world - data.xpos[body]) @ data.xmat[body].reshape(3, 3))
        local = np.concatenate(points)
        center_y, center_z = (local[:, 1:].max(axis=0) + local[:, 1:].min(axis=0)) / 2
        radius = float(np.hypot(local[:, 1] - center_y, local[:, 2] - center_z).max())
        x0, x1 = local[:, 0].min() + radius, local[:, 0].max() - radius
        capsules[name] = (radius, (x0, center_y, center_z), (max(x1, x0 + 1e-3), center_y, center_z))
    return capsules


def build_model(terrain: str = "flat", physics: PhysicsConfig | None = None) -> mujoco.MjModel:
    if terrain not in TERRAINS:
        raise ValueError(f"Unknown terrain {terrain!r}")
    physics = physics or PhysicsConfig()
    root, robot_model = _robot_xml()
    worldbody = root.find("worldbody")
    robot = worldbody.find("body[@name='tail']")
    robot.set("pos", "0 0 0.25")
    ET.SubElement(robot, "freejoint", name="root")
    capsule = physics.collision == "capsule"
    mean_friction = math.sqrt(physics.friction_axial * physics.friction_lateral)
    if physics.lateral == "force":
        mean_friction = physics.friction_axial  # The lateral excess is applied as a force.
    for geom in robot.iter("geom"):
        # CAD meshes stay visible; with capsules they no longer collide.
        geom.attrib.update(group=str(ROBOT_GROUP), contype="0" if capsule else "2",
                           conaffinity="0" if capsule else "1", priority="1",
                           friction=f"{mean_friction} 0.02 0.01", condim="3")
    collision_names = []
    if capsule:
        for name, (radius, start, end) in _link_capsules(robot_model).items():
            body = robot if name == "tail" else robot.find(f".//body[@name='{name}']")
            geom_name = f"{name}_capsule"
            # Explicit inertials from the URDF keep the CAD mass properties.
            ET.SubElement(body, "geom", name=geom_name, type="capsule", size=f"{radius:.5f}",
                          fromto=" ".join(f"{v:.5f}" for v in (*start, *end)),
                          group=str(COLLISION_GROUP), contype="2", conaffinity="1", priority="1",
                          friction=f"{mean_friction} 0.02 0.01", condim="3", rgba="0.9 0.5 0.2 0.4")
            collision_names.append(geom_name)
    ET.SubElement(root, "option", timestep=str(CONTROL_DT / PHYSICS_SUBSTEPS),
                  integrator="implicitfast", cone="elliptic")
    ET.SubElement(worldbody, "light", pos="0 -3 5", dir="0 0 -1", directional="true")
    # A 10 cm checker makes motion visible under a following camera.
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker", width="512", height="512",
                  rgb1="0.68 0.70 0.72", rgb2="0.52 0.55 0.58", mark="edge", markrgb="0.4 0.42 0.45")
    ET.SubElement(asset, "material", name="grid", texture="grid", texrepeat="10 10", texuniform="true")
    ET.SubElement(worldbody, "geom", name="ground", type="plane", material="grid",
                  size="20 20 0.1", friction="1.5 0.02 0.01")
    if terrain == "random":
        # One heightfield whose data each world overwrites from the procedural bank on reset.
        ground = worldbody.find("geom[@name='ground']")
        ground.attrib.update(contype="0", conaffinity="0", rgba="0 0 0 0")
        ground.attrib.pop("material")
        ET.SubElement(asset, "hfield", name="terrain", nrow=str(GRID), ncol=str(GRID),
                      size=f"{HALF_EXTENT} {HALF_EXTENT} {HEIGHT_SCALE} 0.1")
        ET.SubElement(worldbody, "geom", name="terrain", type="hfield", hfield="terrain",
                      material="grid", friction="1.5 0.02 0.01")
    elif terrain == "ramps":
        add_ramp_course(root, worldbody)
    elif terrain == "park":
        add_terrain_park(root, worldbody)
    # Visual park entrance posts must also be excluded from ground height rays.
    for geom in worldbody.findall("geom"):
        if geom.get("contype") == "0":
            geom.set("group", "2")
    if capsule and physics.lateral == "pair":
        # Explicit pairs override dynamic contacts. The park keeps isotropic contacts on its
        # many small obstacle geoms and is anisotropic only on the floor.
        surfaces = ["ground"]
        if terrain == "ramps":
            surfaces += [g.get("name") for g in worldbody.findall("geom")
                         if g.get("name", "").startswith("terrain_")]
        contact = ET.SubElement(root, "contact")
        friction = f"{physics.friction_axial} {physics.friction_lateral} 0.005 0.0001 0.0001"
        for surface in surfaces:
            for geom_name in collision_names:
                ET.SubElement(contact, "pair", geom1=geom_name, geom2=surface,
                              condim="3", friction=friction)
    actuators = ET.SubElement(root, "actuator")
    for name in JOINT_NAMES:
        joint = robot.find(f".//joint[@name='{name}']")
        ET.SubElement(actuators, "position", name=name, joint=name, kp=str(physics.kp),
                      kv=str(physics.kv), ctrlrange=joint.attrib["range"])
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def surface_height(model, data, xy):
    point = np.array([xy[0], xy[1], 5.0])
    distance = mujoco.mj_ray(model, data, point, np.array([0.0, 0.0, -1.0]),
                             GROUND_GROUPS, True, -1, np.empty(1, dtype=np.int32))
    return point[2] - distance if distance >= 0 else 0.0
