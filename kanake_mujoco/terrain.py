from __future__ import annotations

import xml.etree.ElementTree as ET
import math
import numpy as np


PARK_ZONES = (
    (1, "Original ramps", 6., 0.), (2, "Steeper hill", 18., 0.),
    (3, "Narrow bridge", 30., 0.), (4, "Left bank", 6., 8.),
    (5, "Right bank", 18., 8.), (6, "Compound slope", 30., 8.),
    (7, "Speed bumps", 6., -8.), (8, "Stairs", 18., -8.),
    (9, "Alternating steps", 30., -8.), (10, "Uneven ground", 6., 16.),
    (11, "Waves", 18., 16.), (12, "One-sided bumps", 30., 16.),
    (13, "Diagonal ribs", 6., -16.), (14, "Rounded rocks", 18., -16.),
    (15, "Shallow trough", 30., -16.),
)


def _solid(asset, worldbody, name, vertices, faces, color):
    ET.SubElement(asset, "mesh", {
        "name": name + "_mesh", "vertex": " ".join(str(v) for point in vertices for v in point),
        "face": faces,
    })
    ET.SubElement(worldbody, "geom", {
        "name": name, "type": "mesh", "mesh": name + "_mesh", "rgba": color,
        "friction": "1.5 0.02 0.01", "contype": "1", "conaffinity": "1",
    })


def _strip(asset, worldbody, prefix, x0, y, length, width, height, count=24, color="0.55 0.61 0.65 1"):
    faces = "0 2 1 0 3 2 4 5 6 4 6 7 0 1 5 0 5 4 3 7 6 3 6 2 0 4 7 0 7 3 1 2 6 1 6 5"
    for i in range(count):
        a, b = length*i/count, length*(i+1)/count
        za, zb = float(height(a)), float(height(b))
        if max(za, zb) <= 1e-8:
            continue
        vertices = [(x0+a,y-width/2,-.05),(x0+b,y-width/2,-.05),(x0+b,y+width/2,-.05),(x0+a,y+width/2,-.05),
                    (x0+a,y-width/2,za),(x0+b,y-width/2,zb),
                    (x0+b,y+width/2,zb),(x0+a,y+width/2,za)]
        _solid(asset, worldbody, f"{prefix}_{i}", vertices, faces, color)


def _height_grid(asset, worldbody, prefix, x0, y, heights, length=8., width=4., color="0.57 0.64 0.57 1"):
    nx, ny = heights.shape[0]-1, heights.shape[1]-1
    faces = "0 2 1 3 4 5 0 1 4 0 4 3 1 2 5 1 5 4 2 0 3 2 3 5"
    for i in range(nx):
        for j in range(ny):
            quad = [(x0+length*a/nx, y-width/2+width*b/ny, float(heights[a,b])) for a,b in ((i,j),(i+1,j),(i+1,j+1),(i,j+1))]
            for k, ids in enumerate(((0,1,2),(0,2,3))):
                top = [quad[n] for n in ids]
                if max(point[2] for point in top) <= 1e-8:
                    continue
                bottom = [(x,y,-.05) for x,y,z in top]
                _solid(asset, worldbody, f"{prefix}_{i}_{j}_{k}", bottom+top, faces, color)


def _box(worldbody, name, x, y, sx, sy, height, yaw=0., color="0.59 0.60 0.62 1"):
    ET.SubElement(worldbody, "geom", {
        "name": name, "type": "box", "pos": f"{x} {y} {(height-.05)/2}",
        "size": f"{sx/2} {sy/2} {(height+.05)/2}",
        "quat": f"{math.cos(yaw/2)} 0 0 {math.sin(yaw/2)}",
        "rgba": color, "friction": "1.5 0.02 0.01",
    })


def add_terrain_park(root, worldbody):
    add_ramp_course(root, worldbody)
    asset = root.find("asset")
    worldbody.find("geom[@name='ground']").set("size", "45 30 0.1")
    _strip(asset,worldbody,"park_z02_hill",14.,0.,8.,4.,lambda x: .65*min(x/2.5,1.,(8-x)/2.5),32)
    _strip(asset,worldbody,"park_z03_bridge",26.,0.,8.,1.6,lambda x: .25*min(x/2.,1.,(8-x)/2.),24,color="0.62 0.64 0.65 1")
    for zone,x0,sign in ((4,2.,1.),(5,14.,-1.)):
        heights = np.zeros((13,3))
        for i in range(13):
            approach = min(i/3.,1.,(12-i)/3.)
            for j in range(3):
                heights[i,j] = approach * (.2 + sign * (j-1)*.2)
        _height_grid(asset,worldbody,f"park_z{zone:02d}_bank",x0,8.,heights)
    heights = np.zeros((13,5))
    for i in range(13):
        approach = min(i/3.,1.,(12-i)/3.)
        for j in range(5):
            heights[i,j] = approach * (.15 + .25*i/12 + .15*j/4)
    _height_grid(asset,worldbody,"park_z06_compound",26.,8.,heights,color="0.55 0.64 0.66 1")
    def bumps(x):
        return sum(h * math.cos(math.pi*(x-c)/1.4)**2 for c,h in ((1.5,.06),(4.,.09),(6.5,.12)) if abs(x-c)<.7)
    _strip(asset,worldbody,"park_z07_bumps",2.,-8.,8.,4.,bumps,64)
    for i,h in enumerate((.04,.08,.12,.16,.16,.12,.08,.04)):
        _box(worldbody,f"park_z08_step_{i}",14.5+i,-8.,1.,4.,h)
    for i in range(8):
        for side in (-1,1):
            height = .10 if (i%2==0)==(side==1) else .03
            _box(worldbody,f"park_z09_step_{i}_{side}",26.5+i,-8.+side,1.,2.,height)
    rng = np.random.default_rng(42)
    heights = rng.uniform(.02,.12,(11,7))
    heights[0,:]=heights[-1,:]=0.
    heights[:,0]=heights[:,-1]=0.
    _height_grid(asset,worldbody,"park_z10_rough",2.,16.,heights)
    _strip(asset,worldbody,"park_z11_waves",14.,16.,8.,4.,lambda x:.14*math.sin(math.pi*x/2.)**2,48)
    for i,h in enumerate((.04,.08,.12)):
        _strip(asset,worldbody,f"park_z12_single_{i}",26.5+2.3*i,15.5,1.8,.65,lambda x,h=h:h*math.sin(math.pi*x/1.8)**2,12)
    for i in range(5):
        _box(worldbody,f"park_z13_rib_{i}",3.3+1.35*i,-16.,3.2,.22,.06,math.pi/4)
    for i in range(24):
        x,y = rng.uniform(14.6,21.4),rng.uniform(-17.5,-14.5)
        size = rng.uniform(.06,.10)
        ET.SubElement(worldbody,"geom",{
            "name":f"park_z14_rock_{i}","type":"ellipsoid","pos":f"{x} {y} .015",
            "size":f"{size*2} {size*1.7} {size}","rgba":"0.63 0.64 0.65 1","friction":"1.5 .02 .01",
        })
    heights_x = (0.,.15,.15,.05,.05,.15,.15,0.)
    locations = (0.,1.,2.,3.,5.,6.,7.,8.)
    _strip(asset,worldbody,"park_z15_trough",26.,-16.,8.,4.,lambda x:float(np.interp(x,locations,heights_x)),32)
    # Visual entrance posts sit outside the driving lanes and have no collision.
    for zone,name,x,y in PARK_ZONES:
        ET.SubElement(worldbody,"geom",{
            "name":f"park_z{zone:02d}_entrance","type":"cylinder","pos":f"{x-4.3} {y-2.6} .3",
            "size":".08 .3","rgba":"0.8 0.82 0.84 1","contype":"0","conaffinity":"0",
        })


def add_ramp_course(root: ET.Element, worldbody: ET.Element) -> None:
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(worldbody, "light", {
        "name": "terrain_light", "pos": "5 -4 8", "dir": "0 0 -1",
        "directional": "true", "diffuse": "0.5 0.5 0.5", "ambient": "0.12 0.12 0.12",
    })
    width = 4.0
    bottom = -0.05
    faces = "0 2 1 0 3 2 4 5 6 4 6 7 0 1 5 0 5 4 3 7 6 3 6 2 0 4 7 0 7 3 1 2 6 1 6 5"
    for name, x0, x1, z0, z1, color in (
        ("uphill", 2.0, 5.0, 0.0, 0.4, "0.48 0.58 0.65 1"),
        ("plateau", 5.0, 7.0, 0.4, 0.4, "0.60 0.61 0.62 1"),
        ("downhill", 7.0, 10.0, 0.4, 0.0, "0.48 0.63 0.54 1"),
    ):
        # Solid convex segments meet exactly at the road surface and extend below the floor.
        vertices = (
            (x0, -width / 2, bottom), (x1, -width / 2, bottom),
            (x1, width / 2, bottom), (x0, width / 2, bottom),
            (x0, -width / 2, z0), (x1, -width / 2, z1),
            (x1, width / 2, z1), (x0, width / 2, z0),
        )
        mesh_name = f"terrain_{name}_mesh"
        ET.SubElement(asset, "mesh", {
            "name": mesh_name,
            "vertex": " ".join(str(v) for vertex in vertices for v in vertex),
            "face": faces,
        })
        ET.SubElement(worldbody, "geom", {
            "name": f"terrain_{name}", "type": "mesh", "mesh": mesh_name,
            "rgba": color, "friction": "1.5 0.02 0.01",
            "contype": "1", "conaffinity": "1",
        })
