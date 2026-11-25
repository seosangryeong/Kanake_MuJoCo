"""Seeded heightfield terrain bank and curriculum levels.

The terrain kinds, their generators and the final difficulty follow Teraffe's
``procedural_terrain.py`` and ``config/generalization.json`` (flat, wave, stairs, ledge, slope,
rough, bumps; 12–30 cm features, 6–12 cm stairs, 10–20 cm ledges). Real terrain does not shrink
for a 3.8 cm-thick snake, so only the area is smaller. Five curriculum levels scale every height
(0.15 … 1.0) and shrink the share of flat ground (80 % … 5 %), as in Teraffe.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

KINDS = ("flat", "wave", "stairs", "ledge", "slope", "rough", "bumps")
WEIGHTS = (0.05, 0.25, 0.25, 0.20, 0.075, 0.10, 0.075)  # Teraffe's final mix.
HALF_EXTENT = 4.0  # m; 8 m × 8 m holds 2.8 m wavelengths and a 20 s episode.
GRID = 513  # 1.6 cm cells: a 12 cm riser rises at ~83°, close to a real wall to lean on.
HEIGHT_SCALE = 0.50  # m; heightfield elevation range, as in Teraffe.
AMPLITUDE = (0.12, 0.30)
WAVELENGTH = (1.2, 2.8)
STEP_HEIGHT = (0.06, 0.12)
LEDGE_HEIGHT = (0.10, 0.20)
LEVEL_SCALES = (0.15, 0.30, 0.50, 0.75, 1.0)
FLAT_FRACTIONS = (0.80, 0.60, 0.35, 0.15, 0.05)
BANK_PER_LEVEL = 12


def generate(rng, kind, scale):
    axis = np.linspace(-HALF_EXTENT, HALF_EXTENT, GRID)
    x, y = np.meshgrid(axis, axis)
    angle = rng.uniform(-math.pi, math.pi)
    amplitude = scale * rng.uniform(*AMPLITUDE)
    wavelength = rng.uniform(*WAVELENGTH)
    phase = rng.uniform(0, math.tau)
    t = x * math.cos(angle) + y * math.sin(angle)
    if kind == "flat":
        z = np.zeros_like(x)
    elif kind == "wave":
        z = amplitude * (1 + np.sin(math.tau * t / wavelength + phase)) / 2
    elif kind == "stairs":
        step = scale * rng.uniform(*STEP_HEIGHT)
        level = np.floor(np.mod(t + phase, wavelength * 2) / (wavelength / 4)).astype(int)
        z = step * np.minimum(level, 7 - level)
    elif kind == "ledge":
        step = scale * rng.uniform(*LEDGE_HEIGHT)
        z = step * (np.mod(t + phase, wavelength * 2) >= wavelength)
    elif kind == "slope":
        z = amplitude * (1 - np.abs(2 * np.mod(t / wavelength + phase, 1) - 1))
    elif kind == "rough":
        coarse = rng.uniform(0, amplitude, (9, 9))
        knots = np.linspace(-HALF_EXTENT, HALF_EXTENT, 9)
        rows = np.array([np.interp(axis, knots, row) for row in coarse])
        z = np.array([np.interp(axis, knots, rows[:, j]) for j in range(GRID)]).T
    else:  # bumps
        z = amplitude * np.exp(-((np.mod(t + phase, wavelength) - wavelength / 2) / (wavelength * 0.15)) ** 2)
    return np.clip(z, 0, HEIGHT_SCALE).astype(np.float32)


@lru_cache(maxsize=4)
def bank(seed):
    """BANK_PER_LEVEL non-flat terrains per level; identical in every worker for one seed."""
    rng = np.random.default_rng(seed)
    kinds = [k for k in KINDS if k != "flat"]
    weights = np.array([w for k, w in zip(KINDS, WEIGHTS) if k != "flat"])
    counts = np.floor(BANK_PER_LEVEL * weights / weights.sum()).astype(int)
    counts[np.argsort(-(BANK_PER_LEVEL * weights / weights.sum() - counts))[:BANK_PER_LEVEL - counts.sum()]] += 1
    levels = []
    for scale in LEVEL_SCALES:
        grids, names = [], []
        for kind, count in zip(kinds, counts):
            for _ in range(count):
                grids.append(generate(rng, kind, scale))
                names.append(kind)
        levels.append((np.stack(grids), tuple(names)))
    return tuple(levels)
