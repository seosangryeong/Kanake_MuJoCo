"""Find the most trained checkpoint of every run for the play UI's policy list."""

from __future__ import annotations

import json
import re
from pathlib import Path

RL_ROOT = Path(__file__).resolve().parents[1]
CATALOG = RL_ROOT / "config" / "policy_catalog.json"


def discover(run_root=RL_ROOT / "runs"):
    catalog = json.loads(CATALOG.read_text(encoding="utf-8")) if CATALOG.exists() else {}
    hidden = set(catalog.get("_exclude", ()))
    policies = []
    for run in sorted(Path(run_root).glob("*")):
        if not run.is_dir() or run.name in hidden:
            continue
        models = sorted(run.glob("model_*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
        if not models:
            continue
        iteration = int(re.findall(r"\d+", models[-1].stem)[-1])
        # Periodic saves land on multiples of 50; a finished run also saves its last iteration (…99).
        finished = (iteration + 1) % 50 == 0
        policies.append({
            "key": run.name, "checkpoint": models[-1], "iteration": iteration, "finished": finished,
            "description": catalog.get(run.name, ""),
        })
    return policies


def label_of(checkpoint):
    path = Path(checkpoint)
    return f"{path.parent.name}/{path.stem}"
