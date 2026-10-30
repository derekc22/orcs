#!/usr/bin/env python3
"""Compile the original-geometry, light-mass ground largebox pickup variant."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mocke.sonic.env_cfg import sonic_tracking_env_cfg

from orcs.assets.objects import object_spec
from run_engineered_locomomanipulation import PAD_FRICTION, engineering_robot_cfg
from build_engineered_ground_large_model import large_ground_gripper_spec


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/engineering_ground_original_largebox"
MODEL = OUT / "original_largebox_pickup_model_v4.mjb"
MOTION = ROOT / "logs/engineering_locomomanipulation/released_policy_motion_v1.npz"

# Exact compiled local AABB of the repository's largebox convex decomposition.
OBJECT_BOUNDS_LOW = np.asarray((-0.23532704, -0.23105561, -0.20018188))
OBJECT_BOUNDS_HIGH = np.asarray((0.23824476, 0.23054674, 0.20937677))
BOX_HALF = 0.5 * (OBJECT_BOUNDS_HIGH - OBJECT_BOUNDS_LOW)
BOX_CENTER = 0.5 * (OBJECT_BOUNDS_HIGH + OBJECT_BOUNDS_LOW)
# Stable terrain pose measured from the actual collision decomposition.  The
# irregular reconstructed underside rests with a small roll/pitch, so its free
# joint is lower than the raw body-frame minimum-vertex bound.
GROUND_ROOT_Z = 0.17197975
BOX_MASS = 0.05
SOURCE_MASS = 4.5
SOURCE_INERTIA = np.asarray((0.08778145, 0.08499605, 0.07830570))
SOURCE_IPOS = np.asarray((-0.00000570063, 0.000329085, 0.00497456))


def original_largebox_spec() -> mujoco.MjSpec:
    spec = object_spec("largebox", "cvx_dcmp")
    body = list(spec.worldbody.bodies)[0]
    body.name = "object"
    body.joints[0].name = "floating_base_joint"

    # Preserve the actual mesh geometry while making mass/inertia forgiving.
    body.mass = BOX_MASS
    body.fullinertia[:] = (np.nan, 0.0, 0.0, 0.0, 0.0, 0.0)
    body.inertia[:] = SOURCE_INERTIA * (BOX_MASS / SOURCE_MASS)
    body.ipos[:] = SOURCE_IPOS
    body.iquat[:] = (1.0, 0.0, 0.0, 0.0)

    for geom in body.geoms:
        if geom.contype == 0:
            continue
        geom.friction[:] = PAD_FRICTION
        geom.condim = 4
        geom.margin = 0.0
        geom.solref[:] = (0.03, 1.0)
        geom.solimp[:] = (0.8, 0.95, 0.01, 0.5, 2.0)
        geom.contype = 3
        geom.conaffinity = 21
    return spec


def build_model() -> mujoco.MjModel:
    cfg = sonic_tracking_env_cfg(str(MOTION), play=True, il_ordered=False)
    cfg.scene.num_envs = 1
    robot = engineering_robot_cfg()
    robot.spec_fn = large_ground_gripper_spec
    cfg.scene.entities["robot"] = robot
    cfg.scene.entities["object"] = EntityCfg(
        spec_fn=original_largebox_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, GROUND_ROOT_Z), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )
    cfg.sim.nconmax = 256
    cfg.sim.njmax = 900
    cfg.auto_reset = False
    cfg.terminations = {}
    cfg.events = {name: term for name, term in cfg.events.items() if term.mode != "startup"}
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    OUT.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(env.sim.mj_model, str(MODEL), None)
    env.close()
    return mujoco.MjModel.from_binary_path(str(MODEL))


if __name__ == "__main__":
    build_model()
    print(MODEL)
