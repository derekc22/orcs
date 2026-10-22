#!/usr/bin/env python3
"""Compile the separate full-size, light ground-box engineering variant."""

from __future__ import annotations
from pathlib import Path
import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mocke.sonic.env_cfg import sonic_tracking_env_cfg
from run_engineered_locomomanipulation import PAD_FRICTION, GUARD_DENSITY, engineering_gripper_spec, engineering_robot_cfg
from orcs.assets.g1 import find_body

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"logs/engineering_ground_large";MODEL=OUT/"engineered_ground_large_model.mjb";MOTION=ROOT/"logs/engineering_locomomanipulation/released_policy_motion_v1.npz"
# Original large-box height with the square leg-clearance footprint.
BOX_HALF=(.15,.15,.3048);BOX_MASS=.05

def large_ground_gripper_spec():
 spec=engineering_gripper_spec()
 # The tall ground box is deliberately contacted farther from the torso than
 # the small elevated payload.  Extend the same compliant front cradle sphere
 # rather than forcing both heavy arms into the destabilizing fully-extended
 # posture; no new contact mechanism or constraint is introduced.
 for side in ("left","right"):
  body=find_body(spec.worldbody,f"{side}_wrist_yaw_link")
  for geom in body.geoms:
   if geom.name==f"{side}_hand_front_guard_collision":geom.pos[0]+=0.15
 return spec

def large_object_spec():
 spec=mujoco.MjSpec();body=spec.worldbody.add_body(name="object");body.add_freejoint(name="floating_base_joint");body.add_geom(name="object_collision",type=mujoco.mjtGeom.mjGEOM_BOX,size=BOX_HALF,mass=BOX_MASS,friction=PAD_FRICTION,condim=4,margin=0.,solref=(.03,1.),solimp=(.8,.95,.01,.5,2.),contype=3,conaffinity=21,rgba=(.62,.43,.24,1.));return spec
def build_model():
 cfg=sonic_tracking_env_cfg(str(MOTION),play=True,il_ordered=False);cfg.scene.num_envs=1;robot=engineering_robot_cfg();robot.spec_fn=large_ground_gripper_spec;cfg.scene.entities["robot"]=robot;cfg.scene.entities["object"]=EntityCfg(spec_fn=large_object_spec,init_state=EntityCfg.InitialStateCfg(pos=(0.,0.,BOX_HALF[2]+.001),rot=(1.,0.,0.,0.)));cfg.sim.nconmax=192;cfg.sim.njmax=700;cfg.auto_reset=False;cfg.terminations={};cfg.events={n:t for n,t in cfg.events.items() if t.mode!="startup"};env=ManagerBasedRlEnv(cfg=cfg,device="cuda:0");OUT.mkdir(parents=True,exist_ok=True);mujoco.mj_saveModel(env.sim.mj_model,str(MODEL),None);env.close();return mujoco.MjModel.from_binary_path(str(MODEL))
if __name__=="__main__":build_model();print(MODEL)
