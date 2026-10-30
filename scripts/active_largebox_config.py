"""Shared paths for the accepted active clamp, without failed-search code."""
from pathlib import Path
import mujoco
from active_largebox_artifacts import BUNDLE, materialize
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'logs/active_largebox'
INPUTS = BUNDLE / 'generation_inputs'
SOURCE = INPUTS / 'original_largebox_yaw10_vertical_rail_pickup_expert_v1.npz'

def reference_model():
    return materialize(INPUTS / 'original_largebox_vertical_rail_model_v1.mjb')

def name(model, kind, index):
    return mujoco.mj_id2name(model, kind, index).split('/')[-1]
