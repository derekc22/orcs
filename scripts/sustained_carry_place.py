"""Use the proven controlled placement on the unchanged grasp after sustained walk."""
import argparse
import numpy as np
from sustained_carry_motion import OUT
import compact_gripper_place_v5 as place

def run(tag,source,trim_seconds=None):
 place.OUT=OUT
 if trim_seconds is not None:
  a=dict(np.load(OUT/(source+'.npz')));n=int(round(trim_seconds/.005))
  for key in ['ctrl_physics','qpos','qvel','contacts','carry_diagnostics']:
   if key in a and len(a[key])>n:a[key]=a[key][:n]
  short=source+'_stopped';np.savez_compressed(OUT/(short+'.npz'),**a);(OUT/(short+'_model.mjb')).symlink_to((OUT/(source+'_model.mjb')).name);source=short
 place.run(tag,source)
 a=dict(np.load(OUT/(tag+'.npz')));src=np.load(OUT/(source+'.npz'))
 for key in ['walking_start_physics_step','walking_end_physics_step']:
  if key in src:a[key]=src[key]
 np.savez_compressed(OUT/(tag+'.npz'),**a)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',default='walk1');p.add_argument('--tag',default='complete1');p.add_argument('--trim-seconds',type=float);a=p.parse_args();run(a.tag,a.source,a.trim_seconds)
