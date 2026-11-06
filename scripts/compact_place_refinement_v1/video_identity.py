"""Decoded-frame identity check for the unchanged, previously inspected prefix."""
import argparse,json
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('candidate',type=Path);p.add_argument('--frames',type=int,default=1675);a=p.parse_args()
r0=imageio.get_reader(ROOT/'logs/compact_smooth_integration_v2/complete_execution/rollout.mp4');r1=imageio.get_reader(a.candidate);rows=[]
for i,(x,y) in enumerate(zip(r0,r1)):
 if i>=a.frames:break
 delta=abs(x.astype(np.int16)-y.astype(np.int16));rows.append([int(delta.max()),float(delta.mean())])
r0.close();r1.close();rows=np.array(rows);out=dict(baseline='logs/compact_smooth_integration_v2/complete_execution/rollout.mp4',candidate=str(a.candidate),frames_checked=len(rows),last_checked_frame=len(rows)-1,identical_frame_count=int((rows[:,0]==0).sum()),maximum_decoded_pixel_error=float(rows[:,0].max()),maximum_frame_mean_absolute_error=float(rows[:,1].max()),nonidentical_frames=np.flatnonzero(rows[:,0]>0).tolist(),method='All decoded RGB pixels compared at the same frame index, without cropping, filtering, rescaling or timing changes. Physical source controls/qpos/qvel are separately compared exactly.')
a.candidate.with_name('video_prefix_identity.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
