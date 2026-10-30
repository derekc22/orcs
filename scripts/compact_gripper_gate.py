"""Cheap whole-trajectory arm reach gate. No simulated task claims."""
import json
from pathlib import Path
import mujoco,numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'logs/compact_gripper_v1';BASE=ROOT/'logs/active_largebox/accepted_ground_to_ground_v1'
def translated_model(shorten):
 m=mujoco.MjModel.from_binary_path(str(BASE/'expert_model.mjb'));a=np.load(BASE/'expert.npz');d=mujoco.MjData(m);d.qpos[:]=a['initial_qpos'];mujoco.mj_forward(m,d)
 b=m.body('clamp_frame').id;pads=[m.geom(f'clamp_pad{i}').id for i in range(6)];v=np.mean([m.body_pos[m.body(f'clamp_finger{i}').id] for i in range(6)],axis=0);delta=shorten*v/np.linalg.norm(v)
 for i in range(6):m.body_pos[m.body(f'clamp_finger{i}').id]-=delta
 m.geom_pos[m.geom_bodyid==b]-=delta;m.body_ipos[b]-=delta
 return m,a,delta

def run():
 results=[]
 for sh in [.20,.30,.40]:
  m,a,delta=translated_model(sh);old=mujoco.MjModel.from_binary_path(str(BASE/'expert_model.mjb'));d=mujoco.MjData(m);ref=mujoco.MjData(old);aj=m.actuator_trnid[:,0];arm=np.r_[[i for i,j in enumerate(aj) if 'right' in m.joint(int(j)).name and any(t in m.joint(int(j)).name for t in ['shoulder','elbow','wrist'])],35];aq=m.jnt_qposadr[aj[arm]];gq=m.jnt_qposadr[aj[29:35]];pads=[m.geom(f'clamp_pad{i}').id for i in range(6)];seed=a['initial_qpos'][aq].copy();rows=[];poses=[]
  for step in [-1,*range(0,len(a['qpos']),120),len(a['qpos'])-1]:
   q=a['initial_qpos'].copy() if step<0 else a['qpos'][step].copy();ref.qpos[:]=q;ref.qpos[gq]=0;mujoco.mj_forward(old,ref);target=ref.geom_xpos[pads].copy()
   def residual(x):
    d.qpos[:]=q;d.qpos[gq]=0;d.qpos[aq]=x;mujoco.mj_forward(m,d);return np.r_[60*(d.geom_xpos[pads]-target).ravel(),.01*(x-seed)]
   fit=least_squares(residual,seed,bounds=(m.jnt_range[aj[arm],0]+.005,m.jnt_range[aj[arm],1]-.005),max_nfev=120);seed=fit.x;residual(seed);err=np.max(np.linalg.norm(d.geom_xpos[pads]-target,axis=1));bad=[]
   for c in d.contact:
    ns=[m.geom(int(g)).name for g in [c.geom1,c.geom2]]
    if c.dist<-.003 and not any(n.startswith('object/') for n in ns) and any(n.startswith('clamp_') for n in ns):bad.append([ns,float(c.dist)])
   rows.append(dict(step=step,pad_error_m=float(err),interference=bad,cost=fit.cost));poses.append(d.qpos.copy())
  result=dict(shorten_m=sh,delta_local=delta.tolist(),worst_error_m=max(x['pad_error_m'] for x in rows),rows=rows);results.append(result);np.savez(OUT/f'gate_{sh:.2f}.npz',qpos=poses);print(sh,result['worst_error_m'],'bad',[(r['step'],r['interference']) for r in rows if r['interference']],flush=True)
 (OUT/'translation_gate.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':run()
