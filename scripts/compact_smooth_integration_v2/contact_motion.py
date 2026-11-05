"""Independent exact replay of loaded-contact slip and fixed material anchors.

Slip speed is measured at the solver's contact point and tangent plane from
point Jacobians and the SAME pre-integration velocity used for that contact
solve. It is force-weighted per hand; its time integral is not the displacement
of a single persistent contact patch. Fixed material anchors separately bound
object/palm relative excursions and rocking, using the accepted audit method.
"""
import argparse,json
from pathlib import Path
import mujoco,numpy as np
from scipy.spatial.transform import Rotation as R

def material_anchors(m,a,s):
 d=mujoco.MjData(m);d.qpos[:]=a['qpos'][s];d.qvel[:]=a['qvel'][s];d.ctrl[:]=a['ctrl_physics'][s];mujoco.mj_forward(m,d);ob=m.body('object/object').id;anchors=[]
 for side in ['left','right']:
  b=m.body('hand_'+side+'_plate').id;ps=[];ws=[]
  for k,c in enumerate(d.contact):
   if not any(m.geom(int(g)).name.startswith('hand_'+side+'_pad') for g in c.geom) or not any(m.geom_bodyid[g]==ob for g in c.geom):continue
   f=np.zeros(6);mujoco.mj_contactForce(m,d,k,f);w=np.linalg.norm(f[:3])
   if w>.001:ps.append(c.pos.copy());ws.append(w)
  anchors.append(d.xmat[b].reshape(3,3).T@(np.average(ps,axis=0,weights=ws)-d.xpos[b]))
 return anchors

def run(path,out,start_s=None,end_s=None):
 a=np.load(path);m=mujoco.MjModel.from_binary_path(str(path.with_name(path.stem+'_model.mjb')));d=mujoco.MjData(m);d.qpos[:]=a['initial_qpos'];d.qvel[:]=a['initial_qvel'];mujoco.mj_forward(m,d)
 s=int(a['walking_start_physics_step']) if start_s is None else round(start_s/.005);e=min(len(a['ctrl_physics']),int(a['walking_end_physics_step']) if end_s is None else round(end_s/.005));ob=m.body('object/object').id;plates=[m.body('hand_'+side+'_plate').id for side in ['left','right']];J1=np.zeros((3,m.nv));J2=J1.copy();Jr=J1.copy();rows=[];anchors=material_anchors(m,a,s);poses=[];qp=0.;qv=0.
 for step,u in enumerate(a['ctrl_physics'][:e]):
  beforev=d.qvel.copy();d.ctrl[:]=u;mujoco.mj_step(m,d);qp=max(qp,float(np.max(abs(d.qpos-a['qpos'][step]))));qv=max(qv,float(np.max(abs(d.qvel-a['qvel'][step]))))
  if step<s:continue
  # mj_step contact/Jacobian coordinates precede integration, matching beforev.
  totals=np.zeros((2,5));positions=[[],[]];weights=[[],[]]
  for ci,c in enumerate(d.contact):
   b1,b2=m.geom_bodyid[c.geom];
   if ob not in [b1,b2]:continue
   other=c.geom2 if b1==ob else c.geom1;name=m.geom(other).name
   if not name.startswith('hand_'):continue
   i=0 if name.startswith('hand_left_') else 1;f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f);load=max(0,float(f[0]))
   if load<=1e-4:continue
   mujoco.mj_jac(m,d,J1,Jr,c.pos,int(b1));mujoco.mj_jac(m,d,J2,Jr,c.pos,int(b2));relative=c.frame.reshape(3,3)@((J2-J1)@beforev);speed=float(np.linalg.norm(relative[1:]));totals[i]+=np.r_[load,load*speed,load*speed**2,float(speed>.005)*load,speed]
   if name.startswith('hand_'+['left','right'][i]+'_pad'):
    positions[i].append(c.pos.copy());weights[i].append(load)
  vals=[]
  for i in range(2):
   load=max(totals[i,0],1e-20);vals.extend([totals[i,0],totals[i,1]/load,totals[i,2]/load,totals[i,3]/load])
  rows.append(vals)
  # Recompute FK at the exact saved state for material-point excursions.
  fk=mujoco.MjData(m) if step==s else fk;fk.qpos[:]=d.qpos;mujoco.mj_kinematics(m,fk)
  if anchors is None:
   anchors=[fk.xmat[b].reshape(3,3).T@(np.average(positions[i],axis=0,weights=weights[i])-fk.xpos[b]) for i,b in enumerate(plates)]
  row=[]
  for b in plates:
   H=fk.xmat[b].reshape(3,3);row.extend(H.T@(fk.xpos[ob]-fk.xpos[b]));row.extend(R.from_matrix(H.T@fk.xmat[ob].reshape(3,3)).as_quat())
  poses.append(row)
 x=np.array(poses);rows=np.array(rows);result=dict(exact_qpos_error=qp,exact_qvel_error=qv,measurement_interval_s=[s*.005,e*.005],hands={})
 for i,side in enumerate(['left','right']):
  T=x[:,7*i:7*i+3];rot=R.from_quat(x[:,7*i+3:7*i+7]);P=anchors[i];O=rot[0].inv().apply(P-T[0]);delta=rot.apply(O)+T-P
  result['hands'][side]=dict(peak_relative_rotation_deg=float(np.degrees((rot*rot[0].inv()).magnitude().max())),max_COM_excursion_mm=float(np.linalg.norm(T-T[0],axis=1).max()*1000),fixed_material_point_max_tangential_excursion_mm=float(np.linalg.norm(delta[:,:2],axis=1).max()*1000),fixed_material_point_final_tangential_excursion_mm=float(np.linalg.norm(delta[-1,:2])*1000),loaded_contact_tangential_speed_RMS_mps=float(np.sqrt(rows[:,4*i+2].mean())),loaded_contact_mean_tangential_speed_mps=float(rows[:,4*i+1].mean()),integrated_force_weighted_slip_speed_mm=float(rows[:,4*i+1].sum()*.005*1000),normal_force_weighted_fraction_above_5mmps=float(rows[:,4*i+3].mean()))
 result['method']=__doc__;out.mkdir(exist_ok=True,parents=True);(out/'contact_slip.json').write_text(json.dumps(result,indent=2));np.savez_compressed(out/'contact_slip_trace.npz',force_weighted_slip=rows,plate_object_pose=x);print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('artifact',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--start-s',type=float);p.add_argument('--end-s',type=float);a=p.parse_args();run(a.artifact,a.out,a.start_s,a.end_s)
