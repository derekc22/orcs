"""Placement comparison with unchanged windows/method and event-based supplements."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'logs/compact_smooth_integration_v2/complete_execution/rollout.npz'
DT=.005
def rms(x):return float(np.sqrt(np.mean(np.sum(x*x,axis=1))))
def settled(mask,start):
 bad=np.flatnonzero(~mask[start:]);return None if len(bad) and start+bad[-1]+1>=len(mask) else (start+(bad[-1]+1 if len(bad) else 0))*DT

def measure(p):
 a=np.load(p);m=mujoco.MjModel.from_binary_path(str(p.with_name(p.stem+'_model.mjb')));t=np.load(p.with_name(p.stem+'_verification_trace.npz'))['trace'];l=np.load(p.with_name(p.stem+'_locomotion_trace.npz'))['trace'];q=a['qpos'];v=a['qvel'];oj=m.joint('object/floating_base_joint').id;oq=m.jnt_qposadr[oj];od=m.jnt_dofadr[oj]
 rot=R.from_quat(np.roll(q[:,oq+3:oq+7],-1,axis=1));w=rot.apply(v[:,od+3:od+6]);al=np.gradient(w,DT,axis=0);s=int(a['placement_start_physics_step']);e=int(a['recovery_start_physics_step']);op=round(float(a['phase_opening_frame'])*4);touch=s+int(np.flatnonzero(t[s:,0])[0]);last=int(np.flatnonzero(t[:,43])[-1]);tr=int(a['terrain_transfer_start_physics_step']);ws=int(a['walking_start_physics_step']);we=int(a['walking_end_physics_step']);fork=int(a['underfinger_opening_physics_step']);weight=m.body_mass[m.jnt_bodyid[oj]]*9.81
 initialR=R.from_quat(np.roll(a['initial_qpos'][oq+3:oq+7],-1));normal=rot.apply(initialR.inv().apply([0,0,1]));tilt=np.degrees(np.arccos(np.clip(normal[:,2],-1,1)));lin=np.linalg.norm(t[:,20:23],axis=1);ang=np.linalg.norm(w,axis=1)
 grounded=t[:,0].astype(bool);loaded=t[:,42]>.7*weight
 sustained=[]
 for b in np.flatnonzero(loaded & ~np.r_[False,loaded[:-1]]):
  if b>s and loaded[b:b+30].all() and len(loaded[b:b+30])==30:sustained.append(int(b))
 windows={'placement':(s,e),'fixed_baseline_duration':(s,min(s+2021,len(q))),'before_touchdown':(s,touch),'load_transfer':(touch,op),'opening_to_clear':(op,last+1),'after_clear_first_0_5s':(last+1,last+101),'walking':(ws,we)}
 metrics={}
 for name,(b,c) in windows.items():
  metrics[name]=dict(start_s=b*DT,end_s=c*DT,duration_s=(c-b)*DT,object_alpha_rms=rms(al[b:c]),object_alpha_peak=float(np.linalg.norm(al[b:c],axis=1).max()),object_alpha_squared_integral=float((al[b:c]**2).sum()*DT),object_omega_rms=rms(w[b:c]),object_omega_peak=float(ang[b:c].max()),object_COM_speed_peak=float(lin[b:c].max()))
 result=dict(artifact=str(p),controls=len(q),duration_s=len(q)*DT,windows=metrics,events_s=dict(placement=s*DT,fork_open_command=fork*DT,initial_terrain_contact=touch*DT,terrain_70percent_150ms_support_start=sustained[0]*DT,compliant_transfer=tr*DT,deliberate_opening=op*DT,last_hand_contact=last*DT,recovery=e*DT),touchdown=dict(angular_speed_radps=float(ang[touch]),COM_linear_speed_mps=float(lin[touch]),COM_vertical_velocity_mps=float(t[touch,22]),tilt_deg=float(tilt[touch])),terrain_load_transfer=dict(max_tilt_deg=float(tilt[touch:op+1].max()),tilt_range_deg=float(np.ptp(tilt[touch:op+1])),angular_path_rad=float(ang[touch:op+1].sum()*DT),terrain_load_peak_N=float(t[touch:last+1,42].max()),grounded_fraction=float(grounded[touch:op].mean()),terrain_load_at_opening_N=float(t[op,42]),hand_load_at_opening_N=l[op,16:18].tolist()),post_release=dict(max_orientation_error_from_settled_after_clear_first_half_second_deg=float(np.degrees((rot[last+1:last+101]*rot[-1].inv()).magnitude().max())),max_COM_distance_from_settled_after_clear_first_half_second_m=float(np.linalg.norm(t[last+1:last+101,11:14]-t[-1,11:14],axis=1).max()),max_COM_speed_first_half_second_mps=float(lin[op:op+100].max()),max_angular_speed_first_half_second_radps=float(ang[op:op+100].max()),max_tilt_first_half_second_deg=float(tilt[op:op+100].max()),last_hand_to_linear_settled_s=settled(lin<.001,last+1)-(last+1)*DT if settled(lin<.001,last+1) is not None else None,last_hand_to_angular_settled_s=settled(ang<.003,last+1)-(last+1)*DT if settled(ang<.003,last+1) is not None else None,final_1s_COM_speed_peak_mps=float(lin[-200:].max()),final_1s_angular_speed_peak_radps=float(ang[-200:].max())),walking=dict(root_net_displacement_m=float(np.linalg.norm(q[we-1,:2]-q[ws,:2])),minimum_hand_loads_N=l[ws:we,16:18].min(0).tolist()),final_robot=dict(root_linear_speed_peak_1s=float(np.linalg.norm(v[-200:,:3],axis=1).max()),root_angular_speed_peak_1s=float(np.linalg.norm(v[-200:,3:6],axis=1).max()),foot_load_min_1s_N=l[-200:,14:16].min(0).tolist()))
 return result,dict(time=np.arange(len(q))*DT,w=w,al=al,tilt=tilt,lin=lin,terrain=t[:,42],hands=l[:,16:18])

def run(args):
 out=args.output;out.mkdir(exist_ok=True,parents=True);results=[];traces=[]
 for p in [BASE,args.candidate]:
  r,x=measure(p);results.append(r);traces.append(x)
 a=np.load(BASE);b=np.load(args.candidate);prefix=int(b['placement_start_physics_step']);identity={k:bool(np.array_equal(a[k][:prefix],b[k][:prefix])) for k in ['ctrl_physics','qpos','qvel','contacts']};identity['prefix_controls']=prefix;identity['prefix_duration_s']=prefix*DT;identity['models_byte_identical']=BASE.with_name('rollout_model.mjb').read_bytes()==args.candidate.with_name('rollout_model.mjb').read_bytes()
 report=dict(baseline_commit='b2673fdb13329fe238ddecaf47d83e8986a024d6',method='Unfiltered 200 Hz world angular velocities and numpy.gradient at dt=.005, identical to accepted motion_metrics.py. Primary placement window is placement_start to recovery_start for both. Fixed accepted 10.105s and event windows are supplements; no padding/filtering/time warping.',prefix_identity=identity,baseline=results[0],candidate=results[1]);(out/'placement_comparison.json').write_text(json.dumps(report,indent=2))
 fig,ax=plt.subplots(5,1,figsize=(13,13),sharex=True)
 for label,r,x,col in zip(['Accepted b2673fdb','Placement refinement'],results,traces,['#777777','#127c98']):
  s=r['windows']['placement']['start_s'];mask=x['time']>=s;t=x['time'][mask]
  for aa,key,ylabel in [(ax[0],'al','Angular acceleration norm [rad/s²]'),(ax[1],'w','Angular speed [rad/s]'),(ax[2],'tilt','Tilt from initial terrain normal [deg]'),(ax[3],'terrain','Terrain load [N]')]:
   y=x[key][mask];aa.plot(t,np.linalg.norm(y,axis=1) if y.ndim>1 else y,color=col,label=label,alpha=.85);aa.set_ylabel(ylabel)
  ax[4].plot(t,x['hands'][mask].sum(1),color=col,label=label);ax[4].set_ylabel('Combined hand load [N]')
  for aa in ax:
   for event in ['initial_terrain_contact','deliberate_opening']:aa.axvline(r['events_s'][event],color=col,ls=':',alpha=.6)
 ax[0].legend();ax[-1].set_xlabel('Physical time [s]');[aa.grid(alpha=.2) for aa in ax];fig.tight_layout();fig.savefig(out/'placement_comparison.png',dpi=150)
 print(json.dumps(dict(prefix_identity=identity,baseline=results[0],candidate=results[1]),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('candidate',type=Path);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
