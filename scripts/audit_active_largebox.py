#!/usr/bin/env python3
"""Explicit acceptance audit from independent replay traces, never from a score."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,mujoco
from active_largebox_config import reference_model
from active_largebox_artifacts import materialize
from scipy.spatial.transform import Rotation

def run(path):
 a=np.load(path);m=mujoco.MjModel.from_binary_path(str(materialize(path.with_name(path.stem+'_model.mjb'))));v=json.loads(path.with_name(path.stem+'_verification.json').read_text());z=np.load(path.with_name(path.stem+'_verification_trace.npz'));t=z['trace'];N=len(t)//4;task=t[:N*4].reshape(N,4,-1);ob=m.body('object/object').id;oq=m.jnt_qposadr[m.joint('object/floating_base_joint').id]
 # Stored post-step robot poses are used only for geometry measurement here.
 # This audit never steps or transports an object kinematically.
 d=mujoco.MjData(m);pads=[m.geom('clamp_pad'+str(i)).id for i in range(6)];q0=a['initial_qpos'];heights=[]
 for index in [0,160]:
  d.qpos[:]=q0 if index==0 else a['qpos'][index-1];mujoco.mj_kinematics(m,d);heights.append(float(d.geom_xpos[pads,2].mean()))
 air=~t[:,0].astype(bool);e=np.diff(np.r_[False,air,False].astype(int));starts=np.flatnonzero(e==1);ends=np.flatnonzero(e==-1);longest_idx=int(np.argmax(ends-starts));start=int(starts[longest_idx]);landing=int(ends[longest_idx]);opening=int(float(a['phase_opening_frame'])*4) if 'phase_opening_frame' in a else 510*4;withdrawal_end=int(float(a['phase_withdrawal_end_frame'])*4) if 'phase_withdrawal_end_frame' in a else opening+90*4
 landing_v=t[max(0,landing-1),20:23];landing_w=t[max(0,landing-1),23:26];final=t[-200:];after_open=t[opening:];after_withdraw=t[withdrawal_end:]
 initial_rot=Rotation.from_quat(np.roll(q0[oq+3:oq+7],-1));initial_up=initial_rot.inv().apply([0,0,1]);rot=Rotation.from_quat(np.roll(t[:,32:36],-1,axis=1));ups=rot.inv().apply(np.tile([0,0,1],(len(t),1)));tilts=np.arccos(np.clip(ups@initial_up,-1,1))
 def longest_run(mask):
  e=np.diff(np.r_[False,mask,False].astype(int));return int(np.max(np.flatnonzero(e==-1)-np.flatnonzero(e==1),initial=0))
 swing=[];transitions=[]
 for side in [0,1]:
  support=task[:,:,3+side].any(1);swing.append(longest_run(~support));transitions.append(int(np.count_nonzero(np.diff(support.astype(int)))))
 transfer=t[landing:opening];mass=float(m.body_mass[ob]);weight=mass*9.81;load_frames=(transfer[:,0]>0)&transfer[:,1:3].any(1)&(transfer[:,42]>.7*weight)
 object_geoms=[g for g in range(m.ngeom) if m.geom_bodyid[g]==ob and m.geom_contype[g]]
 carried_frames=np.arange(N);carried_frames=carried_frames[(carried_frames*4>=start)&((carried_frames+1)*4<=landing)]
 carried_task=task[carried_frames];carried_swing=[longest_run(~carried_task[:,:,3+side].any(1)) for side in [0,1]]
 support_states=carried_task[:,:,3:5].any(1);left_only=int((support_states[:,0]&~support_states[:,1]).sum());right_only=int((~support_states[:,0]&support_states[:,1]).sum());feet=t[start:landing,14:20].reshape(-1,2,3);foot_carry_motion=np.linalg.norm(feet[:,:,:2]-feet[0,:,:2],axis=2).max(0)
 carried_positions=t[start:landing,11:13];main_transport=float(np.linalg.norm(carried_positions-carried_positions[0],axis=1).max())
 force_limits=np.max(np.abs(m.actuator_forcerange),axis=1);torque_legal=bool(np.all(np.array(v['maximum_abs_actuator_force'])<=force_limits+1e-8))
 source_path=reference_model();old=mujoco.MjModel.from_binary_path(str(source_path));oldbody=old.body('object/object').id;oldgeoms=[g for g in range(old.ngeom) if old.geom_bodyid[g]==oldbody and old.geom_contype[g]];geomerr=0.;mesherr=0.
 if len(object_geoms)!=len(oldgeoms):geomerr=float('inf')
 else:
  for g,og in zip(object_geoms,oldgeoms):
   geomerr=max(geomerr,float(np.max(np.abs(m.geom_pos[g]-old.geom_pos[og]))),float(np.max(np.abs(m.geom_quat[g]-old.geom_quat[og]))))
   mi=int(m.geom_dataid[g]);oi=int(old.geom_dataid[og]);vv=m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi]+m.mesh_vertnum[mi]];ov=old.mesh_vert[old.mesh_vertadr[oi]:old.mesh_vertadr[oi]+old.mesh_vertnum[oi]]
   mesherr=max(mesherr,float(np.max(np.abs(vv-ov))) if vv.shape==ov.shape else float('inf'))
 self_impulse=sum(value for key,value in v['contact_impulses_Ns'].items() if key.startswith('ROBOT_SELF:'))
 peak_before=float(np.max(np.linalg.norm(t[start:landing,20:22],axis=1)));landing_xy=float(np.linalg.norm(landing_v[:2]));post_open_slide=float(np.max(np.linalg.norm(after_open[:,11:13]-after_open[0,11:13],axis=1)));landing_slide=float(np.max(np.linalg.norm(t[landing:,11:13]-t[landing,11:13],axis=1)))
 metrics=dict(reach_down_m=heights[0]-heights[1],air_start_physics_step=start,primary_landing_physics_step=landing,primary_landing_frame=landing/4,landing_velocity_convention='pre-contact physics state (previous post-step sample)',landing_velocity_COM_mps=landing_v.tolist(),landing_post_step_velocity_COM_mps=t[landing,20:23].tolist(),landing_horizontal_speed_mps=landing_xy,landing_vertical_speed_mps=float(abs(landing_v[2])),landing_angular_speed_radps=float(np.linalg.norm(landing_w)),peak_carried_horizontal_speed_mps=peak_before,terrain_accepting_weight_with_gripper_physics_steps=int(load_frames.sum()),continuous_gripper_support_at_landing=bool(t[max(0,landing-4):landing+4,1:3].any(1).all()),placement_slide_from_first_terrain_m=landing_slide,post_opening_slide_m=post_open_slide,post_opening_height_range_m=float(np.ptp(after_open[:,13])),post_opening_max_tip_rad=float(tilts[opening:].max()),post_withdrawal_EE_contact_physics_steps=int(after_withdraw[:,43].astype(bool).sum()),final_1s_max_object_linear_speed_mps=float(np.linalg.norm(final[:,20:23],axis=1).max()),final_1s_max_object_angular_speed_radps=float(np.linalg.norm(final[:,23:26],axis=1).max()),final_1s_root_linear_speed_p95_mps=float(np.percentile(np.linalg.norm(final[:,36:39],axis=1),95)),final_1s_root_angular_speed_p95_radps=float(np.percentile(np.linalg.norm(final[:,39:42],axis=1),95)),foot_swing_longest_task_frames=swing,foot_support_transitions=transitions,carried_foot_swing_longest_task_frames=carried_swing,carried_left_only_frames=left_only,carried_right_only_frames=right_only,carried_foot_motion_m=foot_carry_motion.tolist(),main_continuous_airborne_transport_m=main_transport,actuator_force_limits_respected=torque_legal,robot_self_contact_impulse_Ns=self_impulse,object_mesh_max_vertex_error_m=mesherr,object_geom_pose_max_error=geomerr,object_mass_kg=mass)
 gates=dict(original_geometry=mesherr==0 and geomerr==0 and abs(mass-.05)<1e-12,ground_start=bool(t[:4,0].all()),actual_reach_down=metrics['reach_down_m']>.05,actuated_closure=bool(t[:opening,26:32].max()>.04),free_pickup=v['maximum_lift_m']>.05 and v['longest_free_bilateral_lift_task_frames']>=10,walking_carry=main_transport>=.3 and min(carried_swing)>=3 and min(foot_carry_motion)>.10 and left_only>=3 and right_only>=3,controlled_landing=landing_xy<.20 and abs(landing_v[2])<.15 and np.linalg.norm(landing_w)<1.5 and metrics['continuous_gripper_support_at_landing'],terrain_load_transfer=int(load_frames.sum())>=40,deliberate_opening=bool((final[:,26:32]<-.025).all()),clean_withdrawal=metrics['post_withdrawal_EE_contact_physics_steps']==0 and landing_slide<.04 and post_open_slide<.02 and metrics['post_opening_height_range_m']<.01 and metrics['post_opening_max_tip_rad']<.15,stable_object=metrics['final_1s_max_object_linear_speed_mps']<.01 and metrics['final_1s_max_object_angular_speed_radps']<.03 and bool(final[:,0].all()),stable_robot=v['full_valid'] and metrics['final_1s_root_linear_speed_p95_mps']<.15 and metrics['final_1s_root_angular_speed_p95_radps']<.25,zero_forbidden_contacts=v['forbidden_object_impulse_Ns']<1e-6 and v['invalid_robot_terrain_impulse_Ns']<1e-6 and self_impulse<1e-6,exact_replay=v['exact_qpos_error']==0 and v['exact_qvel_error']==0,physical_actuation=torque_legal and v['maximum_applied_external_force']==0 and m.neq==0 and m.opt.disableflags==0 and not any(int(j)==m.joint('object/floating_base_joint').id for j in m.actuator_trnid[:,0]))
 result={'artifact':str(path),'metrics':metrics,'gates':{k:bool(value) for k,value in gates.items()},'all_machine_gates_pass':bool(all(gates.values())),'visual_review_complete':False,'success':False,'replay_report':v};path.with_name(path.stem+'_acceptance_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('artifact',type=Path);run(p.parse_args().artifact)
