"""Exact continuation: lower, earn terrain support, open both hands and withdraw."""
import argparse
import json,hashlib
import sys,os,shutil
from pathlib import Path
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/compact_integration_v1'),str(ROOT/'scripts/corner_hands_v1'),str(ROOT/'scripts/articulated_gripper_v1'),str(ROOT/'scripts')]
sys.path.append(str(ROOT/'scripts/compact_integration_v2'))
from mechanism import indices
OUT=None
from scipy.spatial.transform import Rotation
from placement_geometry import plan
from physical import Arms,contacts,squeeze_load,smooth,PitchServo,thumb_control
from compact_gripper_balance import Balance
from loaded_arms import LoadedArms
from arms_with_floor import ClearancePlacementArms as PlacementArms
from bounded_balance import BoundedBalance

def run(tag,source,trim=None,force_target=5.,plan_only=False,forward=0.,fork_release_height=None,seconds=17.,plan_cache=None,floor_clearance=0.,transfer_wait=.7):
    a=np.load(OUT/(source+'.npz'));m=mujoco.MjModel.from_binary_path(str(OUT/(source+'_model.mjb')))
    prefix=len(a['ctrl_physics']) if trim is None else int(round(trim/.005))
    d=mujoco.MjData(m);d.qpos[:]=a['initial_qpos'];d.qvel[:]=a['initial_qvel'];mujoco.mj_forward(m,d)
    for u0 in a['ctrl_physics'][:prefix]:d.ctrl[:]=u0;mujoco.mj_step(m,d)
    aj,aq,av,arm,pads=indices(m);wi=np.array([int(np.flatnonzero(aj==m.joint('waist_'+ss+'_joint').id)[0]) for ss in ['yaw','roll','pitch']]);previous_waist=None;arms=PlacementArms(m,d,max_speed=3.,floor_clearance=floor_clearance);balance=BoundedBalance(m,d)
    press_rest=d.qpos[aq[29:31]].copy()
    ob=m.body('object/object').id;oj=m.joint('object/floating_base_joint').id;oq=m.jnt_qposadr[oj];od=m.jnt_dofadr[oj]
    fk=mujoco.MjData(m);fk.qpos[:]=d.qpos;fk.qpos[aq[29:31]]=0.;mujoco.mj_forward(m,fk)
    anchor=fk.geom_xpos[pads].copy();rot=fk.xmat[arms.hands].reshape(2,3,3).copy();normals=rot[:,:,2]
    initial_height=a['initial_qpos'][oq+2];lift=d.qpos[oq+2]-initial_height;start=d.qpos.copy()
    pitch_enabled=bool(a['pitch_servo_enabled']) if 'pitch_servo_enabled' in a else False
    attitude=PitchServo(m,d,normals[0])
    # Solve the deep-squat branch from the ground upwards; continuation from
    # standing otherwise converges to a straight-leg bow with excessive tilt.
    ground=np.load(ROOT/'logs/corner_hands_v1/reach3.npz')['qpos'][1].copy();gd=mujoco.MjData(m);gd.qpos[:]=ground;mujoco.mj_forward(m,gd)
    footids=[m.body(ss+'_ankle_roll_link').id for ss in ['left','right']]
    oldmid=gd.xpos[footids].mean(0);newmid=balance.pos.mean(0)
    hyaw=Rotation.from_quat(np.roll(start[3:7],-1)).as_euler('xyz')[2];heading=np.array([np.cos(hyaw),np.sin(hyaw),0.]);oyaw=Rotation.from_quat(np.roll(ground[3:7],-1)).as_euler('xyz')[2];align=Rotation.from_euler('z',hyaw-oyaw)
    ground[:3]=align.apply(ground[:3]-oldmid)+newmid;ground[3:7]=np.roll((align*Rotation.from_quat(np.roll(ground[3:7],-1))).as_quat(),1)
    original_feet_rot=balance.rot.copy();placement_feet_rot=balance.rot.copy()
    # Preserve the physically loaded stance; forcing both feet to pivot while
    # lowering is a separate unsupported transition in this morphology.
    acts=np.r_[np.arange(29),np.arange(31,33)].astype(int);dr=(Rotation.from_quat(np.roll(ground[3:7],-1))*Rotation.from_quat(np.roll(start[3:7],-1)).inv()).as_rotvec()
    seed=np.r_[ground[:3],dr,ground[aq[acts]]];poses=[];rows=[]
    placement_com=d.subtree_com[1,:2].copy()
    prefix_hash=hashlib.sha256(a['ctrl_physics'][:prefix].tobytes()).hexdigest()
    if plan_cache is None:
        for lower in np.linspace(lift+.025,0,15):
            shift=forward*min(lower/.3,1.)*heading
            predicted=start.copy();predicted[oq+2]-=lower;predicted[oq:oq+3]+=shift
            predicted[aq[33:35]]=0. if fork_release_height is None or lift-lower<fork_release_height else 1.57
            q,seed,r=plan(m,predicted,anchor-[0,0,lower]+shift,rot,balance.pos,placement_feet_rot,seed,max_nfev=220,com_forward=0.,min_root=.38,com_target=placement_com,upright=False,target_yaw_override=hyaw,waist_yaw_reference=start[aq[wi[0]]],root_height_target=max(.38,start[2]-.65*lower))
            print('LOWERING_PLAN',round(lower,4),r,flush=True);poses.append(q);rows.append(r)
        poses=poses[::-1];rows=rows[::-1]
    else:
        cache=np.load(ROOT/plan_cache)
        assert str(cache['prefix_hash'])==prefix_hash, 'Plan cache must match the exact physical prefix'
        poses=list(cache['qpos']);rows=json.loads(str(cache['rows_json']))
    (OUT/(tag+'_placement_plan.json')).write_text(json.dumps(rows,indent=2));np.savez(OUT/(tag+'_placement_plan.npz'),qpos=poses,prefix_hash=prefix_hash,rows_json=json.dumps(rows))
    if plan_only:return
    static_reports=[]
    for q in poses:
        check=mujoco.MjData(m);check.qpos[:]=q;mujoco.mj_forward(m,check);allocator=BoundedBalance(m,check);tau,_=allocator.torques(check,q)
        err=np.array(allocator.last_result['root_wrench_error']);static_reports.append(dict(force_error_N=float(np.linalg.norm(err[:3])),moment_error_Nm=float(np.linalg.norm(err[3:])),cone_slack=allocator.last_result['minimum_cone_slack']))
    (OUT/(tag+'_static_support_gate.json')).write_text(json.dumps(static_reports,indent=2))
    if max(r['foot_error_m'] for r in rows)>.005 or max(r['max_tip_error_m'] for r in rows)>.015 or max(r['moment_error_Nm'] for r in static_reports)>.5 or max(r['force_error_N'] for r in static_reports)>3.:
        print('PLACEMENT_GEOMETRY_GATE_FAILED',flush=True);return
    cs=list(a['ctrl_physics'][:prefix]);qs=list(a['qpos'][:prefix]);vs=list(a['qvel'][:prefix]);cts=list(a['contacts'][:prefix]);diag=[]
    finger_hold=a['finger_hold_control'] if 'finger_hold_control' in a and len(a['finger_hold_control']) else d.ctrl[29:31].copy()
    closure=np.zeros(2);filtered_force=squeeze_load(m,d,normals);z_correction=0.;load_run=0;load_anchor=None;release_step=None;withdrawal_end=-1;withdraw_q=None;sums=np.zeros(6)
    progress=0.;finger_opening_step=0 if fork_release_height is None else None;finger_opened=False;transfer_anchor=None;transfer_start=None;transfer_run=0;opening_thumb=None;opening_press=None
    for step in range(round(seconds/.005)):
        sec=step*.005
        for i in range(2):
            drfoot=Rotation.from_matrix(placement_feet_rot[i]@original_feet_rot[i].T).as_rotvec();balance.rot[i]=Rotation.from_rotvec(drfoot*smooth(sec/1.)).as_matrix()@original_feet_rot[i]
        if finger_opening_step is None and d.qpos[oq+2]-initial_height<fork_release_height:
            finger_opening_step=step;print('UNDERFINGER_OPENING_WHILE_PALMS_RETAIN',sec,flush=True)
        if finger_opening_step is not None and (d.qpos[aq[33:35]]<.12).all():finger_opened=True
        if finger_opening_step is None or finger_opened:progress+=.005
        alpha=smooth(progress/6.);t=alpha*14;idx=min(int(t),13);frac=t-idx;qref=poses[idx]*(1-frac)+poses[idx+1]*frac
        if progress<.7:qref=start*(1-smooth(progress/.7))+qref*smooth(progress/.7)
        target=anchor-[0,0,(lift+.025)*alpha]+forward*min((lift+.025)*alpha/.3,1.)*heading
        if load_anchor is None:
            closure=np.clip(closure+2*(d.qpos[aq[29:31]]-press_rest)*.005,-.025,.035)
            for side in range(2):target[3*side:3*side+3]-=closure[side]*normals[side]
            if sec>1 and d.qpos[oq+2]-initial_height<.07:
                z_correction=float(np.clip(z_correction+(-.028-d.qvel[od+2])*.005,0,.045));target[:,2]+=z_correction
            if pitch_enabled:target=attitude.correct(d,target)
        else:
            target=load_anchor.copy();qref=held_q.copy()
            if sec>5.3 and release_step is None:
                release_step=step;opening_thumb=d.qpos[aq[35:37]].copy();opening_press=d.qpos[aq[29:31]].copy();print('DELIBERATE_OPENING',sec,flush=True)
            if release_step is not None:
                elapsed=(step-release_step)*.005;out=.075*smooth(elapsed/1.5);up=.45*smooth((elapsed-1.7)/2.3);back=.20*smooth((elapsed-1.7)/2.3)
                for side in range(2):target[3*side:3*side+3]+=out*normals[side]+[0,0,up]-back*heading
                if elapsed>=1.7 and withdraw_q is None:
                    final_target=load_anchor.copy()
                    for side in range(2):final_target[side*3:side*3+3]+=.075*normals[side]+[0,0,.45]-.20*heading
                    withdraw_q,_,rr=plan(m,d.qpos.copy(),final_target,rot,balance.pos,balance.rot,max_nfev=200,upright=True,com_forward=0.,min_root=.35)
                    print('WITHDRAW_PLAN',rr,flush=True)
                if withdraw_q is not None:qref=held_q*(1-smooth((elapsed-1.7)/2.3))+withdraw_q*smooth((elapsed-1.7)/2.3)
                withdrawal_end=prefix+release_step+800
        if transfer_anchor is not None and load_anchor is None:
            target=transfer_anchor.copy();qref=transfer_q.copy()
        if step%4==0:arms.solve(d,target,qref)
        qbalance=qref.copy();qbalance[arms.armq]=arms.seed;vbalance=np.zeros(m.nv);vbalance[av[arm]]=arms.vel
        tau,wrench=balance.torques(d,qbalance,vbalance);u=d.qpos[aq]+(tau[av]-m.actuator_biasprm[:,2]*d.qvel[av])/m.actuator_gainprm[:,0];u=arms.apply(d,u)
        preload=force_target if transfer_start is None else force_target+(.05-force_target)*smooth((step-transfer_start)*.005/.7)
        u[29:31]=arms.grip(d,preload) if release_step is None else opening_press+(-.025-opening_press)*smooth((step-release_step)*.005/.6)
        u[35:37]=thumb_control(m,d,torque=.05*preload) if release_step is None else opening_thumb*(1-smooth((step-release_step)*.005/.6))
        wvel=np.zeros(3) if previous_waist is None else np.clip((qref[aq[wi]]-previous_waist)/.005,-2.,2.)
        previous_waist=qref[aq[wi]].copy();u[wi]=qref[aq[wi]]+d.qfrc_bias[av[wi]]/m.actuator_gainprm[wi,0]+(-m.actuator_biasprm[wi,2]/m.actuator_gainprm[wi,0])*wvel
        u[33:35]=1.57 if finger_opening_step is None else 0.
        u=np.clip(u,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);d.ctrl[:]=u;mujoco.mj_step(m,d)
        touch,force,imp,vertical,selfimp,invalid=contacts(m,d);sums+=np.r_[imp,selfimp,invalid]
        net_load=squeeze_load(m,d,normals);filtered_force+=.15*(net_load-filtered_force)
        gentle=np.linalg.norm(d.qvel[od:od+3])<.08 and np.linalg.norm(d.qvel[od+3:od+6])<.3
        transfer_run=transfer_run+1 if touch[0] and vertical>.2*.05*9.81 and gentle and d.qpos[oq+2]-initial_height<.035 and finger_opened else 0
        if transfer_run>=10 and transfer_anchor is None:
            transfer_anchor=target.copy();transfer_q=qref.copy();transfer_start=step
            print('COMPLIANT_TERRAIN_LOAD_TRANSFER',sec,flush=True)
        load_run=load_run+1 if touch[0] and vertical>.7*.05*9.81 and gentle and d.qpos[oq+2]-initial_height<.008 else 0
        if load_run>=30 and load_anchor is None and transfer_start is not None and step-transfer_start>=round(transfer_wait/.005):
            fk.qpos[:]=d.qpos;fk.qpos[aq[29:31]]=0.;mujoco.mj_forward(m,fk)
            load_anchor=fk.geom_xpos[pads].copy();held_q=qref.copy();arms.seed=d.qpos[arms.armq].copy();arms.vel[:]=0
            print('TERRAIN_WEIGHT_TRANSFER',sec,flush=True)
        cs.append(u.copy());qs.append(d.qpos.copy());vs.append(d.qvel.copy());cts.append(touch);diag.append(np.r_[sec,d.qpos[oq+2]-initial_height,force,net_load,vertical,arms.cost,sums])
        if step%100==99:print(tag,round(sec,3),'root',round(d.qpos[2],3),'lift',round(d.qpos[oq+2]-initial_height,3),'touch',touch,'load',round(vertical,3),'cost',round(arms.cost,3),'bad',sums[3:],flush=True)
        if d.qpos[2]<.32 or sums[3]>.00001 or sums[4]>.001 or sums[5]>.001:print('EARLY_REJECT',flush=True);break
        # Hand off exactly at completed withdrawal to recover.py's measured
        # support hold. Continuing this pose-tracking controller after the
        # rise causes drift; its later rejected tail is preserved in the logs.
        if release_step is not None and prefix+step>=withdrawal_end:break
    result=dict(initial_qpos=a['initial_qpos'],initial_qvel=a['initial_qvel'],ctrl_physics=cs,qpos=qs,qvel=vs,contacts=cts,placement_diagnostics=diag,placement_start_physics_step=prefix,terrain_transfer_start_physics_step=np.array(-1 if transfer_start is None else prefix+transfer_start),underfinger_opening_physics_step=np.array(-1 if finger_opening_step is None else prefix+finger_opening_step),phase_opening_frame=np.array(-1 if release_step is None else (prefix+release_step)/4),phase_withdrawal_end_frame=np.array(withdrawal_end/4),finger_hold_control=finger_hold)
    for key in ['walking_start_physics_step','walking_end_physics_step','transition_start_physics_step']:
        if key in a:result[key]=a[key]
    (OUT/(tag+'_controller_result.json')).write_text(json.dumps(dict(prefix=prefix,continuation_seconds=len(diag)*.005,impulses=sums.tolist(),release_step=release_step,withdrawal_complete=bool(release_step is not None and prefix+step>=withdrawal_end)),indent=2))
    np.savez_compressed(OUT/(tag+'.npz'),**result);mujoco.mj_saveModel(m,str(OUT/(tag+'_model.mjb')),None)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--source',required=True);p.add_argument('--trim',type=float);p.add_argument('--force-target',type=float,default=2.);p.add_argument('--plan-only',action='store_true');p.add_argument('--forward',type=float,default=0.);p.add_argument('--fork-release-height',type=float);p.add_argument('--seconds',type=float,default=17.);p.add_argument('--plan-cache');p.add_argument('--floor-clearance',type=float,default=0.);p.add_argument('--transfer-wait',type=float,default=.7);a=p.parse_args()
    source=ROOT/a.source;OUT=ROOT/'logs/compact_smooth_integration_v2'/a.tag;OUT.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(source,OUT/'source.npz');os.link(source.with_name(source.stem+'_model.mjb'),OUT/'source_model.mjb')
    (OUT/'generation_arguments.json').write_text(json.dumps(vars(a),indent=2));run('rollout','source',a.trim,a.force_target,a.plan_only,a.forward,a.fork_release_height,a.seconds,a.plan_cache,a.floor_clearance,a.transfer_wait)
