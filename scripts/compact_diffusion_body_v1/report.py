"""Phase-specific executed causal evidence; all physical gates stay unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT=Path(__file__).resolve().parents[2]
G4=ROOT/'logs/compact_diffusion_authority_v1'
WINDOWS={'pickup':(0,3449),'carry':(3449,4721),'placement':(6701,8092)}
ACTIVE={'pickup':(540,1601),'carry':(3449,4722),'placement':(6701,7761)}
EXACT=['exact_qpos_error','exact_qvel_error','exact_control_error','exact_contact_trace_error','exact_locomotion_trace_error']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def summary(values):
    values=np.asarray(values)
    return dict(rms=float(np.sqrt(np.mean(values**2))),peak=float(abs(values).max()))


def prepare(path,m):
    data=dict(np.load(path/'rollout.npz'))
    torso=m.body('robot/torso_link').id
    fk=mujoco.MjData(m);orient=[];momentum=[]
    for q,v in zip(data['qpos'],data['qvel']):
        fk.qpos[:]=q;fk.qvel[:]=v;mujoco.mj_kinematics(m,fk);mujoco.mj_comPos(m,fk)
        mujoco.mj_comVel(m,fk);mujoco.mj_subtreeVel(m,fk)
        orient.append(fk.xmat[torso].reshape(3,3).copy())
        momentum.append(fk.subtree_angmom[0].copy())
    data['torso_quaternion']=R.from_matrix(np.asarray(orient)).as_quat(scalar_first=True)
    data['whole_system_angular_momentum']=np.asarray(momentum)
    return data


def compare(a,b,arms,waist,windows):
    result={}
    for phase,(s,e) in windows.items():
        qa,qb=a['qpos'][s:e],b['qpos'][s:e];d=qa-qb
        def rotation(x,y):
            rv=(R.from_quat(x,scalar_first=True)*R.from_quat(y,scalar_first=True).inv()).as_rotvec()
            return dict(**summary(np.linalg.norm(rv,axis=1)),per_axis_rms=np.sqrt(np.mean(rv**2,axis=0)).tolist())
        arm=d[:,arms];w=d[:,waist];control=a['ctrl_physics'][s:e]-b['ctrl_physics'][s:e]
        result[phase]=dict(root_position_m=dict(**summary(np.linalg.norm(d[:,:3],axis=1)),per_axis_rms=np.sqrt(np.mean(d[:,:3]**2,axis=0)).tolist()),
            root_orientation_rad=rotation(qa[:,3:7],qb[:,3:7]),
            torso_orientation_rad=rotation(a['torso_quaternion'][s:e],b['torso_quaternion'][s:e]),
            waist_joints_rad=dict(**summary(w),per_joint_rms=np.sqrt(np.mean(w**2,axis=0)).tolist(),per_joint_peak=abs(w).max(0).tolist()),
            arms_rad=dict(**summary(arm),per_joint_rms=np.sqrt(np.mean(arm**2,axis=0)).tolist()),
            object_position_m=summary(np.linalg.norm(d[:,44:47],axis=1)),
            object_orientation_rad=rotation(qa[:,47:51],qb[:,47:51]),
            actuator_commands=dict(**summary(control),per_actuator_rms=np.sqrt(np.mean(control**2,axis=0)).tolist(),
                per_actuator_peak=abs(control).max(0).tolist(),native_joint_targets_rad=summary(control[:,:29]),
                gripper_press_targets_m=summary(control[:,29:31]),gripper_hinge_targets_rad=summary(control[:,31:])) )
    return result


def tracking(new,old,parameters,waist_yaw):
    out={}
    root_angle=(R.from_quat(new['qpos'][:,3:7],scalar_first=True)*R.from_quat(old['qpos'][:,3:7],scalar_first=True).inv()).as_rotvec()[:,2]
    # References index the pre-step state; compare executed post-step states
    # with the next reference state, without time fitting or alignment.
    target=np.r_[parameters[1:],parameters[-1:]]
    for phase,(s,e) in ACTIVE.items():
        values={}
        for name,actual,expected in [
            ('root_height',new['qpos'][:,2]-old['qpos'][:,2],target[:,2]),
            ('root_yaw',root_angle,target[:,3]),
            ('waist_yaw',new['qpos'][:,waist_yaw]-old['qpos'][:,waist_yaw],target[:,16])]:
            x,y=actual[s:e],expected[s:e]
            values[name]=dict(actual=summary(x),requested=summary(y),tracking_error=summary(x-y),
                correlation=float(np.corrcoef(x,y)[0,1]) if np.std(x)>1e-10 and np.std(y)>1e-10 else None,
                least_squares_gain=float(x@y/(y@y)) if y@y>1e-15 else None)
        out[phase]=values
    return out


def physical_quality(path,baseline):
    t=np.load(path/'rollout_verification_trace.npz')['trace'];b=np.load(baseline/'rollout_verification_trace.npz')['trace']
    # Full placement descent and initial terrain contact, with no smoothing.
    s,e=6701,7880
    def measures(t):
        v=t[s:e,20:23];w=t[s:e,23:26];acc=np.gradient(v,.005,axis=0)
        return dict(placement_object_linear_velocity=summary(np.linalg.norm(v,axis=1)),
            placement_object_angular_velocity=summary(np.linalg.norm(w,axis=1)),
            placement_object_acceleration=summary(np.linalg.norm(acc,axis=1)),
            terrain_support_peak_N=float(t[7750:8092,42].max()))
    return dict(goal5=measures(t),goal4=measures(b),interval_ticks=[s,e])


def flight_momentum(data,path):
    l=np.load(path/'rollout_locomotion_trace.npz')['trace'];t=np.load(path/'rollout_verification_trace.npz')['trace']
    flight=(l[:,14:16].sum(1)<1.)&(t[:,0]==0)
    edges=np.diff(np.r_[False,flight,False].astype(int));intervals=[]
    for s,e in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        if e-s<3:continue
        H=data['whole_system_angular_momentum'][s:e]
        intervals.append(dict(ticks=[int(s),int(e)],angular_momentum_change=summary(np.linalg.norm(H-H[0],axis=1))))
    return dict(scope='All robot and box bodies, about total COM; measured actual force-free intervals, no applied forces or fitted alignment.',intervals=intervals,
        maximum_angular_momentum_change=float(max([x['angular_momentum_change']['peak'] for x in intervals] or [0.])))


def report(out):
    m=mujoco.MjModel.from_binary_path(str(out/'execution_42/rollout_model.mjb'))
    names=['robot/waist_yaw_joint','robot/waist_roll_joint','robot/waist_pitch_joint']
    waist=m.jnt_qposadr[[m.joint(s).id for s in names]]
    arms=m.jnt_qposadr[[j for j in range(m.njnt) if any(s in m.joint(j).name for s in ['shoulder_','elbow_','wrist_'])]]
    new={str(s):prepare(out/f'execution_{s}',m) for s in [42,43]}
    old={str(s):prepare(G4/f'execution_{s}',m) for s in [42,43]}
    disabled=prepare(out/'disabled',m)
    comparisons={}
    for key,a,b in [('goal5_42_vs_43',new['42'],new['43']),('goal4_42_vs_43',old['42'],old['43']),
                    ('42_vs_disabled',new['42'],disabled),('43_vs_disabled',new['43'],disabled),
                    ('42_vs_own_goal4',new['42'],old['42']),('43_vs_own_goal4',new['43'],old['43'])]:
        comparisons[key]=dict(task_phase=compare(a,b,arms,waist,WINDOWS),active_phase=compare(a,b,arms,waist,ACTIVE))
    assessments={str(s):json.loads((out/f'execution_{s}/task_assessment.json').read_text()) for s in [42,43]}
    replay={str(s):json.loads((out/f'replay_{s}/execution.json').read_text()) for s in [42,43]}
    tracks={str(s):tracking(new[str(s)],old[str(s)],np.load(out/f'body_final_{s}_diagnostics.npz')['parameters'],waist[0]) for s in [42,43]}
    quality={str(s):physical_quality(out/f'execution_{s}',G4/f'execution_{s}') for s in [42,43]}
    retention={};expanded={}
    for phase in ACTIVE:
        a=comparisons['goal5_42_vs_43']['active_phase'][phase];b=comparisons['goal4_42_vs_43']['active_phase'][phase]
        retention[phase]={k:a[k]['rms']/b[k]['rms'] for k in ['arms_rad','object_position_m']}
        expanded[phase]=dict(root_position_rms_gain=a['root_position_m']['rms']/b['root_position_m']['rms'],
            root_orientation_rms_gain=a['root_orientation_rad']['rms']/b['root_orientation_rad']['rms'],
            torso_orientation_rms_gain=a['torso_orientation_rad']['rms']/b['torso_orientation_rad']['rms'],
            waist_yaw_rms_gain=a['waist_joints_rad']['per_joint_rms'][0]/b['waist_joints_rad']['per_joint_rms'][0],
            measurable=bool(a['root_position_m']['rms']>.0005 and a['root_orientation_rad']['rms']>.0015 and a['waist_joints_rad']['per_joint_rms'][0]>.002))
    preserved=json.loads((out/'preservation_before.json').read_text());preserved={s:sha(ROOT/s)==h for s,h in preserved.items()}
    inference={}
    for seed in [42,43,0]:
        z=np.load(out/f'inference_recheck/sample_{seed}.npz')
        original=np.load(ROOT/'logs/compact_diffusion_v1'/('calibration' if seed==0 else 'proposals')/f'sample_{seed}.npz')
        inference[str(seed)]={k:bool(np.array_equal(z[k],original[k])) for k in z.files}
    prior_disabled=np.load(G4/'disabled/rollout.npz')
    disabled_errors={k:float(abs(disabled[k]-prior_disabled[k]).max()) for k in ['qpos','qvel','ctrl_physics']}
    contacts={}
    for seed in [42,43]:
        a=np.load(out/f'execution_{seed}/contact_attribution.npz');b=np.load(out/f'replay_{seed}/contact_attribution.npz')
        contacts[str(seed)]=all(np.array_equal(a[k],b[k]) for k in a.files)
    initial=all(np.array_equal(a[k],disabled[k]) for a in new.values() for k in ['initial_qpos','initial_qvel'])
    mode_sources={str(s):json.loads((out/f'body_final_{s}.json').read_text()) for s in [42,43]}
    goals=[np.load(out/f'feasible_{s}.npz') for s in [42,43]];zero=np.load(out/'disabled.npz')
    native_joints=m.actuator_trnid[:29,0];native_q=m.jnt_qposadr[native_joints];native_v=m.jnt_dofadr[native_joints]
    actuation={};realization={}
    for seed in [42,43]:
        key=str(seed);data=new[key];baseline=old[key]
        def position_excess(q):
            return np.maximum(np.maximum(q[:,native_q]-m.jnt_range[native_joints,1],m.jnt_range[native_joints,0]-q[:,native_q]),0.).max(0)
        actuation[key]=dict(native_joint_names=[m.joint(j).name for j in native_joints],
            maximum_effort_excess_backend_N_or_Nm=assessments[key]['maximum_effort_excess_backend'],
            actual_native_velocity_peak_radps=abs(data['qvel'][:,native_v]).max(0).tolist(),
            goal4_native_velocity_peak_radps=abs(baseline['qvel'][:,native_v]).max(0).tolist(),
            actual_native_soft_stop_excursion_rad=position_excess(data['qpos']).tolist(),
            goal4_native_soft_stop_excursion_rad=position_excess(baseline['qpos']).tolist(),
            limit_contract='The accepted model has unchanged torque and soft position limits and no actuator velocity cutoff. Reference-added native-joint velocity is bounded to 0.7 rad/s. Actual velocities and soft-stop excursions are reported against Goal 4 without relaxing model parameters.')
        projected=np.load(out/f'body_projection_{seed}/rollout.npz')
        realization[key]={k:float(abs(data[k]-projected[k]).max()) for k in ['qpos','qvel','ctrl_physics']}
    quality_pass=all(d['goal5']['placement_object_linear_velocity']['peak']<1.2*d['goal4']['placement_object_linear_velocity']['peak']
        and d['goal5']['placement_object_acceleration']['rms']<1.25*d['goal4']['placement_object_acceleration']['rms'] for d in quality.values())
    grip_pass=all(a['minimum_walk_hand_force_N'][i]>.7*json.loads((G4/f'execution_{s}/task_assessment.json').read_text())['minimum_walk_hand_force_N'][i] for s,a in assessments.items() for i in [0,1])
    gates=dict(two_complete_physical_successes=all(a['status']=='SUCCESS' and len(a['numeric_gates'])==22 and all(a['numeric_gates'].values()) for a in assessments.values()),
        two_exact_independent_replays=all(all(a[k]==0 for k in EXACT) for a in replay.values()),full_contact_replay_exact=all(contacts.values()),
        identical_initial_states=initial,identical_task_goals=all(np.array_equal(g[k],zero[k]) for g in goals for k in ['goal_position','goal_quaternion']),
        disabled_exactly_reproduces_goal4_baseline=all(x==0 for x in disabled_errors.values()),
        substantial_arm_and_object_authority_retained=all(v>.8 for phase in retention.values() for v in phase.values()),
        expanded_body_motion_in_all_phases=all(p['measurable'] for p in expanded.values()),
        body_motion_clearly_above_goal4=all(p['root_position_rms_gain']>2 and p['root_orientation_rms_gain']>3 and p['waist_yaw_rms_gain']>1.25 and p['torso_orientation_rms_gain']>1.2 for p in expanded.values())
            and sum(p['waist_yaw_rms_gain']>3 and p['torso_orientation_rms_gain']>1.5 for p in expanded.values())>=2,
        body_follows_frozen_sample_target=all(d['correlation'] is not None and d['correlation']>.5 for sample in tracks.values() for phase in sample.values() for d in phase.values()),
        grasp_quality_retained=grip_pass,placement_smoothness_retained=quality_pass,
        walking_quality_retained=all(a['alternating_replants']>=9 and a['walking_root_displacement_m']>2.7 for a in assessments.values()),
        fresh_frozen_samples_exact=all(all(x.values()) for x in inference.values()),prior_files_unchanged=all(preserved.values()),
        identical_refinement_settings=all(mode_sources['42'][k]==mode_sources['43'][k] for k in ['scale','centroidal_guard_ticks','compiler_sha256']) and all(mode_sources['42']['diffusion'][k]==mode_sources['43']['diffusion'][k] for k in ['mode_gains','mode_limits','calibration_sha256','frozen_checkpoint_sha256']),
        physical_realization_retains_motion=all(d['qpos']==0 and d['qvel']==0 and d['ctrl_physics']<1e-12 for d in realization.values()),
        tracked_files_unchanged=not subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).strip())
    result=dict(status='SUCCESS' if all(gates.values()) else 'FAILURE',gates=gates,
        comparisons=comparisons,authority_expansion=expanded,arm_object_retention=retention,body_target_tracking=tracks,
        physical_assessments=assessments,placement_quality=quality,replay_errors={s:{k:r[k] for k in EXACT} for s,r in replay.items()},
        actual_actuation_and_joint_limits=actuation,physical_projection_vs_execution_max_errors=realization,
        actual_flight_momentum={str(s):dict(goal5=flight_momentum(new[str(s)],out/f'execution_{s}'),goal4=flight_momentum(old[str(s)],G4/f'execution_{s}')) for s in [42,43]},
        exact_contact_replay=contacts,disabled_baseline_errors=disabled_errors,fresh_frozen_inference_equality=inference,
        refinement=mode_sources,prior_preservation=preserved,windows=dict(task_phase=WINDOWS,active_phase=ACTIVE),
        waist_joint_names=names,arm_qpos_columns=arms.tolist(),actuator_names=[m.actuator(i).name for i in range(m.nu)],
        attribution='Same fixed samples, calibration, initial state, goals, support clock, physical model and controller. Height and yaw targets are frozen-sample innovations. Root xy and leg/arm compensation enforce constraints; these are not counted as independent diffusion inputs. No alignment is fitted to executed motion. Paired Goal 4 ablations isolate the added body contribution.',
        authority_interpretation='A 6–7/10 rating is qualitative. This evidence tests successful coordinated torso/waist and selected root motion with retained Goal 4 arm/object authority; thresholds alone do not calibrate that rating.',
        body_comparison_interpretation='Goal 4 already has executed waist/torso variation from expert feedback to arm/object changes despite fixed body references. Goal 5 is compared with that actual variation, including signed counter-rotation/cancellation. A uniform threefold increase of every body component in every phase is not required or claimed. The paired ablations and source-target tracking identify direct new body authority; all phases must increase and at least two must have strong waist/torso gains.',
        limitations=['Root height and yaw plus waist yaw are the only independent new body coordinates.',
            'Root roll/pitch and waist roll/pitch remain expert references. Root xy, legs and arms compensate deterministically for COM, soles and grasp.',
            'Body progress pauses near flight; support sequence, footsteps and gripper mechanics remain expert responsibilities.',
            'Offline differential torque and momentum screens are local approximations. Only the actual ORCS runs establish success for this fixed model/task.'],
        videos={str(s):str(out/f'execution_{s}/rollout.mp4') for s in [42,43]},
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        files_changed=[str(p.relative_to(ROOT)) for p in sorted((ROOT/'scripts/compact_diffusion_body_v1').iterdir()) if p.is_file()],
        reproduction_command='OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -B scripts/compact_diffusion_body_v1/reproduce.py --out logs/compact_diffusion_body_v1_reproduction')
    result['new_source_sha256']={s:sha(ROOT/s) for s in result['files_changed']}
    (out/'goal5_assessment.json').write_text(json.dumps(result,indent=2))
    artifacts=[p for p in out.rglob('*') if p.is_file() and p.suffix in ['.json','.npz','.mjb','.mp4','.png','.pdf'] and p.name!='goal5_manifest.json']
    (out/'goal5_manifest.json').write_text(json.dumps(dict(source_sha256=result['new_source_sha256'],artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in artifacts}),indent=2))
    print(json.dumps(dict(status=result['status'],gates=gates,expansion=expanded,retention=retention),indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();report(a.out.resolve())
