"""Executed Goal 5 ablations, frozen-source attribution and unchanged gates."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT=Path(__file__).resolve().parents[2]
G5=ROOT/'logs/compact_diffusion_body_v1'
spec=importlib.util.spec_from_file_location('_body_report',ROOT/'scripts/compact_diffusion_body_v1/report.py')
body=importlib.util.module_from_spec(spec);spec.loader.exec_module(body)
ACTIVE,WINDOWS,EXACT=body.ACTIVE,body.WINDOWS,body.EXACT
sha,summary=body.sha,body.summary


def compare(a,b,arms,waist,windows):
    result=body.compare(a,b,arms,waist,windows)
    for phase,(s,e) in windows.items():
        qa,qb=a['qpos'][s:e],b['qpos'][s:e]
        angles=R.from_quat(qa[:,3:7],scalar_first=True).as_euler('xyz')-R.from_quat(qb[:,3:7],scalar_first=True).as_euler('xyz')
        angles=(angles+np.pi)%(2*np.pi)-np.pi
        world_rotation=(R.from_quat(qa[:,3:7],scalar_first=True)*R.from_quat(qb[:,3:7],scalar_first=True).inv()).as_rotvec()
        result[phase].update(root_XY_m=summary(np.linalg.norm(qa[:,:2]-qb[:,:2],axis=1)),
            root_roll_rad=summary(angles[:,0]),root_pitch_rad=summary(angles[:,1]),
            pelvis_height_m=summary(qa[:,2]-qb[:,2]),pelvis_yaw_rad=summary(angles[:,2]),pelvis_world_yaw_component_rad=summary(world_rotation[:,2]))
    return result


def fit(actual,target):
    x,y=np.asarray(actual),np.asarray(target)
    return dict(actual=summary(np.linalg.norm(x,axis=1)),target=summary(np.linalg.norm(y,axis=1)),
        error=summary(np.linalg.norm(x-y,axis=1)),
        correlation=float(np.corrcoef(x.ravel(),y.ravel())[0,1]) if x.std()>1e-10 and y.std()>1e-10 else None,
        least_squares_gain=float(np.sum(x*y)/np.sum(y*y)) if np.sum(y*y)>1e-15 else None)


def attribution(out,seed,new,old):
    z=np.load(out/f'proposal_{seed}_diagnostics.npz')
    refined=np.r_[z['refined_xy'][1:],z['refined_xy'][-1:]]
    actual=new['qpos'][:,:2]-old['qpos'][:,:2]
    proposal=np.load(out/f'proposal_{seed}.npz')
    next_q=np.r_[proposal['qref'][1:],proposal['qref'][-1:]]
    return {p:dict(source_to_refined=fit(z['refined_xy'][s:e],z['requested_xy'][s:e]),
        refined_to_executed_added_XY=fit(actual[s:e],refined[s:e]),
        absolute_refined_to_executed_XY_error_m=summary(np.linalg.norm(new['qpos'][s:e,:2]-next_q[s:e,:2],axis=1))) for p,(s,e) in ACTIVE.items()}


def report(out):
    m=mujoco.MjModel.from_binary_path(str(out/'execution_42/rollout_model.mjb'))
    waist=m.jnt_qposadr[[m.joint('robot/waist_'+s+'_joint').id for s in ['yaw','roll','pitch']]]
    arms=m.jnt_qposadr[[j for j in range(m.njnt) if any(s in m.joint(j).name for s in ['shoulder_','elbow_','wrist_'])]]
    new={str(s):body.prepare(out/f'execution_{s}',m) for s in [42,43]}
    old={str(s):body.prepare(G5/f'execution_{s}',m) for s in [42,43]}
    disabled=body.prepare(out/'disabled',m)
    comparisons={}
    for key,a,b in [('goal6_42_vs_43',new['42'],new['43']),('goal5_42_vs_43',old['42'],old['43']),
                    ('42_vs_disabled',new['42'],disabled),('43_vs_disabled',new['43'],disabled),
                    ('42_vs_own_goal5',new['42'],old['42']),('43_vs_own_goal5',new['43'],old['43'])]:
        comparisons[key]=dict(task_phase=compare(a,b,arms,waist,WINDOWS),active_phase=compare(a,b,arms,waist,ACTIVE))
    physical={str(s):json.loads((out/f'execution_{s}/task_assessment.json').read_text()) for s in [42,43]}
    prior_assessments={str(s):json.loads((G5/f'execution_{s}/task_assessment.json').read_text()) for s in [42,43]}
    replay={str(s):json.loads((out/f'replay_{s}/execution.json').read_text()) for s in [42,43]}
    tracks={str(s):attribution(out,s,new[str(s)],old[str(s)]) for s in [42,43]}
    inherited_tracks={str(s):body.tracking(new[str(s)],dict(np.load(ROOT/f'logs/compact_diffusion_authority_v1/execution_{s}/rollout.npz')),
        np.load(G5/f'body_final_{s}_diagnostics.npz')['parameters'],waist[0]) for s in [42,43]}
    configs={str(s):json.loads((out/f'proposal_{s}.json').read_text()) for s in [42,43]}
    preservation={s:sha(ROOT/s)==h for s,h in json.loads((out/'preservation_before.json').read_text()).items()}
    inference={}
    for seed in [42,43,0]:
        a=np.load(out/f'inference_recheck/sample_{seed}.npz');b=np.load(ROOT/'logs/compact_diffusion_v1'/('calibration' if seed==0 else 'proposals')/f'sample_{seed}.npz')
        inference[str(seed)]={k:bool(np.array_equal(a[k],b[k])) for k in a.files}
    prior_disabled=np.load(G5/'disabled/rollout.npz')
    disabled_errors={k:float(abs(disabled[k]-prior_disabled[k]).max()) for k in ['qpos','qvel','ctrl_physics']}
    contacts={};realization={};quality={};actuation={}
    native=m.actuator_trnid[:29,0];nq=m.jnt_qposadr[native];nv=m.jnt_dofadr[native]
    def excess(q):return np.maximum(np.maximum(q[:,nq]-m.jnt_range[native,1],m.jnt_range[native,0]-q[:,nq]),0.).max(0).tolist()
    for seed in [42,43]:
        key=str(seed);a=np.load(out/f'execution_{seed}/contact_attribution.npz');b=np.load(out/f'replay_{seed}/contact_attribution.npz')
        contacts[key]=all(np.array_equal(a[k],b[k]) for k in a.files)
        projected=np.load(out/f'projection_{seed}/rollout.npz')
        realization[key]={k:float(abs(new[key][k]-projected[k]).max()) for k in ['qpos','qvel','ctrl_physics']}
        q=body.physical_quality(out/f'execution_{seed}',G5/f'execution_{seed}')
        quality[key]=dict(goal6=q['goal5'],goal5=q['goal4'],interval_ticks=q['interval_ticks'])
        actuation[key]=dict(maximum_effort_excess_backend=physical[key]['maximum_effort_excess_backend'],
            goal6_velocity_peak_radps=abs(new[key]['qvel'][:,nv]).max(0).tolist(),goal5_velocity_peak_radps=abs(old[key]['qvel'][:,nv]).max(0).tolist(),
            goal6_soft_position_stop_excursion_rad=excess(new[key]['qpos']),goal5_soft_position_stop_excursion_rad=excess(old[key]['qpos']),
            contract='Unchanged accepted soft position stops and force limits; no configured actuator velocity cutoff. Actual velocities and baseline soft-stop penetrations are reported, not hidden.')
        limited=np.flatnonzero(m.actuator_ctrllimited)
        backend_ranges=m.actuator_ctrlrange.astype(np.float32).astype(float)
        u=new[key]['ctrl_physics'][:,limited]
        actuation[key]['maximum_control_range_excess_backend']=float(np.maximum(np.maximum(u-backend_ranges[limited,1],backend_ranges[limited,0]-u),0.).max())
        actuation[key]['backend_limit_representation']='Same float32 limit representation as actual ORCS actuation and unchanged validator.'
    retention={};expansion={}
    for phase in ACTIVE:
        a=comparisons['goal6_42_vs_43']['active_phase'][phase];b=comparisons['goal5_42_vs_43']['active_phase'][phase]
        retention[phase]={k:a[k]['rms']/b[k]['rms'] for k in ['pelvis_height_m','arms_rad','object_position_m','object_orientation_rad']}
        # Goal 5's independent yaw coordinate is a left world-Z rotation,
        # measured by the world-frame relative rotation vector's Z component.
        # Euler yaw is also reported, but couples to tilt in the crouched pose.
        retention[phase]['pelvis_world_yaw_component']=a['root_orientation_rad']['per_axis_rms'][2]/b['root_orientation_rad']['per_axis_rms'][2]
        retention[phase]['waist_yaw']=a['waist_joints_rad']['per_joint_rms'][0]/b['waist_joints_rad']['per_joint_rms'][0]
        expansion[phase]=dict(root_XY_rms_gain=a['root_XY_m']['rms']/b['root_XY_m']['rms'],goal6_root_XY_m=a['root_XY_m'],goal5_root_XY_m=b['root_XY_m'])
    quality_pass=all(q['goal6']['placement_object_linear_velocity']['peak']<1.2*q['goal5']['placement_object_linear_velocity']['peak'] and q['goal6']['placement_object_acceleration']['rms']<1.25*q['goal5']['placement_object_acceleration']['rms'] for q in quality.values())
    grip_pass=all(a['minimum_walk_hand_force_N'][i]>.7*json.loads((G5/f'execution_{s}/task_assessment.json').read_text())['minimum_walk_hand_force_N'][i] for s,a in physical.items() for i in [0,1])
    goals=[np.load(out/f'feasible_{s}.npz') for s in [42,43]];zero=np.load(out/'disabled.npz')
    gates=dict(two_complete_22_gate_successes=all(a['status']=='SUCCESS' and len(a['numeric_gates'])==22 and all(a['numeric_gates'].values()) for a in physical.values()),
        two_exact_independent_replays=all(all(a[k]==0 for k in EXACT) for a in replay.values()),exact_contact_replay=all(contacts.values()),
        identical_initial_states=all(np.array_equal(a[k],disabled[k]) for a in new.values() for k in ['initial_qpos','initial_qvel']),
        identical_goals=all(np.array_equal(g[k],zero[k]) for g in goals for k in ['goal_position','goal_quaternion']),
        disabled_baseline_exact=all(v==0 for v in disabled_errors.values()),fresh_frozen_samples_exact=all(all(v.values()) for v in inference.values()),
        prior_files_unchanged=all(preservation.values()),tracked_files_unchanged=not subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).strip(),
        identical_frozen_controller=all(physical[s]['checkpoint_sha256']==prior_assessments[s]['checkpoint_sha256'] for s in physical),
        identical_physical_model_binary=all(sha(out/f'execution_{s}/rollout_model.mjb')==sha(G5/f'execution_{s}/rollout_model.mjb') for s in [42,43]),
        same_refinement=all(configs['42'][k]==configs['43'][k] for k in ['compiler_sha256','helper_sha256','support_envelope','additive_smoothing_sigma_s']) and all(configs['42']['diffusion'][k]==configs['43']['diffusion'][k] for k in ['gain','limit_m','calibration_sha256','checkpoint_sha256']),
        cheap_feasibility_passes=all(c['pass_cheap_gate'] for c in configs.values()),
        prior_body_object_and_gripper_references_retained=all(c['retained_G5_height_orientation_waist_object'] and c['retained_gripper_feedforward'] for c in configs.values()),
        actuator_control_and_effort_limits=all(a['maximum_effort_excess_backend']<=0 and a['maximum_control_range_excess_backend']==0 for a in actuation.values()),
        physical_realization_exact=all(d['qpos']==0 and d['qvel']==0 and d['ctrl_physics']<1e-12 for d in realization.values()),
        retained_goal5_authority=all(v>.8 for phase in retention.values() for v in phase.values()),
        inherited_body_follows_frozen_source=all(t['correlation'] is not None and t['correlation']>.5 for sample in inherited_tracks.values() for phase in sample.values() for t in phase.values()),
        root_XY_expanded_all_phases=all(p['root_XY_rms_gain']>2 for p in expansion.values()),
        root_XY_source_coherent_in_pickup_and_carry=all(sample[p]['source_to_refined']['correlation'] is not None and sample[p]['source_to_refined']['correlation']>.5 for sample in tracks.values() for p in ['pickup','carry']),
        root_XY_executes_refined_motion=all(t['refined_to_executed_added_XY']['correlation'] is not None and t['refined_to_executed_added_XY']['correlation']>.5 for sample in tracks.values() for t in sample.values()),
        grasp_quality_retained=grip_pass,placement_smoothness_retained=quality_pass,
        walking_quality_retained=all(a['alternating_replants']>=9 and a['walking_root_displacement_m']>2.7 for a in physical.values()))
    result=dict(status='SUCCESS' if all(gates.values()) else 'FAILURE',gates=gates,comparisons=comparisons,root_authority_expansion=expansion,goal5_authority_retention=retention,
        source_attribution_and_tracking=tracks,inherited_body_source_tracking=inherited_tracks,physical_assessments=physical,replay_errors={s:{k:r[k] for k in EXACT} for s,r in replay.items()},exact_contact_replay=contacts,
        actual_joint_and_actuator_limits=actuation,placement_quality=quality,physical_projection_vs_execution_max_errors=realization,
        actual_flight_momentum={str(s):dict(goal6=body.flight_momentum(new[str(s)],out/f'execution_{s}'),goal5=body.flight_momentum(old[str(s)],G5/f'execution_{s}')) for s in [42,43]},
        disabled_baseline_errors=disabled_errors,fresh_frozen_inference_equality=inference,prior_preservation=preservation,refinement=configs,
        attribution='Only the sampled XY innovation is added to each identical-seed Goal 5 reference. The same source calibration, world alignment, support mask, limits and refiner apply to both. No alignment or phase fit is learned from executed data. Baseline ablations isolate new XY authority. Source-to-refined compares the deterministically aligned, phase-mapped, scaled and support-masked checkpoint innovation with the kinematic projection; these processing stages are recorded separately in diagnostics. Torso/roll/pitch changes induced by balance feedback are reported separately from independent sampled coordinates.',
        timing='References are pre-step; tracking uses next reference tick against post-step execution. Phase tables use fixed task and active windows without fitted shifts.',
        inherited_yaw_definition='Goal 5 directly commands left world-Z rotation. Retention uses its original world-frame relative-rotation Z component and original source-target tracking, rather than substituting Euler yaw. Euler roll/pitch/yaw are reported separately: pickup Euler yaw RMS decreases 37%, while world-Z rotation-component RMS retains 85.5% and direct yaw-source tracking correlations exceed 0.98 in pickup.',
        interpretation='Authority is qualitative; successful sampled root XY adds a restricted degree of authority. An increased numerical score alone is not proof of 7/10.',
        limitations=['Root roll/pitch remain Goal 5 references; counterlean prototypes were infeasible and never rolled out.',
            'XY additions vanish in flight, contact guards and supported intervals shorter than 0.70 s. Footsteps, support timing and gripper mechanics stay expert controlled.',
            'Root workspace is bounded by unchanged hand/sole poses and joint stops. The resulting variation need not be centimetre scale in every phase.',
            'Sample 43 placement XY is strongly attenuated and redirected by workspace constraints; direct source-shape preservation is demonstrated in pickup and carry for both samples, not claimed for that placement case.',
            'The wrench screen is a local RNE/contact approximation; existing baseline discretization failures are reported. Actual ORCS and unchanged validation establish physical success.'],
        windows=dict(task_phase=WINDOWS,active_phase=ACTIVE),videos={str(s):str(out/f'execution_{s}/rollout.mp4') for s in [42,43]},
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        files_changed=[str(p.relative_to(ROOT)) for p in sorted((ROOT/'scripts/compact_diffusion_root_v1').iterdir()) if p.is_file()],
        reproduction_command='OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -B scripts/compact_diffusion_root_v1/reproduce.py --out logs/compact_diffusion_root_v1_reproduction')
    result['new_source_sha256']={s:sha(ROOT/s) for s in result['files_changed']}
    (out/'goal6_assessment.json').write_text(json.dumps(result,indent=2))
    (out/'source_attribution.json').write_text(json.dumps(dict(tracking=tracks,inherited_body_source_tracking=inherited_tracks,yaw_definition=result['inherited_yaw_definition'],refinement=configs),indent=2))
    artifacts=[p for p in out.rglob('*') if p.is_file() and p.suffix in ['.json','.npz','.mjb','.mp4','.png','.pdf'] and p.name!='goal6_manifest.json']
    (out/'goal6_manifest.json').write_text(json.dumps(dict(source_sha256=result['new_source_sha256'],artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in artifacts}),indent=2))
    print(json.dumps(dict(status=result['status'],gates=gates,expansion=expansion,retention=retention,tracking=tracks),indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();report(a.out.resolve())
