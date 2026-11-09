"""Assess the full-source pickup separately from the unmet 8/10 objective."""
import argparse
import csv
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from report import ROOT,G6,g6,sha,summary,fit,authority_proxy,compare_quality,quality,EXACT
from primary_targets import G7


def components(q,base,arms,waist):
    angle=(R.from_quat(q[:,3:7],scalar_first=True)*R.from_quat(base[:,3:7],scalar_first=True).inv()).as_rotvec()
    obj=(R.from_quat(q[:,47:51],scalar_first=True)*R.from_quat(base[:,47:51],scalar_first=True).inv()).as_rotvec()
    return dict(root_XY_m=q[:,:2]-base[:,:2],root_horizontal_rotation_rad=angle[:,:2],
        pelvis_height_m=q[:,2:3]-base[:,2:3],pelvis_world_yaw_rad=angle[:,2:3],
        waist_rad=q[:,waist]-base[:,waist],arms_rad=q[:,arms]-base[:,arms],
        object_position_m=q[:,44:47]-base[:,44:47],object_orientation_rad=obj)


def report(out,projection=False):
    parent=G7;group='pickup_projection' if projection else 'pickup_execution'
    m=mujoco.MjModel.from_binary_path(str(out/f'{group}_42/rollout_model.mjb'))
    arms=m.jnt_qposadr[[j for j in range(m.njnt) if any(s in m.joint(j).name for s in ['shoulder_','elbow_','wrist_'])]]
    waist=m.jnt_qposadr[[m.joint('robot/waist_'+s+'_joint').id for s in ['yaw','roll','pitch']]]
    new={str(s):g6.body.prepare(out/f'{group}_{s}',m) for s in [42,43]}
    old={str(s):g6.body.prepare(G6/f'execution_{s}',m) for s in [42,43]}
    expert=g6.body.prepare(G6/'disabled',m)
    comparisons={}
    for key,a,b in [('goal7_42_vs_43',new['42'],new['43']),('goal6_42_vs_43',old['42'],old['43']),
        ('goal7_42_vs_disabled',new['42'],expert),('goal7_43_vs_disabled',new['43'],expert),
        ('goal6_42_vs_disabled',old['42'],expert),('goal6_43_vs_disabled',old['43'],expert),
        ('goal7_42_vs_own_goal6',new['42'],old['42']),('goal7_43_vs_own_goal6',new['43'],old['43'])]:
        comparisons[key]={name:g6.compare(a,b,arms,waist,window) for name,window in [('active_phase',g6.ACTIVE),('task_phase',g6.WINDOWS)]}
    qualities={str(s):compare_quality(quality(out/f'{group}_{s}',m),quality(G6/f'execution_{s}',m)) for s in [42,43]}
    lift_quality={}
    for seed in [42,43]:
        row={}
        for name,path in [('goal7',out/f'{group}_{seed}'),('goal6',G6/f'execution_{seed}')]:
            trace=np.load(path/'rollout_verification_trace.npz')['trace'];v=trace[641:1601,20:23];a=np.gradient(v,.005,axis=0)
            row[name]=dict(box_speed_mps=summary(np.linalg.norm(v,axis=1)),box_acceleration_mps2=summary(np.linalg.norm(a,axis=1)))
        lift_quality[str(seed)]=row
    retention={}
    for phase in g6.ACTIVE:
        a=comparisons['goal7_42_vs_43']['active_phase'][phase];b=comparisons['goal6_42_vs_43']['active_phase'][phase]
        retention[phase]={k:a[k]['rms']/b[k]['rms'] for k in ['root_XY_m','pelvis_height_m','arms_rad','object_position_m','object_orientation_rad']}
        retention[phase]['waist_yaw']=a['waist_joints_rad']['per_joint_rms'][0]/b['waist_joints_rad']['per_joint_rms'][0]
        retention[phase]['pelvis_world_yaw']=a['pelvis_world_yaw_component_rad']['rms']/b['pelvis_world_yaw_component_rad']['rms']
    tracks={}
    for seed in [42,43]:
        key=str(seed);base=np.load(parent/f'feasible_{seed}.npz')['qref']
        diagnostic=out/f'pickup_{seed}_diagnostics.npz'
        if seed==42 and not diagnostic.exists():diagnostic=out/'corrected_42_diagnostics.npz'
        target=np.load(diagnostic)['qtarget']
        reference=np.load(out/f'pickup_{seed}.npz')['qref']
        src=components(target,base,arms,waist);ref=components(reference,base,arms,waist)
        actual=components(new[key]['qpos'],np.r_[base[1:],base[-1:]],arms,waist)
        s,e=641,1601
        tracks[key]=dict(full_source_pickup={name:dict(source_to_refined=fit(ref[name][s:e],src[name][s:e]),
            refined_to_executed=fit(actual[name][s:e],np.r_[ref[name][1:],ref[name][-1:]][s:e])) for name in src},
            retained_carry_and_placement_attribution=str(parent/'source_attribution.json'))
    original=json.loads((parent/'goal7_assessment.json').read_text())
    preserve={p:sha(ROOT/p)==h for p,h in json.loads((parent/'preservation_before.json').read_text()).items()}
    unrelated={p:sha(ROOT/p)==h for p,h in json.loads((parent/'unrelated_untracked_hashes.json').read_text()).items()}
    physical={str(s):json.loads((out/f'pickup_projection_{s}/physical_screen.json').read_text()) for s in [42,43]}
    configs={str(s):json.loads((out/f'pickup_{s}.json').read_text()) for s in [42,43]}
    refs={str(s):np.load(out/f'pickup_{s}.npz') for s in [42,43]}
    executions={str(s):json.loads((out/f'{group}_{s}/execution.json').read_text()) for s in [42,43]}
    baseline_goal=np.load(G6/'disabled.npz')
    gates=dict(two_complete_physical_screens=all(a['pass_physical_screen'] for a in physical.values()),
        same_initial_conditions=all(np.array_equal(a[k],expert[k]) for a in new.values() for k in ['initial_qpos','initial_qvel']),
        same_physical_model=all(sha(out/f'{group}_{s}/rollout_model.mjb')==sha(G6/f'execution_{s}/rollout_model.mjb') for s in [42,43]),
        walking_grasp_placement_quality_retained=all(a['pass_quality'] for a in qualities.values()),
        preserved_accepted_files=all(preserve.values()),preserved_unrelated_untracked_files=all(unrelated.values()),
        no_tracked_diff=not subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).strip(),
        no_index_diff=not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip(),
        strongest_incremental_source_unchanged=all(sha(ROOT/p)==h for p,h in original['source_sha256'].items()),
        common_compiler_and_helpers=all(configs['42'][k]==configs['43'][k] for k in ['compiler_sha256','helper_sha256']),
        common_refinement_options=configs['42']['source'].get('projection_options')==configs['43']['source'].get('projection_options'),
        same_reference_precision=all(refs[s][k].dtype==np.dtype('float64') for s in refs for k in ['qref','vref','uref']),
        prior_sample_dependent_authority_retained=all(value>.8 for phase in retention.values() for value in phase.values()),
        same_task_goals=all(np.array_equal(refs[s][k],baseline_goal[k]) for s in refs for k in ['goal_position','goal_quaternion']),
        same_frozen_controller=all(executions[str(s)]['checkpoint_sha256']==json.loads((G6/f'execution_{s}/execution.json').read_text())['checkpoint_sha256'] for s in [42,43]),
        preserved_gripper_control_references=all(np.array_equal(refs[str(s)]['uref'][:,29:],np.load(G7/f'feasible_{s}.npz')['uref'][:,29:]) for s in [42,43]))
    rejection_files=[p for p in sorted((G7/'primary').glob('*.json')) if p.stem in ['corrected_42','continuous_placement_42','correction_placement_42','fixed_motion_42','fixed_angles_42']]
    rejection_fields=['pass_cheap_gate','maximum_contact_position_error_m','maximum_contact_angle_error_rad','geometry_failure_count',
        'maximum_predicted_actuator_force_excess_Nm','maximum_control_range_excess','maximum_added_native_velocity_radps','maximum_control_increment_rad','simulator_stepped']
    rejections={}
    for path in rejection_files:
        record=json.loads(path.read_text());rejections[path.stem]=dict(path=str(path),sha256=sha(path),introduced_dynamic_infeasibilities=len(record.get('introduced_dynamic_infeasibilities',[])),**{k:record[k] for k in rejection_fields if k in record})
    result=dict(status='FAILURE',eight_of_ten_achieved=False,verification_stage='projection' if projection else 'final',
        reason='Full native frozen proposals now supply the pickup interior, but source-primary carry/placement fail geometry or loaded-dynamics checks. They retain the successful Goal 6 backbone plus the verified incremental tilt. This is substantial pickup progress, not established diffusion dominance across all phases.',
        physical_screens=physical,comparisons=comparisons,quality=qualities,postgrasp_lift_quality=lift_quality,source_attribution_and_tracking=tracks,prior_authority_retention=retention,
        authority_proxy={str(s):authority_proxy(new[str(s)],expert,arms,waist) for s in [42,43]},
        authority_proxy_contract=original['authority_proxy_contract'],candidate_gates=gates,
        frozen_checkpoint=original['fixed_checkpoint_provenance'],preservation=preserve,unrelated_preservation=unrelated,
        goal6_to_goal7_contract='Identical fixed physical initial state, controller/checkpoint/model, task goal, contact/foot schedule and refinement settings. No execution-fitted spatial or temporal alignment. RMS magnitude is not an authority percentage.',
        source_contract='Pickup: event-aligned full native sample curves with task endpoints and necessary contact projection, plus prior sample-derived motions. Carry/placement: prior verified source-derived contributions. Both samples use the same deterministic refinement. Raw source, requested, refined and actual states remain distinct.',
        compiler_records=configs,
        source_primary_rejections=rejections,
        expert_dependencies=['Footstep locations and support/contact schedule','Compact-gripper actuation and contact/release mechanics',
            'Accepted carry and placement task-scale reference backbone','Task endpoint anchors and frozen feedback stabilization'],
        limitations=['No verified source-primary carry or placement.','Quarter-scale tilt avoids unavailable support moments; no forced flight/contact-transition body tilt.',
            'Full-proposal carry/placement introduces unavailable external moments, actuator/velocity excess, or cannot meet the fixed sole/grasp/workspace constraints.',
            'Hard actuator limits are unchanged. Compliant joint stops and the absence of configured velocity cutoffs are reported separately from actuation violations.'],
        windows=dict(active_phase=g6.ACTIVE,task_phase=g6.WINDOWS),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    if not projection:
        assessments={str(s):json.loads((out/f'pickup_execution_{s}/task_assessment.json').read_text()) for s in [42,43]}
        replays={str(s):json.loads((out/f'pickup_replay_{s}/execution.json').read_text()) for s in [42,43]}
        gates['two_unchanged_22_gate_successes']=all(a['status']=='SUCCESS' and len(a['numeric_gates'])==22 and all(a['numeric_gates'].values()) for a in assessments.values())
        gates['two_independent_exact_replays']=all(all(r[k]==0 for k in EXACT) for r in replays.values())
        gates['zero_actuator_effort_excess']=all(a['maximum_effort_excess_backend']<=0 for a in assessments.values())
        limits=m.actuator_ctrlrange.astype(np.float32).astype(float);limited=np.flatnonzero(m.actuator_ctrllimited)
        gates['zero_actuator_control_excess']=all(np.maximum(np.maximum(a['ctrl_physics'][:,limited]-limits[limited,1],limits[limited,0]-a['ctrl_physics'][:,limited]),0).max()==0 for a in new.values())
        realization={};contacts={};actuation={}
        native=m.actuator_trnid[:29,0];nq=m.jnt_qposadr[native];nv=m.jnt_dofadr[native]
        def excursion(q):return np.maximum(np.maximum(q[:,nq]-m.jnt_range[native,1],m.jnt_range[native,0]-q[:,nq]),0).max(0).tolist()
        for seed in [42,43]:
            key=str(seed);p=np.load(out/f'pickup_projection_{seed}/rollout.npz')
            realization[key]={k:float(abs(new[key][k]-p[k]).max()) for k in ['qpos','qvel','ctrl_physics']}
            a=np.load(out/f'pickup_execution_{seed}/contact_attribution.npz');b=np.load(out/f'pickup_replay_{seed}/contact_attribution.npz');contacts[key]=all(np.array_equal(a[k],b[k]) for k in a.files)
            actuation[key]=dict(native_joint_names=[m.joint(j).name for j in native],
                maximum_effort_excess_backend=assessments[key]['maximum_effort_excess_backend'],
                goal7_native_velocity_peak_radps=abs(new[key]['qvel'][:,nv]).max(0).tolist(),goal6_native_velocity_peak_radps=abs(old[key]['qvel'][:,nv]).max(0).tolist(),
                goal7_compliant_position_stop_excursion_rad=excursion(new[key]['qpos']),goal6_compliant_position_stop_excursion_rad=excursion(old[key]['qpos']),
                contract='Unchanged actual hard actuator command/force caps. No actuator velocity cutoff is configured in this model. Existing compliant joint-stop excursions are reported separately; they are not removed by changing the model.')
        gates['physical_projection_retained']=all(a['qpos']==a['qvel']==0 and a['ctrl_physics']<1e-12 for a in realization.values())
        gates['exact_contact_replay']=all(contacts.values())
        result.update(physical_assessments=assessments,replay_errors={s:{k:a[k] for k in EXACT} for s,a in replays.items()},physical_projection_vs_execution=realization,actuation=actuation,
            videos={str(s):str(out/f'pickup_execution_{s}/rollout.mp4') for s in [42,43]},
            reproduction_command='OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -B scripts/compact_diffusion_pose_v1/primary_reproduce.py --out logs/compact_diffusion_pose_v1_pickup_reproduction')
    result['candidate_status']='SUCCESS' if all(gates.values()) else 'FAILURE'
    result['source_sha256']={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'scripts/compact_diffusion_pose_v1').iterdir()) if p.is_file()}
    name='pickup_projection_assessment.json' if projection else 'pickup_assessment.json'
    (out/name).write_text(json.dumps(result,indent=2))
    (out/'pickup_source_attribution.json').write_text(json.dumps(dict(tracking=tracks,comparisons=comparisons,authority_proxy=result['authority_proxy'],contract=result['authority_proxy_contract']),indent=2))
    with (out/'pickup_comparison.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['phase','component','goal6_42_RMS','goal6_42_peak','goal7_42_RMS','goal7_42_peak','goal6_43_RMS','goal6_43_peak','goal7_43_RMS','goal7_43_peak'])
        for phase in g6.ACTIVE:
            for component in ['root_XY_m','root_roll_rad','root_pitch_rad','pelvis_height_m','pelvis_yaw_rad','waist_joints_rad','arms_rad','object_position_m','object_orientation_rad','actuator_commands']:
                values=[]
                for seed in ['42','43']:
                    for goal in ['goal6','goal7']:
                        v=comparisons[f'{goal}_{seed}_vs_disabled']['active_phase'][phase][component];values.extend([v['rms'],v['peak']])
                writer.writerow([phase,component,*values])
    print(json.dumps(dict(status=result['status'],candidate_status=result['candidate_status'],gates=gates),indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--projection',action='store_true');a=p.parse_args();report(a.out.resolve(),a.projection)
