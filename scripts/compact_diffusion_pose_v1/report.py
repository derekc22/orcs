"""Goal 7 evidence: physical validity and 8/10 authority are separate claims."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from quality import quality,compare as compare_quality

ROOT=Path(__file__).resolve().parents[2];G6=ROOT/'logs/compact_diffusion_root_v1'
spec=importlib.util.spec_from_file_location('_goal6_report',ROOT/'scripts/compact_diffusion_root_v1/report.py')
g6=importlib.util.module_from_spec(spec);spec.loader.exec_module(g6)
ACTIVE,WINDOWS,EXACT=g6.ACTIVE,g6.WINDOWS,g6.EXACT
sha,summary,fit=g6.sha,g6.summary,g6.fit


def tracking(out,seed,new,old):
    z=np.load(out/f'safe_{seed}_diagnostics.npz');p=z['parameters'];target=np.r_[p[1:],p[-1:]]
    source=z['requested_modes'];wi=z['waist_variable_indices']
    delta=(R.from_quat(new['qpos'][:,3:7],scalar_first=True)*R.from_quat(old['qpos'][:,3:7],scalar_first=True).inv()).as_rotvec()[:,:2]
    waist=new['qpos'][:,20:22]-old['qpos'][:,20:22]
    result={phase:dict(root_source_to_refined=fit(p[s:e,:2],source[s:e,:2]),waist_source_to_refined=fit(p[s:e][:,wi],source[s:e,2:]),
        root_refined_to_executed=fit(delta[s:e],target[s:e,:2]),waist_refined_to_executed=fit(waist[s:e],target[s:e][:,wi])) for phase,(s,e) in ACTIVE.items()}
    for phase,(s,e) in ACTIVE.items():
        result[phase]['per_axis']={}
        for ax,name in enumerate(['root_world_horizontal_X','root_world_horizontal_Y']):
            result[phase]['per_axis'][name]=dict(source_to_refined=fit(p[s:e,ax:ax+1],source[s:e,ax:ax+1]),refined_to_executed=fit(delta[s:e,ax:ax+1],target[s:e,ax:ax+1]))
        for ax,name in enumerate(['waist_roll','waist_pitch']):
            j=wi[ax]
            result[phase]['per_axis'][name]=dict(source_to_refined=fit(p[s:e,j:j+1],source[s:e,ax+2:ax+3]),refined_to_executed=fit(waist[s:e,ax:ax+1],target[s:e,j:j+1]))
    return result


def authority_proxy(new,expert,arms,waist):
    result={}
    for phase,(s,e) in ACTIVE.items():
        def coords(data):
            q=data['qpos'][s:e];orientation=(R.from_quat(q[:,3:7],scalar_first=True)*R.from_quat(q[:1,3:7],scalar_first=True).inv()).as_rotvec()
            obj=(R.from_quat(q[:,47:51],scalar_first=True)*R.from_quat(q[:1,47:51],scalar_first=True).inv()).as_rotvec()
            return dict(root_XY=q[:,:2],root_orientation=orientation,pelvis_height=q[:,2:3],waist=q[:,waist],arms=q[:,arms],object_position=q[:,44:47],object_orientation=obj,native_controls=data['ctrl_physics'][s:e,:29])
        n,b=coords(new),coords(expert);u=np.linspace(0,1,e-s)[:,None]
        values={}
        for key in n:
            ec=b[key]-(1-u)*b[key][0]-u*b[key][-1];dc=n[key]-b[key];dc=dc-(1-u)*dc[0]-u*dc[-1]
            er=float(np.sqrt(np.mean(ec**2)));dr=float(np.sqrt(np.mean(dc**2)))
            values[key]=dict(expert_curvature_rms=er,diffusion_intervention_curvature_rms=dr,intervention_to_expert_rms_ratio=dr/er if er>1e-12 else None)
        result[phase]=values
    return result


def report(out,screen=False):
    group='projection' if screen else 'execution';m=mujoco.MjModel.from_binary_path(str(out/f'{group}_42/rollout_model.mjb'))
    waist=m.jnt_qposadr[[m.joint('robot/waist_'+s+'_joint').id for s in ['yaw','roll','pitch']]]
    arms=m.jnt_qposadr[[j for j in range(m.njnt) if any(s in m.joint(j).name for s in ['shoulder_','elbow_','wrist_'])]]
    new={str(s):g6.body.prepare(out/f'{group}_{s}',m) for s in [42,43]};old={str(s):g6.body.prepare(G6/f'execution_{s}',m) for s in [42,43]};disabled=g6.body.prepare(out/'disabled',m)
    comparisons={}
    for key,a,b in [('goal7_42_vs_43',new['42'],new['43']),('goal6_42_vs_43',old['42'],old['43']),('42_vs_disabled',new['42'],disabled),('43_vs_disabled',new['43'],disabled),('42_vs_own_goal6',new['42'],old['42']),('43_vs_own_goal6',new['43'],old['43'])]:
        comparisons[key]=dict(active_phase=g6.compare(a,b,arms,waist,ACTIVE),task_phase=g6.compare(a,b,arms,waist,WINDOWS))
    retention={}
    for phase in ACTIVE:
        a=comparisons['goal7_42_vs_43']['active_phase'][phase];b=comparisons['goal6_42_vs_43']['active_phase'][phase]
        retention[phase]={k:a[k]['rms']/b[k]['rms'] for k in ['root_XY_m','pelvis_height_m','pelvis_world_yaw_component_rad','arms_rad','object_position_m','object_orientation_rad']}
        retention[phase]['waist_yaw']=a['waist_joints_rad']['per_joint_rms'][0]/b['waist_joints_rad']['per_joint_rms'][0]
    qualities={str(s):compare_quality(quality(out/f'{group}_{s}',m),quality(G6/f'execution_{s}',m)) for s in [42,43]}
    tracks={str(s):tracking(out,s,new[str(s)],old[str(s)]) for s in [42,43]}
    configs={str(s):json.loads((out/f'safe_{s}.json').read_text()) for s in [42,43]}
    physical_screens={str(s):json.loads((out/f'projection_{s}/physical_screen.json').read_text()) for s in [42,43]}
    baseline=np.load(G6/'disabled/rollout.npz');disabled_errors={k:float(abs(disabled[k]-baseline[k]).max()) for k in ['qpos','qvel','ctrl_physics']}
    preservation={p:sha(ROOT/p)==h for p,h in json.loads((out/'preservation_before.json').read_text()).items()}
    gates=dict(two_complete_physical_screens=all(p['pass_physical_screen'] for p in physical_screens.values()),
        identical_initial_states=all(np.array_equal(n[k],disabled[k]) for n in new.values() for k in ['initial_qpos','initial_qvel']),
        disabled_matches_goal6_exactly=all(v==0 for v in disabled_errors.values()),
        prior_sources_and_baselines_unchanged=all(preservation.values()),walking_grasp_and_placement_quality_retained=all(q['pass_quality'] for q in qualities.values()),
        previous_authority_retained=all(v>.8 for p in retention.values() for v in p.values()),
        same_refinement=all(configs['42'][k]==configs['43'][k] for k in ['scale','compiler_sha256','support_envelope','smoothing_sigma_s','added_root_acceleration_limit_radps2']),
        both_cheap_screens_pass=all(c['pass_cheap_gate'] for c in configs.values()),
        same_physical_model=all(sha(out/f'{group}_{s}/rollout_model.mjb')==sha(G6/f'execution_{s}/rollout_model.mjb') for s in [42,43]),
        no_changes_to_tracked_code=not subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).strip())
    result=dict(status='PROVISIONAL' if screen else 'FAILURE',eight_of_ten_achieved=False,
        reason='The bounded candidate adds sampled root tilt and coordinated waist motion to an accepted reference backbone. It does not establish diffusion as the primary source of the permitted robot/object trajectories, so physical success alone cannot justify 8/10.',
        candidate_gates=gates,comparisons=comparisons,previous_authority_retention=retention,quality=qualities,source_attribution_and_tracking=tracks,
        authority_proxy={str(s):authority_proxy(new[str(s)],disabled,arms,waist) for s in [42,43]},
        authority_proxy_contract='Counterfactual added-motion curvature relative to the diffusion-disabled expert, with each fixed phase endpoint bridge removed. These non-orthogonal RMS ratios describe motion magnitude; they are not causal percentages or a calibrated authority score.',
        raw_to_reference_contract='Cached August checkpoint root/waist samples minus the same seed-0 calibration, fixed world alignment, fixed support clock, fixed common gains and bounds. Native and time-mapped innovations, requested supported modes, and both projection stages are saved without execution-fitted alignment.',
        refinement=configs,physical_screens=physical_screens,disabled_baseline_errors=disabled_errors,preservation=preservation,
        fixed_checkpoint_provenance=json.loads((G6/'inference_recheck/provenance.json').read_text()),
        prior_exact_frozen_inference_equality=json.loads((G6/'goal6_assessment.json').read_text())['fresh_frozen_inference_equality'],
        independent_inputs=['root world-horizontal relative rotation','waist roll/pitch'],
        expert_dependencies=['Footstep locations and support/contact timing','Compact-gripper mechanics and opening/release sequence','Accepted task-scale reference backbone and physical feasible region','Necessary balance and contact stabilization by the unchanged learned feedback controller'],
        limitations=['The full-amplitude sample 43 tilt requires an unavailable support moment at 19.6 s; an identical 0.25 source scale is used for both samples.',
            'Adding strict native acceleration boxes produced incompatible moving workspace constraints and was rejected before physical execution.',
            'Carry waist roll/pitch is redirected into balance compensation, with negative source-to-refined correlations for both samples. Execution of that compensation does not establish independent diffusion-primary waist-tilt control.',
            'Added tilt is zero during flight, contact guards and short support bouts. Root XYZ, waist yaw, object and gripper references remain Goal 6 inputs.',
            'This is an incremental constrained tilt extension, not a verified primary-source replacement. No 8/10 claim is made.'],
        windows=dict(active_phase=ACTIVE,task_phase=WINDOWS),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    if screen:
        result['pass_projection_screen']=all(gates.values());(out/'projection_assessment.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(dict(pass_projection_screen=result['pass_projection_screen'],gates=gates,retention=retention,quality_gates={s:q['gates'] for s,q in qualities.items()}),indent=2));return result
    assessments={str(s):json.loads((out/f'execution_{s}/task_assessment.json').read_text()) for s in [42,43]};replays={str(s):json.loads((out/f'replay_{s}/execution.json').read_text()) for s in [42,43]}
    goals=[np.load(out/f'feasible_{s}.npz') for s in [42,43]];zero=np.load(out/'disabled.npz');limits=m.actuator_ctrlrange.astype(np.float32).astype(float);limited=np.flatnonzero(m.actuator_ctrllimited)
    gates.update(two_unchanged_22_gate_successes=all(a['status']=='SUCCESS' and len(a['numeric_gates'])==22 and all(a['numeric_gates'].values()) for a in assessments.values()),
        two_independent_exact_replays=all(all(r[k]==0 for k in EXACT) for r in replays.values()),identical_task_goals=all(np.array_equal(g[k],zero[k]) for g in goals for k in ['goal_position','goal_quaternion']),
        identical_frozen_controller=all(a['checkpoint_sha256']==json.loads((G6/f'execution_{s}/task_assessment.json').read_text())['checkpoint_sha256'] for s,a in assessments.items()))
    actuation={};realization={};contacts={}
    native_joints=m.actuator_trnid[:29,0];native_q=m.jnt_qposadr[native_joints];native_v=m.jnt_dofadr[native_joints]
    def position_excess(q):return np.maximum(np.maximum(q[:,native_q]-m.jnt_range[native_joints,1],m.jnt_range[native_joints,0]-q[:,native_q]),0.).max(0).tolist()
    for s in [42,43]:
        key=str(s);u=new[key]['ctrl_physics'][:,limited]
        actuation[key]=dict(maximum_control_range_excess_backend=float(np.maximum(np.maximum(u-limits[limited,1],limits[limited,0]-u),0.).max()),maximum_effort_excess_backend=assessments[key]['maximum_effort_excess_backend'])
        actuation[key].update(native_joint_names=[m.joint(j).name for j in native_joints],
            goal7_native_velocity_peak_radps=abs(new[key]['qvel'][:,native_v]).max(0).tolist(),goal6_native_velocity_peak_radps=abs(old[key]['qvel'][:,native_v]).max(0).tolist(),
            goal7_native_soft_position_stop_excursion_rad=position_excess(new[key]['qpos']),goal6_native_soft_position_stop_excursion_rad=position_excess(old[key]['qpos']),
            limit_contract='Unchanged hard actuator command/effort limits, compared in the actual float32 backend representation. Joint stops are compliant and already have small baseline penetrations; these are reported separately. The unchanged model has no configured actuator velocity cutoff.')
        projection=np.load(out/f'projection_{s}/rollout.npz');realization[key]={k:float(abs(new[key][k]-projection[k]).max()) for k in ['qpos','qvel','ctrl_physics']}
        a=np.load(out/f'execution_{s}/contact_attribution.npz');b=np.load(out/f'replay_{s}/contact_attribution.npz');contacts[key]=all(np.array_equal(a[k],b[k]) for k in a.files)
    gates.update(no_actuator_control_or_effort_violations=all(a['maximum_control_range_excess_backend']==0 and a['maximum_effort_excess_backend']<=0 for a in actuation.values()),
        realization_retains_physical_motion=all(r['qpos']==0 and r['qvel']==0 and r['ctrl_physics']<1e-12 for r in realization.values()),contact_replay_exact=all(contacts.values()))
    result.update(candidate_status='SUCCESS' if all(gates.values()) else 'FAILURE',physical_assessments=assessments,replay_errors={s:{k:r[k] for k in EXACT} for s,r in replays.items()},actuation=actuation,physical_projection_vs_execution=realization,
        videos={str(s):str(out/f'execution_{s}/rollout.mp4') for s in [42,43]},
        reproduction_command='OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -B scripts/compact_diffusion_pose_v1/reproduce.py --out logs/compact_diffusion_pose_v1_reproduction')
    from plot import plot
    plot(out)
    files=[p for p in sorted((ROOT/'scripts/compact_diffusion_pose_v1').iterdir()) if p.is_file()];result['source_sha256']={str(p.relative_to(ROOT)):sha(p) for p in files}
    (out/'goal7_assessment.json').write_text(json.dumps(result,indent=2))
    (out/'source_attribution.json').write_text(json.dumps(dict(tracking=tracks,authority_proxy=result['authority_proxy'],proxy_contract=result['authority_proxy_contract'],eight_of_ten_achieved=False),indent=2))
    artifacts=[p for p in out.rglob('*') if p.is_file() and p.suffix in ['.json','.npz','.mjb','.mp4','.png','.pdf'] and p.name!='goal7_manifest.json']
    (out/'goal7_manifest.json').write_text(json.dumps(dict(source_sha256=result['source_sha256'],artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in artifacts}),indent=2))
    print(json.dumps(dict(status=result['status'],candidate_status=result['candidate_status'],candidate_gates=gates,retention=retention),indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--screen',action='store_true');a=p.parse_args();report(a.out.resolve(),a.screen)
