"""Executed-motion evidence and fail-closed Goal 4 assessment."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
from refine import PHASES
sys.path.insert(0,str(ROOT/'scripts/compact_diffusion_v1'))
from evidence import WINDOWS


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(a,b,arms,actuators,windows):
    result={}
    for phase,(s,e) in windows.items():
        qa,qb=a['qpos'][s:e],b['qpos'][s:e]
        def position(start,end):
            values=np.linalg.norm(qa[:,start:end]-qb[:,start:end],axis=1)
            return dict(rms=float(np.sqrt(np.mean(values**2))),peak=float(values.max()))
        def angle(start,end):
            values=np.linalg.norm((Rotation.from_quat(qa[:,start:end],scalar_first=True)*Rotation.from_quat(qb[:,start:end],scalar_first=True).inv()).as_rotvec(),axis=1)
            return dict(rms=float(np.sqrt(np.mean(values**2))),peak=float(values.max()))
        arm=qa[:,arms]-qb[:,arms]
        ctrl=a['ctrl_physics'][s:e]-b['ctrl_physics'][s:e]
        result[phase]=dict(root_position_m=position(0,3),root_orientation_rad=angle(3,7),
            object_position_m=position(44,47),object_orientation_rad=angle(47,51),
            arm_joint_rad=dict(rms=float(np.sqrt(np.mean(arm**2))),peak=float(abs(arm).max()),per_joint_rms=np.sqrt(np.mean(arm**2,axis=0)).tolist()),
            actuator_commands=dict(rms=float(np.sqrt(np.mean(ctrl**2))),peak=float(abs(ctrl).max()),arm_rms=float(np.sqrt(np.mean(ctrl[:,actuators]**2))),per_actuator_rms=np.sqrt(np.mean(ctrl**2,axis=0)).tolist()))
    return result


def report(out):
    m=mujoco.MjModel.from_binary_path(str(out/'execution_42/rollout_model.mjb'))
    j=[j for j in range(m.njnt) if any(t in m.joint(j).name for t in ['shoulder_','elbow_','wrist_'])]
    aq=m.jnt_qposadr[j];act=np.array([np.flatnonzero(m.actuator_trnid[:,0]==a)[0] for a in j])
    a,b,disabled=[np.load(out/p/'rollout.npz') for p in ['execution_42','execution_43','disabled']]
    initial=all(np.array_equal(x[k],disabled[k]) for x in [a,b] for k in ['initial_qpos','initial_qvel'])
    active={k:(round(s/.005),round(e/.005)+1) for k,(s,e,_,_) in PHASES.items()}
    pairs={'42_vs_disabled':(a,disabled),'43_vs_disabled':(b,disabled),'42_vs_43':(a,b)}
    comparisons={name:dict(task_phase=compare(x,y,aq,act,WINDOWS),active_phase=compare(x,y,aq,act,active)) for name,(x,y) in pairs.items()}
    refs=[np.load(out/f'feasible_{seed}.npz') for seed in [42,43]]
    disabled_ref=np.load(out/'disabled.npz')
    identical_goals=all(np.array_equal(r[k],disabled_ref[k]) for r in refs for k in ['goal_position','goal_quaternion'])
    assessments={str(seed):json.loads((out/f'execution_{seed}/task_assessment.json').read_text()) for seed in [42,43]}
    exact_keys=['exact_qpos_error','exact_qvel_error','exact_control_error','exact_contact_trace_error','exact_locomotion_trace_error']
    replays={str(seed):json.loads((out/f'replay_{seed}/execution.json').read_text()) for seed in [42,43]}
    causal=comparisons['42_vs_43']['active_phase']
    substantial={phase:bool(d['object_position_m']['peak']>=.01 and d['object_position_m']['rms']>=.005 and d['arm_joint_rad']['rms']>=.01) for phase,d in causal.items()}
    measurable={phase:bool(d['object_position_m']['rms']>=.0005 and d['arm_joint_rad']['rms']>=.002 and d['actuator_commands']['rms']>1e-6) for phase,d in causal.items()}
    old=json.loads((ROOT/'logs/compact_diffusion_v1/causality.json').read_text())['comparisons']['42_vs_43']
    goal2a=np.load(ROOT/'logs/compact_diffusion_v1/execution_42/rollout.npz')
    goal2b=np.load(ROOT/'logs/compact_diffusion_v1/execution_43/rollout.npz')
    old_arm={phase:float(np.sqrt(np.mean((goal2a['qpos'][s:e,aq]-goal2b['qpos'][s:e,aq])**2))) for phase,(s,e) in WINDOWS.items()}
    gain={phase:dict(goal2_object_rms_m=old[phase]['object_rms_m'],goal4_object_rms_m=comparisons['42_vs_43']['task_phase'][phase]['object_position_m']['rms'],
            object_rms_ratio=comparisons['42_vs_43']['task_phase'][phase]['object_position_m']['rms']/old[phase]['object_rms_m'],
            goal2_arm_rms_rad=old_arm[phase],goal4_arm_rms_rad=comparisons['42_vs_43']['task_phase'][phase]['arm_joint_rad']['rms'],
            arm_rms_ratio=comparisons['42_vs_43']['task_phase'][phase]['arm_joint_rad']['rms']/old_arm[phase]) for phase in WINDOWS}
    preservation=json.loads((out/'preservation_before.json').read_text())
    preserved={path:sha(ROOT/path)==h for path,h in preservation.items()}
    inference={}
    for seed in [42,43,0]:
        fresh=np.load(out/f'inference_recheck/sample_{seed}.npz')
        oldsample=np.load(ROOT/'logs/compact_diffusion_v1'/('calibration' if seed==0 else 'proposals')/f'sample_{seed}.npz')
        inference[str(seed)]={key:bool(np.array_equal(fresh[key],oldsample[key])) for key in fresh.files}
    full_contact_replays={}
    projection_equality={}
    for seed in [42,43]:
        first=np.load(out/f'execution_{seed}/contact_attribution.npz');second=np.load(out/f'replay_{seed}/contact_attribution.npz')
        full_contact_replays[str(seed)]=all(np.array_equal(first[k],second[k]) for k in first.files)
        projection=np.load(out/f'projection_{seed}/rollout.npz');final=np.load(out/f'execution_{seed}/rollout.npz')
        projection_equality[str(seed)]={k:float(abs(final[k]-projection[k]).max()) for k in ['qpos','qvel','ctrl_physics']}
    base=np.load(ROOT/'logs/compact_orcs_v1/complete_execution/rollout.npz')
    disabled_exact={k:float(abs(disabled[k]-base[k]).max()) for k in ['qpos','qvel','ctrl_physics']}
    limits={}
    for seed,run in [(42,a),(43,b)]:
        execution=json.loads((out/f'execution_{seed}/execution.json').read_text())
        limit=np.max(abs(m.actuator_forcerange),axis=1).astype(np.float32).astype(np.float64)
        forces=np.asarray(execution['maximum_abs_actuator_force'])
        limits[str(seed)]=dict(maximum_effort_excess_backend=float((forces-limit).max()),
            peak_arm_joint_velocity_radps=abs(run['qvel'][:,m.jnt_dofadr[j]]).max(0).tolist(),
            baseline_peak_arm_joint_velocity_radps=abs(disabled['qvel'][:,m.jnt_dofadr[j]]).max(0).tolist(),
            maximum_arm_soft_joint_limit_excursion_rad=float(np.maximum(run['qpos'][:,aq]-m.jnt_range[j,1],m.jnt_range[j,0]-run['qpos'][:,aq]).max()),
            baseline_maximum_arm_soft_joint_limit_excursion_rad=float(np.maximum(disabled['qpos'][:,aq]-m.jnt_range[j,1],m.jnt_range[j,0]-disabled['qpos'][:,aq]).max()),
            velocity_limit_contract='The accepted compiled model has no actuator velocity cutoff. Actual per-joint peaks are reported; the refiner bounds added arm velocity to 0.7 rad/s. No velocity limit was introduced or relaxed.')
    gates=dict(two_physical_successes=all(x['status']=='SUCCESS' and len(x['numeric_gates'])==22 and all(x['numeric_gates'].values()) for x in assessments.values()),
        two_independent_exact_replays=all(all(replays[s][k]==0 for k in exact_keys) for s in replays),
        exact_full_contact_attribution=all(full_contact_replays.values()),identical_initial_states=initial,
        identical_task_goals=identical_goals,
        disabled_exact_successful_baseline=all(x==0 for x in disabled_exact.values()),
        substantial_actual_sample_difference_in_two_phases=sum(substantial.values())>=2,
        measurable_actual_sample_difference_in_every_phase=all(measurable.values()),
        genuine_frozen_inference_reproduced=all(all(x.values()) for x in inference.values()),
        preserved_prior_sources_and_artifacts=all(preserved.values()),
        tracked_files_unchanged=not subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).strip())
    result=dict(status='SUCCESS' if all(gates.values()) else 'FAILURE',gates=gates,
        authority_criterion='At least two phases: >=1 cm peak and >=5 mm active RMS executed sample-dependent object translation plus >=0.01 rad arm RMS; every phase >=0.5 mm object RMS and >=0.002 rad arm RMS. This operational criterion is not a calibrated 1–10 authority scale.',
        substantial_phases=substantial,measurable_phases=measurable,goal2_comparison=gain,
        comparisons=comparisons,physical_assessments=assessments,
        actual_actuation_and_joint_limits=limits,
        replay_errors={s:{k:v[k] for k in exact_keys} for s,v in replays.items()},
        exact_contact_attribution=full_contact_replays,disabled_baseline_errors=disabled_exact,
        physical_projection_vs_final_execution_max_errors=projection_equality,
        fresh_frozen_inference_equality=inference,prior_preservation=preserved,
        frozen_diffusion=json.loads((out/'inference_recheck/provenance.json').read_text()),
        reference_refinement={str(seed):json.loads((out/f'dense_{seed}.json').read_text()) for seed in [42,43]},
        arms=[m.joint(x).name for x in j],actuators=[m.actuator(x).name for x in range(m.nu)],
        windows=dict(task_phase=WINDOWS,active_phase=active),
        videos={str(seed):str(out/f'execution_{seed}/rollout.mp4') for seed in [42,43]},
        limitations=['Fixed expert footstep positions, support schedule, root/waist references, gripper actuation and contact topology.',
            'Only arm configuration and carried-object pose are retargeted; learned feedback and physical parameters are frozen.',
            'The 50 g original-sized box and fixed accepted scenario are the only validated physical task.',
            'Frozen native-conditioned samples are retargeted; direct compact-task conditioning is not claimed.'],
        files_changed=[str(x.relative_to(ROOT)) for x in sorted((ROOT/'scripts/compact_diffusion_authority_v1').iterdir()) if x.is_file()],
        new_source_sha256={str(x.relative_to(ROOT)):sha(x) for x in sorted((ROOT/'scripts/compact_diffusion_authority_v1').iterdir()) if x.is_file()},
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        reproduction_command='OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -B scripts/compact_diffusion_authority_v1/reproduce.py --out logs/compact_diffusion_authority_v1_reproduction')
    (out/'goal4_assessment.json').write_text(json.dumps(result,indent=2))
    artifact_paths=[p for p in out.rglob('*') if p.is_file() and p.suffix in ['.json','.npz','.mjb','.mp4','.png','.pdf'] and p.name!='goal4_manifest.json']
    (out/'goal4_manifest.json').write_text(json.dumps(dict(source_sha256=result['new_source_sha256'],artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in artifact_paths}),indent=2))
    print(json.dumps(dict(status=result['status'],gates=gates,gain=gain,active_sample_difference=causal),indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();report(a.out.resolve())
