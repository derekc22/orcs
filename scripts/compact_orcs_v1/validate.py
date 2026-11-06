"""Accepted task criteria applied to actual ORCS substep traces and replay."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/compact_integration_v1'))
from metrics import measure,runs


def validate(path,replay):
    a=np.load(path/'rollout.npz');e=json.loads((path/'execution.json').read_text())
    r=json.loads((replay/'execution.json').read_text());m=mujoco.MjModel.from_binary_path(str(path/'rollout_model.mjb'))
    t=np.load(path/'rollout_verification_trace.npz')['trace'];l=np.load(path/'rollout_locomotion_trace.npz')['trace']
    s=int(a['walking_start_physics_step']);stop=int(a['walking_end_physics_step']);op=round(float(a['phase_opening_frame'])*4)
    verification=dict(exact_qpos_error=r['exact_qpos_error'],exact_qvel_error=r['exact_qvel_error'],
        exact_control_error=r['exact_control_error'],replay_backend='Independent registered ORCS UOLM / MJWarp action replay',
        maximum_lift_m=float(t[:,5].max()),contact_impulses_Ns=e['contact_impulses_Ns'],
        forbidden_object_impulse_Ns=float(e['bad_impulses_Ns'][0]),invalid_robot_terrain_impulse_Ns=float(e['bad_impulses_Ns'][2]),
        maximum_applied_external_force=e['maximum_applied_external_force'],equality_constraints=int(m.neq),
        final_object_linear_speed_mps=float(np.linalg.norm(t[-1,20:23])),final_object_angular_speed_radps=float(np.linalg.norm(t[-1,23:26])),
        final_robot_root_speed=float(np.linalg.norm(t[-1,36:42])),maximum_abs_actuator_force=e['maximum_abs_actuator_force'])
    (path/'rollout_verification.json').write_text(json.dumps(verification,indent=2))
    metrics=measure(path)
    weight=m.body_mass[m.body('object/object').id]*9.81
    last=np.flatnonzero(t[:,43]>0);last_ee=int(last[-1]) if len(last) else -1
    support=[b for b,c in runs((t[:,0]>0)&(t[:,42]>.7*weight)) if c-b>=30 and b>stop]
    aq=m.jnt_qposadr[m.actuator_trnid[:,0]];hand=a['qpos'][-1,aq[[29,30,33,34,35,36]]]
    # MJWarp uploads these unchanged limits as float32. Compare against the
    # actual backend representation (e.g. 0.3 -> 0.30000001192092896), not a
    # different double-precision literal or an enlarged physical tolerance.
    source_effort_limits=np.max(abs(m.actuator_forcerange),axis=1)
    backend_effort_limits=source_effort_limits.astype(np.float32).astype(np.float64)
    full=len(t)==9629;walk=len(t)>=stop;has_open=len(t)>op
    model_audit=json.loads((path/'model_equivalence.json').read_text())
    video=json.loads((path/'video_metadata.json').read_text())
    preserved=json.loads((path/'baseline_preservation.json').read_text())
    retained=metrics.get('bilateral_retained',{})
    gates=dict(full_execution=full,initial_ground_support=bool(t[0,0] and t[0,42]>.7*weight),
        terrain_free_lift_over_5cm=bool(walk and np.min(t[s:stop,5])>.05),
        actual_retained_root_walk_over_2m=bool(retained.get('root_net_displacement_m',0)>=2),
        alternating_footsteps_at_least_5=bool(retained.get('longest_consecutive_alternating_replants',0)>=5),
        airborne_and_bilateral_grip_through_walk=bool(walk and not t[s:stop,0].any() and (l[s:stop,16:18]>1e-4).all()),
        continuous_physical_grasp_while_airborne=bool(last_ee>=0 and (l[(t[:,0]==0)&(np.arange(len(t))<=last_ee),16:18].sum(1)>1e-4).all()),
        support_gated_opening=bool(has_open and support and min(support)<op and t[op,42]>.7*weight),
        physical_open_hands=bool(full and (hand[:2]<-.02).all() and (hand[2:4]<.12).all() and (abs(hand[4:])<.02).all()),
        clean_withdrawal=bool(full and last_ee>=0 and last_ee<len(t)-200 and t[last_ee+1:,0].all() and not t[last_ee+1:,43].any()),
        stable_final_robot=bool(full and np.linalg.norm(t[-200:,36:39],axis=1).max()<.01 and np.linalg.norm(t[-200:,39:42],axis=1).max()<.03 and (l[-200:,14:16].min(0)>10).all()),
        stable_final_object=bool(full and metrics['final_1s_object_linear_speed_max_mps']<.001 and metrics['final_1s_object_angular_speed_max_radps']<.003),
        zero_forbidden_object_impulse=e['bad_impulses_Ns'][0]==0,
        zero_self_and_invalid_terrain=bool(e['bad_impulses_Ns'][1]==0 and e['bad_impulses_Ns'][2]==0),
        independent_exact_replay=bool(r['exact_qpos_error']==0 and r['exact_qvel_error']==0 and r['exact_control_error']==0 and r['exact_contact_trace_error']==0 and r['exact_locomotion_trace_error']==0 and r['steps']==9629),
        physical_model_integrity=bool(m.neq==0 and m.nmocap==0 and m.ntendon==0 and m.opt.disableflags==0 and e['maximum_applied_external_force']==0),
        effort_limits_preserved=bool(np.all(np.asarray(e['maximum_abs_actuator_force'])<=backend_effort_limits)),
        genuine_orcs_pipeline=bool(e['env_class']=='mjlab.envs.manager_based_rl_env.ManagerBasedRlEnv' and e['sim_class']=='mjlab.sim.sim.Simulation' and e['actor_class']=='CompactFeedbackActor' and len(e['managers'])==5),
        reference_clock=bool(np.array_equal(a['reference_frame'],np.minimum(np.arange(1,len(t)+1),9628))),
        accepted_model_equivalence=bool(model_audit['pass_model_equivalence']),
        full_task_video=bool((path/'rollout.mp4').is_file() and video['frames']==len(t)//4 and video['fps']==50.),
        accepted_baseline_preserved=all(v['unchanged'] for v in preserved.values()),
    )
    baseline=np.load(ROOT/'logs/compact_policy_learning_v1/complete_execution/rollout.npz')
    compare=dict(root_position_rms_m=float(np.sqrt(np.mean((a['qpos'][:,:3]-baseline['qpos'][:len(t),:3])**2))),
                 object_position_rms_m=float(np.sqrt(np.mean((a['qpos'][:,44:47]-baseline['qpos'][:len(t),44:47])**2))),
                 maximum_abs_control_difference=float(abs(a['ctrl_physics']-baseline['ctrl_physics'][:len(t)]).max()))
    result=dict(status='SUCCESS' if all(gates.values()) else 'FAILURE',numeric_gates=gates,failed_gates=[k for k,v in gates.items() if not v],
        physics_steps=len(t),duration_seconds=len(t)*.005,walking_root_displacement_m=retained.get('root_net_displacement_m',0),
        alternating_replants=retained.get('longest_consecutive_alternating_replants',0),maximum_lift_m=float(t[:,5].max()),
        minimum_walk_hand_force_N=l[s:stop,16:18].min(0).tolist() if walk else None,
        terrain_support_start_s=min(support)*.005 if support else None,opening_s=op*.005,
        terrain_load_at_opening_N=float(t[op,42]) if has_open else None,last_hand_contact_s=last_ee*.005,
        final_root_linear_speed_max_mps=float(np.linalg.norm(t[-200:,36:39],axis=1).max()),
        final_root_angular_speed_max_radps=float(np.linalg.norm(t[-200:,39:42],axis=1).max()),
        final_object_linear_speed_max_mps=metrics['final_1s_object_linear_speed_max_mps'],
        final_object_angular_speed_max_radps=metrics['final_1s_object_angular_speed_max_radps'],
        bad_impulses_Ns=e['bad_impulses_Ns'],standalone_comparison=compare,
        maximum_effort_excess_backend=float(np.max(np.asarray(e['maximum_abs_actuator_force'])-backend_effort_limits)),
        maximum_effort_limit_representation_roundoff=float(np.max(abs(backend_effort_limits-source_effort_limits))),
        replay={k:r[k] for k in ['exact_qpos_error','exact_qvel_error','exact_control_error','exact_contact_trace_error','exact_locomotion_trace_error']},
        checkpoint_sha256=hashlib.sha256((ROOT/'logs/compact_policy_learning_v1/selected_policy.npz').read_bytes()).hexdigest())
    (path/'task_metrics.json').write_text(json.dumps(metrics,indent=2))
    (path/'task_assessment.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('artifact',type=Path);p.add_argument('--replay',type=Path,required=True);a=p.parse_args();validate(a.artifact,a.replay)
