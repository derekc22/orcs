"""Independent raw-control replay and conservative complete-task assessment.

The unchanged accepted verifier attributes contacts and measures every physical
step. This wrapper also handles partial failures without assuming that grasp or
placement events exist. Visual acceptance is reported separately, never forged.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import mujoco
import numpy as np
from controller import ROOT, OUT
sys.path.insert(0,str(ROOT/'scripts/compact_integration_v1'))
from metrics import measure, runs


def assess(path,video=False,replay=True):
    path=Path(path);a=np.load(path/'rollout.npz')
    if not len(a['qpos']):
        result=dict(complete_numeric_success=False,failure='invalid_initial_state',numeric_gates=dict(initial_state_valid=False),failed_gates=['initial_state_valid'],visual_review_complete=False)
        (path/'evaluation.json').write_text(json.dumps(result,indent=2));return result
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MUJOCO_GL='egl')
    if replay:
        with (path/'verify.log').open('w') as log:
            subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/compact_policy_learning_v1/replay.py'),str(path/'rollout.npz')]+(['--video'] if video else []),env=env,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
    m=mujoco.MjModel.from_binary_path(str(path/'rollout_model.mjb'))
    v=json.loads((path/'rollout_verification.json').read_text());r=measure(path)
    r.setdefault('bilateral_retained',dict(root_net_displacement_m=0.,longest_consecutive_alternating_replants=0,qualifying_replants=[]))
    t=np.load(path/'rollout_verification_trace.npz')['trace'];l=np.load(path/'rollout_locomotion_trace.npz')['trace']
    s=int(a['walking_start_physics_step']);e=int(a['walking_end_physics_step']);op=round(float(a['phase_opening_frame'])*4)
    weight=m.body_mass[m.body('object/object').id]*9.81
    full=len(t)==9629;walk=len(t)>=e;has_open=len(t)>op
    last=np.flatnonzero(t[:,43]>0);last_ee=int(last[-1]) if len(last) else -1
    final=slice(-min(200,len(t)),None)
    aq=m.jnt_qposadr[m.actuator_trnid[:,0]];hand=a['qpos'][-1,aq[[29,30,33,34,35,36]]]
    support=[b for b,c in runs((t[:,0]>0)&(t[:,42]>.7*weight)) if c-b>=30 and b>e]
    gates=dict(full_execution=full,initial_ground_support=bool(t[0,0] and t[0,42]>.7*weight),
        terrain_free_lift_over_5cm=bool(walk and np.min(t[s:e,5])>.05),
        actual_retained_root_walk_over_2m=bool(r['bilateral_retained']['root_net_displacement_m']>=2),
        alternating_footsteps_at_least_5=bool(r['bilateral_retained']['longest_consecutive_alternating_replants']>=5),
        airborne_and_bilateral_grip_through_walk=bool(walk and not t[s:e,0].any() and (l[s:e,16:18]>1e-4).all()),
        continuous_physical_grasp_while_airborne=bool(last_ee>=0 and (l[(t[:,0]==0)&(np.arange(len(t))<=last_ee),16:18].sum(1)>1e-4).all()),
        support_gated_opening=bool(has_open and support and min(support)<op and t[op,42]>.7*weight),
        physical_open_hands=bool(full and (hand[:2]<-.02).all() and (hand[2:4]<.12).all() and (abs(hand[4:])<.02).all()),
        clean_withdrawal=bool(full and last_ee>=0 and last_ee<len(t)-200 and t[last_ee+1:,0].all() and not t[last_ee+1:,43].any()),
        stable_final_robot=bool(full and np.linalg.norm(t[final,36:39],axis=1).max()<.01 and np.linalg.norm(t[final,39:42],axis=1).max()<.03 and (l[final,14:16].min(0)>10).all()),
        stable_final_object=bool(full and r['final_1s_object_linear_speed_max_mps']<.001 and r['final_1s_object_angular_speed_max_radps']<.003),
        zero_forbidden_object_impulse=v['forbidden_object_impulse_Ns']==0,
        zero_self_and_invalid_terrain=bool(r['robot_self_impulse_Ns']==0 and r['invalid_terrain_impulse_Ns']==0),
        independent_exact_replay=bool(v['exact_qpos_error']==0 and v['exact_qvel_error']==0),
        physical_model_integrity=bool(m.neq==0 and m.nmocap==0 and m.ntendon==0 and m.opt.disableflags==0 and v['maximum_applied_external_force']==0),
        effort_limits_preserved=bool(np.all(np.asarray(v['maximum_abs_actuator_force'])<=np.max(abs(m.actuator_forcerange),axis=1)+1e-9)))
    r.update(numeric_gates=gates,complete_numeric_success=all(gates.values()),failed_gates=[k for k,x in gates.items() if not x],
        minimum_walk_hand_force_N=l[s:min(e,len(l)),16:18].min(0).tolist() if len(l)>s else None,
        last_hand_contact_s=last_ee*.005,terrain_support_start_s=min(support)*.005 if support else None,
        ground_load_at_opening_N=float(t[op,42]) if has_open else None,
        final_root_linear_speed_max_mps=float(np.linalg.norm(t[final,36:39],axis=1).max()),
        final_root_angular_speed_max_radps=float(np.linalg.norm(t[final,39:42],axis=1).max()),
        visual_review_complete=json.loads((path/'visual_review.json').read_text()).get('full_sequence_visual_review_complete',False) if (path/'visual_review.json').exists() else False,
        full_video_rendered=(path/'rollout.mp4').exists())
    (path/'evaluation.json').write_text(json.dumps(r,indent=2))
    print(path.name, 'numeric_pass',r['complete_numeric_success'],'walk',r['bilateral_retained']['root_net_displacement_m'],
          'steps',r['bilateral_retained']['longest_consecutive_alternating_replants'],'failed',r['failed_gates'],flush=True)
    return r


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directories',type=Path,nargs='+');p.add_argument('--video',action='store_true');p.add_argument('--workers',type=int,default=1);p.add_argument('--reuse-replay',action='store_true');args=p.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda path:assess(path,args.video,not args.reuse_replay),args.directories))
