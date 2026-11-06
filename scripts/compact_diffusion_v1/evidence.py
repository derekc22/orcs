"""Physical screening and deterministic, phase-specific diffusion causality."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/compact_integration_v1'))
from metrics import runs,steps_in

WINDOWS={'pickup':(0,3449),'carry':(3449,4721),'placement':(6701,8092)}
NATIVE=np.r_[7:29,33:40]


def physical(path):
    a=np.load(path/'rollout.npz');e=json.loads((path/'execution.json').read_text())
    t=np.load(path/'rollout_verification_trace.npz')['trace'];l=np.load(path/'rollout_locomotion_trace.npz')['trace']
    m=mujoco.MjModel.from_binary_path(str(path/'rollout_model.mjb'))
    s=int(a['walking_start_physics_step']);end=int(a['walking_end_physics_step'])
    if len(t)!=9629:return {'pass':False,'steps':len(t)}
    weight=m.body_mass[m.body('object/object').id]*9.81
    order=[v['side'] for v in steps_in(l,s,end)]
    longest=0;current=0;previous=None
    for side in order:
        current=current+1 if previous is not None and side!=previous else 1
        longest=max(longest,current);previous=side
    displacement=float(np.linalg.norm(a['qpos'][end-1,:2]-a['qpos'][s,:2]))
    last=int(np.flatnonzero(t[:,43]>0)[-1]);op=round(float(a['phase_opening_frame'])*4)
    support=[b for b,c in runs((t[:,0]>0)&(t[:,42]>.7*weight)) if c-b>=30 and b>end]
    gates=dict(initial_support=bool(t[0,0] and t[0,42]>.7*weight),
        airborne_grip=bool(not t[s:end,0].any() and (l[s:end,16:18]>1e-4).all() and t[s:end,5].min()>.05),
        actual_walk=displacement>=2,alternating_steps=longest>=5,
        airborne_contact_continuity=bool((l[(t[:,0]==0)&(np.arange(len(t))<=last),16:18].sum(1)>1e-4).all()),
        support_before_open=bool(support and min(support)<op and t[op,42]>.7*weight),
        withdrawal=bool(last<len(t)-200 and t[last+1:,0].all() and not t[last+1:,43].any()),
        stable_robot=bool(np.linalg.norm(t[-200:,36:39],axis=1).max()<.01 and np.linalg.norm(t[-200:,39:42],axis=1).max()<.03 and (l[-200:,14:16].min(0)>10).all()),
        stable_object=bool(np.linalg.norm(t[-200:,20:23],axis=1).max()<.001 and np.linalg.norm(t[-200:,23:26],axis=1).max()<.003),
        zero_bad_impulses=all(v==0 for v in e['bad_impulses_Ns']))
    result=dict(pass_physical_screen=all(gates.values()),gates=gates,walk_m=displacement,alternating_replants=longest,
        maximum_lift_m=float(t[:,5].max()),bad_impulses_Ns=e['bad_impulses_Ns'],
        note='Preliminary screen only; unchanged Goal 1 validator must independently verify final rollout, replay and video.')
    (path/'physical_screen.json').write_text(json.dumps(result,indent=2));return result


def difference(a,b):
    result={}
    for phase,(s,e) in WINDOWS.items():
        q=a['qpos'][s:e]-b['qpos'][s:e]
        root=float(np.sqrt(np.mean(np.sum(q[:,:3]**2,axis=1))))
        obj=float(np.sqrt(np.mean(np.sum(q[:,44:47]**2,axis=1))))
        native=float(np.sqrt(np.mean(q[:,NATIVE]**2)))
        ctrl=float(np.sqrt(np.mean((a['ctrl_physics'][s:e]-b['ctrl_physics'][s:e])**2)))
        native_ctrl=float(np.sqrt(np.mean((a['ctrl_physics'][s:e,:29]-b['ctrl_physics'][s:e,:29])**2)))
        result[phase]=dict(root_rms_m=root,object_rms_m=obj,native_joint_rms_rad=native,
            actuator_command_rms=ctrl,native_actuator_command_rms_rad=native_ctrl,pass_meaningful_motion=bool(root>=.0005 or obj>=.0005 or native>=.002),
            pass_command_difference=ctrl>1e-6)
    return result


def causal(paths,baseline,out,references=None):
    a,b=[np.load(p/'rollout.npz') for p in paths];base=np.load(baseline/'rollout.npz')
    initial=all(np.array_equal(z[k],base[k]) for z in [a,b] for k in ['initial_qpos','initial_qvel'])
    result=dict(identical_initial_state=initial,thresholds=dict(root_or_object_rms_m=.0005,native_joint_rms_rad=.002),
        comparisons={'42_vs_baseline':difference(a,base),'43_vs_baseline':difference(b,base),'42_vs_43':difference(a,b)})
    if references:
        refs=[np.load(p) for p in references]
        result['identical_task_goals']=all(np.array_equal(refs[0][k],refs[1][k]) for k in ['goal_position','goal_quaternion'])
        result['reference_comparison']=difference(dict(qpos=refs[0]['qref'],ctrl_physics=refs[0]['uref']),dict(qpos=refs[1]['qref'],ctrl_physics=refs[1]['uref']))
        expert=np.load(ROOT/'logs/compact_policy_learning_v1/selected_policy.npz')
        result['reference_vs_expert']={str(seed):difference(dict(qpos=r['qref'],ctrl_physics=r['uref']),dict(qpos=expert['qref'],ctrl_physics=expert['uref'])) for seed,r in zip([42,43],refs)}
    result['pass_causality']=initial and all(p['pass_meaningful_motion'] and p['pass_command_difference'] for c in result['comparisons'].values() for p in c.values())
    if references:
        result['pass_causality'] &= result['identical_task_goals'] and all(p['pass_meaningful_motion'] for p in result['reference_comparison'].values())
    out.write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='mode',required=True)
    x=s.add_parser('physical');x.add_argument('path',type=Path)
    x=s.add_parser('causal');x.add_argument('--runs',type=Path,nargs=2,required=True);x.add_argument('--baseline',type=Path,required=True);x.add_argument('--out',type=Path,required=True);x.add_argument('--references',type=Path,nargs=2)
    a=p.parse_args();r=physical(a.path) if a.mode=='physical' else causal(a.runs,a.baseline,a.out,a.references)
    print(json.dumps(r,indent=2))
