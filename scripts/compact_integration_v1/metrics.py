"""Conservative partial-result measurements from independent replay traces."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'logs/compact_integration_v1'


def runs(mask):
    edges=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))


def steps_in(l,s,e):
    support=l[:,14:16]>=10.
    feet=l[:,6:12].reshape(-1,2,3)
    events=[]
    for side in range(2):
        for off,on in runs(~support[:,side]):
            if off<s or on>=e or (on-off)*.005<.06:continue
            clear=float(l[off:on,12+side].max())
            distance=float(np.linalg.norm(feet[on,side,:2]-feet[off,side,:2]))
            if clear<.02 or distance<.12:continue
            if support[max(0,off-12):off,side].mean()<.5 or support[on:on+12,side].mean()<.5:continue
            events.append(dict(side=['left','right'][side],liftoff_s=off*.005,replant_s=on*.005,
                               clearance_m=clear,same_foot_relocation_m=distance))
    return sorted(events,key=lambda ev:ev['replant_s'])


def measure(p):
    a=np.load(p/'rollout.npz')
    v=json.loads((p/'rollout_verification.json').read_text())
    t=np.load(p/'rollout_verification_trace.npz')['trace']
    l=np.load(p/'rollout_locomotion_trace.npz')['trace']
    q=a['qpos'];N=len(q)
    s=int(a['walking_start_physics_step']) if 'walking_start_physics_step' in a else -1
    e=min(N,int(a['walking_end_physics_step'])) if 'walking_end_physics_step' in a else -1
    impulse=v['contact_impulses_Ns']
    forbidden={k:x for k,x in impulse.items() if k!='terrain' and not k.startswith(('hand_','ROBOT_SELF:','INVALID_TERRAIN:'))}
    result=dict(duration_s=N*.005,maximum_COM_lift_m=v['maximum_lift_m'],
                exact_qpos_error=v['exact_qpos_error'],exact_qvel_error=v['exact_qvel_error'],
                forbidden_object_impulses_Ns=forbidden,robot_self_impulse_Ns=sum(x for k,x in impulse.items() if k.startswith('ROBOT_SELF:')),
                invalid_terrain_impulse_Ns=v['invalid_robot_terrain_impulse_Ns'],
                airborne_intervals_s=[[x*.005,y*.005] for x,y in runs(t[:,0]==0)],
                final_object_linear_speed_mps=v['final_object_linear_speed_mps'],
                final_object_angular_speed_radps=v['final_object_angular_speed_radps'],
                final_robot_root_speed=v['final_robot_root_speed'],
                final_1s_object_linear_speed_max_mps=float(np.linalg.norm(t[-200:,20:23],axis=1).max()),
                final_1s_object_angular_speed_max_radps=float(np.linalg.norm(t[-200:,23:26],axis=1).max()),
                maximum_external_force=v['maximum_applied_external_force'],equalities=v['equality_constraints'],
                native_limits_note='All force limits are the preserved source model limits; no model arrays were modified.')
    if 0<=s<e:
        # Force/contact continuity, terrain clearance, and non-EE object contact
        # independently bound the interval; a thrown airborne box is excluded.
        zero_bad=np.cumsum(t[:,8])<1e-8
        for key,held in [('any_hand_retained',l[:,16:18].sum(1)>1e-4),
                         ('bilateral_retained',(l[:,16:18]>1e-4).all(1))]:
            mask=(t[:,0]==0)&held&zero_bad
            intervals=[(s+x,s+y) for x,y in runs(mask[s:e])]
            pairs=[]
            for begin,end in intervals:
                net=float(np.linalg.norm(q[end-1,:2]-q[begin,:2]))
                pairs.append((net,begin,end))
            net,begin,end=max(pairs,default=(0.,s,s))
            events=steps_in(l,begin,end) if end>begin else []
            order=[ev['side'] for ev in events]
            longest_alternating=0;current=0;previous=None
            for side in order:
                current=current+1 if previous is not None and side!=previous else 1
                longest_alternating=max(longest_alternating,current);previous=side
            result[key]=dict(interval_s=[begin*.005,end*.005],duration_s=(end-begin)*.005,
                             root_net_displacement_m=net,qualifying_replants=events,
                             replant_order=order,longest_consecutive_alternating_replants=longest_alternating,
                             terrain_contact_steps=int(t[begin:end,0].sum()))
        result['explicit_walk_phase_realized_s']=[s*.005,e*.005]
        result['root_net_entire_realized_walk_phase_m']=float(np.linalg.norm(q[e-1,:2]-q[s,:2]))
        result['walking_phase_note']='The entire phase can include retention failure; only the force-bounded intervals above count as retained carry.'
    else:
        result['walking_carry_distance_m']=0.
        result['qualifying_replants']=[]
    return result


if __name__=='__main__':
    results={}
    for p in sorted(OUT.iterdir()):
        if not (p/'rollout_verification.json').exists():continue
        results[p.name]=measure(p)
    (OUT/'partial_metrics.json').write_text(json.dumps(results,indent=2))
    for name,r in results.items():
        walk=r.get('any_hand_retained',{})
        print(name,'lift',round(r['maximum_COM_lift_m'],4),'walk',round(walk.get('root_net_displacement_m',0.),4),
              'replants',len(walk.get('qualifying_replants',[])),'bad',r['forbidden_object_impulses_Ns'],
              'self',round(r['robot_self_impulse_Ns'],6),'replay',r['exact_qpos_error'],r['exact_qvel_error'])
