"""Final recovery after terrain placement and complete physical withdrawal."""
import argparse,json,sys
from pathlib import Path
import mujoco,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/corner_hands_v1'),str(ROOT/'scripts/articulated_gripper_v1'),str(ROOT/'scripts')]
sys.path.append(str(ROOT/'scripts/compact_integration_v2'))
from mechanism import indices
from physical import contacts,Arms,thumb_control
from compact_gripper_balance import Balance

def run(args):
    source=ROOT/args.source;out=ROOT/'logs/compact_place_refinement_v1'/args.tag;out.mkdir(parents=True,exist_ok=False)
    a=np.load(source);m=mujoco.MjModel.from_binary_path(str(source.with_name(source.stem+'_model.mjb')));d=mujoco.MjData(m);d.qpos[:]=a['initial_qpos'];d.qvel[:]=a['initial_qvel'];mujoco.mj_forward(m,d)
    prefix=args.prefix if args.prefix is not None else len(a['ctrl_physics'])
    for u in a['ctrl_physics'][:prefix]:d.ctrl[:]=u;mujoco.mj_step(m,d)
    aj,aq,av,arm,pads=indices(m);loaded=d.ctrl.copy();rest=d.qpos.copy();bias0=d.qfrc_bias.copy();balance=Balance(m,d);arms=Arms(m,rest);rows=[];sums=np.zeros(6);reason='completed';cs=list(a['ctrl_physics'][:prefix]);qs=list(a['qpos'][:prefix]);vs=list(a['qvel'][:prefix]);cts=list(a['contacts'][:prefix]);fc=d.qfrc_constraint.copy()
    if args.controller=='bounded':
        from bounded_balance import BoundedBalance
        balance=BoundedBalance(m,d)
    for step in range(round(args.seconds/.005)):
        if args.controller in ['balance','bounded']:
            tau,_=balance.torques(d,rest)
            u=d.qpos[aq]+(tau[av]-m.actuator_biasprm[:,2]*d.qvel[av])/m.actuator_gainprm[:,0]
        elif args.controller=='loaded':u=loaded.copy()
        else:
            fc=.8*fc+.2*d.qfrc_constraint
            u=rest[aq]+(d.qfrc_bias[av]-fc[av])/m.actuator_gainprm[:,0]
        u[arm]=loaded[arm]+(d.qfrc_bias[av[arm]]-bias0[av[arm]])/m.actuator_gainprm[arm,0]
        u[29:31]=-.025;u[33:35]=0.;u[35:37]=0.
        u=np.clip(u,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);d.ctrl[:]=u;mujoco.mj_step(m,d)
        touch,N,imp,v,si,invalid=contacts(m,d);sums+=np.r_[imp,si,invalid]
        cs.append(u.copy());qs.append(d.qpos.copy());vs.append(d.qvel.copy());cts.append(touch);rows.append(np.r_[step*.005,d.qpos[:3],d.qvel[:6],N,sums])
        if step%100==99:print(args.tag,round(step*.005,3),'root',np.round(d.qpos[:3],3),'rootv',np.round(d.qvel[:6],3),'N',np.round(N,2),'bad',sums[3:],flush=True)
        if sums[3]>1e-5:reason='forbidden_object'
        elif sums[4]>.001:reason='self_contact'
        elif sums[5]>.001:reason='invalid_terrain'
        elif not touch[0]:reason='object_not_grounded'
        elif touch[1] or touch[2]:reason='hand_recontact'
        elif d.qpos[2]<.32:reason='fall'
        if reason!='completed':print('EARLY_REJECT',reason,flush=True);break
    data=dict(initial_qpos=a['initial_qpos'],initial_qvel=a['initial_qvel'],ctrl_physics=cs,qpos=qs,qvel=vs,contacts=cts,settling_diagnostics=rows,phase_opening_frame=a['phase_opening_frame'],phase_withdrawal_end_frame=a['phase_withdrawal_end_frame'],recovery_start_physics_step=prefix)
    for key in ['terrain_transfer_start_physics_step','underfinger_opening_physics_step','placement_start_physics_step','transition_start_physics_step']:
        if key in a:data[key]=a[key]
    for key in ['walking_start_physics_step','walking_end_physics_step']:data[key]=a[key]
    np.savez_compressed(out/'rollout.npz',**data);mujoco.mj_saveModel(m,str(out/'rollout_model.mjb'),None)
    result=dict(arguments=vars(args),reason=reason,prefix=prefix,continuation_seconds=len(rows)*.005,impulses=sums.tolist());(out/'generation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--tag',required=True);p.add_argument('--prefix',type=int);p.add_argument('--seconds',type=float,default=4.);p.add_argument('--controller',choices=['balance','loaded','contact','bounded'],default='bounded');run(p.parse_args())
