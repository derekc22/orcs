"""Execute the registered compact UOLM task through its native policy runner."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import mjlab
import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from orcs.tasks.uolm.compact import TASK_ID
from orcs.tasks.uolm.compact.model import BASE, CHECKPOINT
from model_audit import audit as audit_model


def run(out, steps, replay=None, task=TASK_ID, reference_path=None):
    out.mkdir(parents=True, exist_ok=False)
    cfg = load_env_cfg(task, play=True)
    if reference_path is not None:
        cfg.commands['motion'].refined_reference_path=str(reference_path.resolve())
    start = time.time()
    env = ManagerBasedRlEnv(cfg=cfg, device='cpu')
    wrapper = RslRlVecEnvWrapper(env, clip_actions=None)
    runner = MjlabOnPolicyRunner(wrapper, asdict(load_rl_cfg(task)), device='cpu')
    policy = runner.get_inference_policy(device='cpu')
    obs = wrapper.get_observations()
    recorded = np.load(replay/'rollout.npz') if replay else None
    q0 = env.sim.data.qpos[0].numpy().copy(); v0 = env.sim.data.qvel[0].numpy().copy()
    mujoco.mj_saveModel(env.sim.mj_model, str(out/'rollout_model.mjb'), None)
    assert audit_model(out)['pass_model_equivalence'], 'Compiled physical model differs from the accepted baseline'
    provenance=json.loads((BASE.parent/'provenance.json').read_text())
    preserved={}
    for source in [CHECKPOINT,BASE/'rollout.npz',BASE/'rollout_model.mjb']:
        name=str(source.relative_to(Path(__file__).resolve().parents[2]))
        actual=hashlib.sha256(source.read_bytes()).hexdigest();expected=provenance['hashes'][name]
        preserved[name]=dict(sha256=actual,expected=expected,unchanged=actual==expected)
    (out/'baseline_preservation.json').write_text(json.dumps(preserved,indent=2))
    assert all(v['unchanged'] for v in preserved.values()), 'Accepted source artifact changed'
    qs=[];vs=[];us=[];actions=[];rewards=[];frames=[]
    for i in range(steps):
        with torch.inference_mode():
            action=policy(obs) if recorded is None else torch.as_tensor(recorded['actions'][i:i+1])
            obs,reward,done,info=wrapper.step(action)
        qs.append(env.sim.data.qpos[0].numpy().copy());vs.append(env.sim.data.qvel[0].numpy().copy())
        us.append(env.sim.data.ctrl[0].numpy().copy());rewards.append(float(reward[0]));frames.append(int(env.command_manager.get_term('motion').time_steps[0]))
        actions.append(action[0].cpu().numpy().copy())
        if i<5 or i%100==99:
            print('STEP',i+1,'root',qs[-1][:3], 'box',qs[-1][44:47],'elapsed',round(time.time()-start,2),flush=True)
        if bool(done[0]):break
    with np.load(BASE/'rollout.npz') as source:
        phase={k:source[k].copy() for k in source.files if 'step' in k or 'frame' in k}
    audit=env.compact_measurements
    np.savez_compressed(out/'rollout.npz',initial_qpos=q0,initial_qvel=v0,qpos=qs,qvel=vs,ctrl_physics=us,actions=actions,reward=rewards,reference_frame=frames,**phase)
    np.savez_compressed(out/'rollout_verification_trace.npz',trace=audit.trace)
    np.savez_compressed(out/'rollout_locomotion_trace.npz',trace=audit.locomotion)
    np.savez_compressed(out/'contact_attribution.npz',contacts=audit.contact_rows,geom_names=np.array([env.sim.mj_model.geom(i).name or '' for i in range(env.sim.mj_model.ngeom)]))
    report=dict(task=task,refined_reference_path=str(reference_path.resolve()) if reference_path else None,env_class=type(env).__module__+'.'+type(env).__name__,sim_class=type(env.sim).__module__+'.'+type(env.sim).__name__,
                runner_class=type(runner).__module__+'.'+type(runner).__name__,actor_class=type(runner.alg._raw_actor).__name__,
                command_class=type(env.command_manager.get_term('motion')).__name__,
                action_class=type(env.action_manager.get_term('joint_pos')).__name__,
                managers={n:type(getattr(env,n)).__name__ for n in ['command_manager','observation_manager','action_manager','reward_manager','termination_manager']},
                steps=len(qs),physics_dt=env.physics_dt,step_dt=env.step_dt,reset_state_writes=env.command_manager.get_term('motion').reset_state_writes,
                physics_step_counter=env._sim_step_counter,episode_step_counter=int(env.episode_length_buf[0]),
                metric_substeps=len(audit.trace),reward_total=float(sum(rewards)),
                checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),training_performed=False,
                contact_impulses_Ns=dict(audit.sums),bad_impulses_Ns=audit.bad.tolist(),maximum_abs_actuator_force=audit.max_force.tolist(),maximum_applied_external_force=audit.max_external,
                terminal_terms={n:bool(env.termination_manager.get_term(n)[0]) for n in env.termination_manager.active_terms},elapsed=time.time()-start)
    if recorded is not None:
        report['replay_source']=str(replay)
        report['exact_qpos_error']=float(np.max(abs(np.asarray(qs)-recorded['qpos'][:len(qs)])))
        report['exact_qvel_error']=float(np.max(abs(np.asarray(vs)-recorded['qvel'][:len(vs)])))
        report['exact_control_error']=float(np.max(abs(np.asarray(us)-recorded['ctrl_physics'][:len(us)])))
        oldtrace=np.load(replay/'rollout_verification_trace.npz')['trace']
        report['exact_contact_trace_error']=float(np.max(abs(np.asarray(audit.trace)-oldtrace[:len(qs)])))
        oldlocomotion=np.load(replay/'rollout_locomotion_trace.npz')['trace']
        report['exact_locomotion_trace_error']=float(np.max(abs(np.asarray(audit.locomotion)-oldlocomotion[:len(qs)])))
    (out/'execution.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=9629);p.add_argument('--replay',type=Path);p.add_argument('--task',default=TASK_ID);p.add_argument('--reference',type=Path);a=p.parse_args();run(a.out,a.steps,a.replay,a.task,a.reference)
