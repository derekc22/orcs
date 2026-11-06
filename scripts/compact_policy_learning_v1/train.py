"""Empirical local dynamics identification and finite-horizon policy optimization.

Fit 98-state / 37-action, contact-dependent transition regressions from 256
orthogonal randomized physical probes at each 20ms reference interval. Held-out
Gaussian probes measure prediction error. Bellman/Riccati recursion optimizes
the time-varying linear residual policy on these learned dynamics.
"""
import argparse
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path
import mujoco
import numpy as np
from scipy.linalg import hadamard
from controller import SOURCE, MODEL, OUT, error

SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    a = np.load(SOURCE)
    m = mujoco.MjModel.from_binary_path(str(MODEL))
    d = mujoco.MjData(m)
    d.qpos[:] = a['initial_qpos']; d.qvel[:] = a['initial_qvel']
    mujoco.mj_forward(m, d)
    states = []; q = []; v = []
    for u in a['ctrl_physics']:
        state = np.empty(mujoco.mj_stateSize(m, SPEC))
        mujoco.mj_getState(m, d, state, SPEC)
        states.append(state); q.append(d.qpos.copy()); v.append(d.qvel.copy())
        d.ctrl[:] = u; mujoco.mj_step(m, d)
    assert np.array_equal(d.qpos, a['qpos'][-1])
    assert np.array_equal(d.qvel, a['qvel'][-1])
    np.savez(OUT/'reference.npz', integration=states, qref=q, vref=v, uref=a['ctrl_physics'])
    return dict(source=str(SOURCE), baseline_commit='f65e57e648db0a0d8728c8650b42a597797da9b5',
                source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                reference_exact=True, states=len(states), nv=m.nv, nu=m.nu)


def fit_chunk(args):
    indices, stride, seed, samples, run_dir = args
    run_dir=Path(run_dir)
    m = mujoco.MjModel.from_binary_path(str(MODEL)); d = mujoco.MjData(m)
    r = np.load(run_dir/'reference.npz'); qref=r['qref']; vref=r['vref']; controls=r['uref']; states=r['integration']
    n = 2*m.nv; p=n+m.nu
    # Small probes identify tangent dynamics; these are training data, not the
    # independently frozen initial-state robustness distribution.
    scale=np.r_[np.full(m.nv, 5e-5), np.full(m.nv, 5e-4), np.full(m.nu, 1e-4)]
    scale[:3]=1e-5; scale[43:46]=5e-6; scale[n+29:n+31]=1e-5
    rng=np.random.default_rng(seed)
    H=hadamard(samples).astype(float)[:, 1:p+1]
    H *= rng.choice([-1.,1.],p)
    fits=[]; diagnostics=[]
    for block in indices:
        start=block*stride; stop=min(start+stride,len(controls)); state=states[start]
        def transition(x):
            mujoco.mj_setState(m,d,state,SPEC)
            mujoco.mj_integratePos(m,d.qpos,x[:m.nv],1.)
            d.qvel[:] += x[m.nv:n]
            for u in controls[start:stop]:
                d.ctrl[:]=np.clip(u+x[n:],m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
                mujoco.mj_step(m,d)
            return d.qpos.copy(), d.qvel.copy()
        baseq,basev=transition(np.zeros(p))
        exact=float(np.max(abs(baseq-qref[stop]))) if stop<len(qref) else 0.
        Y=np.array([error(m,baseq,basev,*transition(h*scale)) for h in H])
        W=(H.T @ Y)/samples/scale[:,None]
        A=W[:n].T; B=W[n:].T
        fits.append((block,A,B))
        if block%25==0 or len(indices)<20:
            X=rng.normal(size=(12,p))*scale*.5
            actual=np.array([error(m,baseq,basev,*transition(x)) for x in X])
            pred=X @ W
            normal=np.maximum(np.std(Y,axis=0),1e-9)
            rel=float(np.linalg.norm((actual-pred)/normal)/max(np.linalg.norm(actual/normal),1e-9))
            diagnostics.append(dict(block=int(block),time_s=start*.005,relative_validation_error=rel,
                                    restored_nominal_error=exact,fit_residual=float(np.linalg.norm(H @ (W*scale[:,None])-Y)/max(np.linalg.norm(Y),1e-15))))
    return fits,diagnostics


def optimize(model_path, output, cost_profile='default'):
    r=np.load(OUT/'reference.npz'); models=np.load(model_path)
    A=models['A']; B=models['B']; n=A.shape[1]; nu=B.shape[2]; nv=n//2
    q=np.r_[np.full(nv,100.), np.full(nv,1.)]
    q[:3]=1e4; q[3:6]=2000.; q[nv:nv+3]=100.;q[nv+3:nv+6]=20.
    q[43:46]=1e4; q[46:49]=300.; q[nv+43:nv+46]=100.;q[nv+46:nv+49]=2.
    R=np.full(nu,100.); R[29:31]=1e6
    if cost_profile=='grasp':
        q[43:46]*=10.;q[46:49]*=10.;R[:10]*=.3;R[19:23]*=.3
    Q=np.diag(q); RR=np.diag(R); P=Q*10; K=np.empty((len(A),nu,n))
    for i in range(len(A)-1,-1,-1):
        PA=P @ A[i]; PB=P @ B[i]
        gain=np.linalg.solve(RR+B[i].T @ PB,B[i].T @ PA)
        K[i]=gain; P=Q+A[i].T @ PA-A[i].T @ PB @ gain; P=(P+P.T)*.5
    bound=np.full(nu,.15);bound[29:31]=.003;bound[33:37]=.10
    np.savez_compressed(output,K=K,qref=r['qref'],vref=r['vref'],uref=r['uref'],
                        stride=models['stride'],action_bound=bound,state_cost=q,action_cost=R)
    return dict(checkpoint=str(output),cost_profile=cost_profile,maximum_gain=float(abs(K).max()))


def main():
    global OUT
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['pilot','train','optimize'],default='pilot')
    p.add_argument('--workers',type=int,default=8);p.add_argument('--stride',type=int,default=4)
    p.add_argument('--samples',type=int,default=256);p.add_argument('--seed',type=int,default=11052026)
    p.add_argument('--cost-profile',default='default');p.add_argument('--output',type=Path)
    p.add_argument('--run-dir',type=Path,default=OUT)
    args=p.parse_args();OUT=args.run_dir;args.output=args.output or OUT/'policy.npz';begin=time.time()
    if args.mode=='optimize':
        print(json.dumps(optimize(OUT/'dynamics.npz',args.output,args.cost_profile)));return
    meta=prepare();count=(meta['states']+args.stride-1)//args.stride
    blocks=np.array([0,125,500,850,875,1050,1175,1475,1700,1900,2000,2200,2350]) if args.mode=='pilot' else np.arange(count)
    chunks=[x for x in np.array_split(blocks,args.workers) if len(x)]
    allfits=[];diags=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(fit_chunk,(chunk,args.stride,args.seed+i,args.samples,OUT)) for i,chunk in enumerate(chunks)]
        for future in concurrent.futures.as_completed(futures):
            fits,diag=future.result();allfits.extend(fits);diags.extend(diag)
            print('identified',len(allfits),'/',len(blocks),'elapsed',round(time.time()-begin,1),flush=True)
    allfits.sort(key=lambda x:x[0]);diags.sort(key=lambda x:x['block'])
    np.savez_compressed(OUT/('pilot_dynamics.npz' if args.mode=='pilot' else 'dynamics.npz'),
                        blocks=[x[0] for x in allfits],A=[x[1] for x in allfits],B=[x[2] for x in allfits],stride=args.stride)
    meta.update(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                training_transitions=len(blocks)*args.samples,validation=diags,elapsed_s=time.time()-begin)
    if args.mode=='train':meta['optimization']=optimize(OUT/'dynamics.npz',args.output,args.cost_profile)
    (OUT/('pilot.json' if args.mode=='pilot' else 'training.json')).write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2),flush=True)

if __name__=='__main__':main()
