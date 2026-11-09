"""Contact-loaded inverse dynamics for the fixed physical model; no stepping.

The previous small-offset refiner used a free-base RNE difference. Larger
posture changes also change the foot Jacobian/load torque. This module includes
that term, with nonnegative generators of the actual pyramidal friction cone.
"""
import numpy as np
import mujoco
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation as R

from primary_targets import ROOT
import sys
sys.path.insert(0,str(ROOT/'scripts/compact_diffusion_root_v1'))
from root_refine import skew,wrench


def loaded(m,d,loads,actual_native_effort=None):
    native=m.actuator_trnid[:29,0];nv=m.jnt_dofadr[native]
    tau=np.zeros(m.nv);mujoco.mj_rne(m,d,1,tau)
    # Equal-share object wrench includes gravity and object angular dynamics.
    # The accepted baseline's internal grasp-force residual is retained by
    # differential feedforward; no contact or attachment is added to the model.
    F=tau[43:46];M=R.from_quat(d.qpos[47:51],scalar_first=True).apply(tau[46:49])
    hands=[m.body('hand_'+s).id for s in ['left','right']]
    residual_moment=M-sum(np.cross(d.xpos[h]-d.qpos[44:47],F/2) for h in hands)
    effort=tau[nv].copy();jp=np.zeros((3,m.nv));jr=jp.copy()
    for h in hands:
        mujoco.mj_jacBody(m,d,jp,jr,h);effort+=jp[:,nv].T@(F/2)+jr[:,nv].T@(residual_moment/2)
    points=[]
    for side,load in zip(['left','right'],loads):
        if load<30:continue
        candidates=[]
        for g in range(m.ngeom):
            if not (m.geom(g).name or '').startswith(side+'_foot') or not m.geom_contype[g]:continue
            for sign in [-1,1]:
                p=d.geom_xpos[g]+sign*m.geom_size[g,1]*d.geom_xmat[g].reshape(3,3)[:,2]-np.r_[0.,0.,m.geom_size[g,0]]
                candidates.append((p,m.geom_friction[g,0]))
        z=min(p[0][2] for p in candidates);selected=[p for p in candidates if p[0][2]<=z+.003]
        for p,mu in selected:points.append((p,m.body(side+'_ankle_roll_link').id,mu,float(load)/len(selected)))
    if not points:return dict(feasible=False,reason='No authoritative foot support')
    W=[];G=[];prior=[]
    for p,body,mu,fz in points:
        p=p.copy();p[2]=0.;mujoco.mj_jac(m,d,jp,jr,p,body)
        for f in [np.r_[mu,0.,1.],np.r_[-mu,0.,1.],np.r_[0.,mu,1.],np.r_[0.,-mu,1.]]:
            W.append(np.r_[f,np.cross(p-d.subtree_com[0],f)]);G.append(jp[:,nv].T@f);prior.append(fz/4)
    W=np.array(W).T;G=np.array(G).T;prior=np.array(prior);target=wrench(m,d,tau)
    prior*=max(target[2],1.)/max(4*sum(p[3] for p in points),1.)*4
    A=[1e5*W,.03*np.eye(len(prior))];b=[1e5*target,.03*prior]
    if actual_native_effort is not None:
        leg=np.array([i for i,j in enumerate(native) if any(t in m.joint(j).name for t in ['hip_','knee_','ankle_'])])
        A.append(.3*G[leg]);b.append(.3*(effort[leg]-actual_native_effort[leg]))
    solution=lsq_linear(np.concatenate(A),np.concatenate(b),bounds=(0.,np.inf),method='bvls',tol=1e-10).x
    required=effort-G@solution;error=W@solution-target
    cap=np.max(abs(m.actuator_forcerange[:29]),axis=1)
    return dict(feasible=bool(abs(error).max()<.001 and (abs(required)-cap).max()<.01),
        effort=required,wrench_error=error,maximum_effort_excess_Nm=float((abs(required)-cap).max()),
        contact_generators=solution,whole_system_wrench=target)
