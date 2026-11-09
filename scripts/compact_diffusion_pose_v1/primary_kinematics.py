"""Hierarchical source-pose projection with fixed soles and bilateral grasp."""
import mujoco
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation as R

from primary_targets import ROOT
import sys
sys.path.insert(0,str(ROOT/'scripts/compact_diffusion_root_v1'))
from root_refine import left_jacobian,skew


class Projector:
    def __init__(self,m,base):
        self.m=m;self.base=base;self.old=mujoco.MjData(m);self.fk=mujoco.MjData(m);self.warm=mujoco.MjData(m)
        self.joints=[j for j in range(m.njnt) if any(t in m.joint(j).name for t in ['hip_','knee_','ankle_','shoulder_','elbow_','wrist_'])]
        self.jq=m.jnt_qposadr[self.joints];self.jv=m.jnt_dofadr[self.joints]
        self.legs=np.array([k for k,j in enumerate(self.joints) if any(t in m.joint(j).name for t in ['hip_','knee_','ankle_'])]);self.arms=np.array([k for k in range(26) if k not in self.legs])
        self.waist=[m.joint('waist_'+n+'_joint').id for n in ['roll','pitch']];self.wq=m.jnt_qposadr[self.waist];self.wv=m.jnt_dofadr[self.waist]
        self.feet=[m.body(s+'_ankle_roll_link').id for s in ['left','right']];self.hands=[m.body('hand_'+s).id for s in ['left','right']];self.ob=m.body('object/object').id
        self.B=np.zeros((m.nv,38));self.B[:2,:2]=np.eye(2);self.B[np.ix_(self.jv,np.arange(2,28))]=np.eye(26);self.B[43:46,28:31]=np.eye(3)
        self.B[np.ix_(self.wv,np.arange(36,38))]=np.eye(2);self.jp=np.zeros((3,m.nv));self.jr=self.jp.copy()
        def excess(joints,cols):return np.maximum(np.maximum(base['qref'][:,cols]-m.jnt_range[joints,1],m.jnt_range[joints,0]-base['qref'][:,cols]),0.).max(0)
        self.excess=excess(self.joints,self.jq);self.we=excess(self.waist,self.wq)
        boxes=[g for g in range(m.ngeom) if m.geom_bodyid[g]==self.ob and m.geom_contype[g]]
        robot=[g for g in range(m.ngeom) if m.geom_bodyid[g]>0 and m.geom_bodyid[g]!=self.ob and m.geom_contype[g] and not (m.geom(g).name or '').startswith('hand_')]
        self.box_pairs=[(r,b) for r in robot for b in boxes if (m.geom_contype[r]&m.geom_conaffinity[b]) or(m.geom_contype[b]&m.geom_conaffinity[r])]
        self.box_pair_array=np.asarray(self.box_pairs,dtype=int)
        self.box_radius_squared=(m.geom_rbound[self.box_pair_array[:,0]]+m.geom_rbound[self.box_pair_array[:,1]]+.02)**2
        terrain=[g for g in range(m.ngeom) if (m.geom(g).name or '')=='terrain'];self.ground_pairs=[(g,terrain[0]) for g in boxes] if terrain else []
        self.self_pairs=set()
        for n1,n2 in [('torso_collision','left_shoulder_yaw_collision')]:
            try:self.self_pairs.add((m.geom(n1).id,m.geom(n2).id))
            except KeyError:pass

    def forbidden(self,c):
        m=self.m;g1,g2=map(int,c.geom);b1,b2=m.geom_bodyid[[g1,g2]];n1,n2=[m.geom(g).name or '' for g in [g1,g2]]
        return (self.ob in [b1,b2] and not(n1.startswith('hand_') or n2.startswith('hand_') or 'terrain' in [n1,n2])) or(self.ob not in [b1,b2] and b1>0 and b2>0) or('terrain' in [n1,n2] and not('foot' in n1 or 'foot' in n2 or self.ob in [b1,b2]))

    def project(self,tick,target,qwarm,previous=None,max_iterations=40,body_projection=True,com_override=None,com_tube=.001,continuity=None,fixed_motion=False,fixed_angles=False):
        m=self.m;q0=self.base['qref'][tick];old=self.old;fk=self.fk;B=self.B;jp,jr=self.jp,self.jr
        old.qpos[:]=q0;mujoco.mj_fwdPosition(m,old);positions=old.xpos[self.feet].copy();rotations=R.from_matrix(old.xmat[self.feet].reshape(2,3,3));rel=old.xpos[self.hands]-q0[44:47];handrot=R.from_matrix(old.xmat[self.hands].reshape(2,3,3))
        self.warm.qpos[:]=qwarm;mujoco.mj_fwdPosition(m,self.warm);com=self.warm.subtree_com[0,:2].copy() if com_override is None else np.asarray(com_override)
        desired=np.r_[target[:2]-q0[:2],target[self.jq]-q0[self.jq],target[44:47]-q0[44:47],(R.from_quat(target[47:51],scalar_first=True)*R.from_quat(q0[47:51],scalar_first=True).inv()).as_rotvec(),np.zeros(4)]
        desired[2+self.legs]=qwarm[self.jq[self.legs]]-q0[self.jq[self.legs]]
        low=np.r_[-.20,-.20,m.jnt_range[self.joints,0]-self.excess-q0[self.jq],np.repeat(-.4,3),np.repeat(-1.3,3),-.15,-.15,m.jnt_range[self.waist,0]-self.we-target[self.wq]]
        high=np.r_[.20,.20,m.jnt_range[self.joints,1]+self.excess-q0[self.jq],np.repeat(.4,3),np.repeat(1.3,3),.15,.15,m.jnt_range[self.waist,1]+self.we-target[self.wq]]
        weights=np.r_[1.,1.,np.repeat(.01,26),np.ones(3),np.repeat(.2,3),3.,3.,1.,1.];weights[2+self.arms]=.05
        baseline_ground=min([mujoco.mj_geomDistance(m,old,g1,g2,.03,None) for g1,g2 in self.ground_pairs] or [.01]);ground_clearance=np.clip(baseline_ground,0.,.008)
        def encode(q):return np.r_[q[:2]-q0[:2],q[self.jq]-q0[self.jq],q[44:47]-q0[44:47],(R.from_quat(q[47:51],scalar_first=True)*R.from_quat(q0[47:51],scalar_first=True).inv()).as_rotvec(),(R.from_quat(q[3:7],scalar_first=True)*R.from_quat(target[3:7],scalar_first=True).inv()).as_rotvec()[:2],q[self.wq]-target[self.wq]]
        if continuity is not None:
            prior,older=map(encode,continuity)
            predicted=2*prior-older
        def residual(p,allow_body,jac=False):
            fk.qpos[:]=q0;fk.qpos[2]=target[2];fk.qpos[3:7]=(R.from_rotvec(np.r_[p[34:36],0.])*R.from_quat(target[3:7],scalar_first=True)).as_quat(scalar_first=True);fk.qpos[19:22]=target[19:22];fk.qpos[self.wq]+=p[36:38]
            fk.qpos[:2]+=p[:2];fk.qpos[self.jq]+=p[2:28];fk.qpos[44:47]+=p[28:31];rot=R.from_rotvec(p[31:34]);fk.qpos[47:51]=(rot*R.from_quat(q0[47:51],scalar_first=True)).as_quat(scalar_first=True)
            mujoco.mj_fwdPosition(m,fk);B[46:49,31:34]=R.from_quat(fk.qpos[47:51],scalar_first=True).inv().as_matrix()@left_jacobian(p[31:34])
            B[3:6,34:36]=R.from_quat(fk.qpos[3:7],scalar_first=True).inv().as_matrix()@left_jacobian(np.r_[p[34:36],0.])[:,:2]
            errors=[];rows=[]
            for k,h in enumerate(self.feet):
                errors.extend([1000*(positions[k]-fk.xpos[h]),300*(rotations[k]*R.from_matrix(fk.xmat[h].reshape(3,3)).inv()).as_rotvec()])
                if jac:mujoco.mj_jacBody(m,fk,jp,jr,h);rows.extend([1000*jp@B,300*jr@B])
            for k,h in enumerate(self.hands):
                rr=rot.apply(rel[k]);errors.extend([1000*(fk.qpos[44:47]+rr-fk.xpos[h]),300*(rot*handrot[k]*R.from_matrix(fk.xmat[h].reshape(3,3)).inv()).as_rotvec()])
                if jac:
                    mujoco.mj_jacBody(m,fk,jp,jr,h);pp=jp@B;aa=jr@B;pp[:,28:31]-=np.eye(3);pp[:,31:34]+=skew(rr)@left_jacobian(p[31:34]);aa[:,31:34]-=left_jacobian(p[31:34]);rows.extend([1000*pp,300*aa])
            offset=fk.subtree_com[0,:2]-com;norm=np.linalg.norm(offset)
            normal=offset/max(norm,1e-12)
            errors.extend([10*(com-fk.subtree_com[0,:2]),np.array([300*min(0.,com_tube-norm)]),weights*(desired-p)])
            if jac:
                mujoco.mj_jacSubtreeCom(m,fk,jp,0);rows.extend([10*jp[:2]@B,300*normal[None]@jp[:2]@B if norm>com_tube else np.zeros((1,38)),np.diag(weights)])
            if continuity is not None:
                # Soft acceleration and velocity costs on transported offsets,
                # with the same exact sole/grasp geometry. These avoid changing
                # solution branches independently at every physics tick.
                errors.extend([20*(predicted-p),4*(prior-p)])
                if jac:rows.extend([20*np.eye(38),4*np.eye(38)])
            for c in fk.contact:
                if self.forbidden(c):self.self_pairs.add(tuple(map(int,c.geom)))
            delta_centres=fk.geom_xpos[self.box_pair_array[:,0]]-fk.geom_xpos[self.box_pair_array[:,1]]
            near=np.sum(delta_centres**2,axis=1)<=self.box_radius_squared
            pairs=[tuple(v) for v in self.box_pair_array[near]]
            for g1,g2 in [*pairs,*self.self_pairs,*self.ground_pairs]:
                ground=(g1,g2) in self.ground_pairs;margin=ground_clearance if ground else .008
                plane=int(m.geom_type[g1])==int(mujoco.mjtGeom.mjGEOM_PLANE) or int(m.geom_type[g2])==int(mujoco.mjtGeom.mjGEOM_PLANE)
                if not ground and not plane and np.linalg.norm(fk.geom_xpos[g1]-fk.geom_xpos[g2])>m.geom_rbound[g1]+m.geom_rbound[g2]+.02:continue
                ft=np.zeros(6);distance=mujoco.mj_geomDistance(m,fk,g1,g2,.025,ft)
                if distance>=margin:continue
                errors.append(np.array([1500*(margin-distance)]))
                if jac:
                    normal=(ft[3:]-ft[:3])/distance if abs(distance)>1e-9 else np.r_[0.,0.,1.];J=[]
                    for g,point in [(g1,ft[:3]),(g2,ft[3:])]:mujoco.mj_jac(m,fk,jp,jr,point,int(m.geom_bodyid[g]));J.append(jp@B)
                    rows.append(1500*normal[None]@(J[1]-J[0]))
            return np.concatenate(errors),np.concatenate(rows) if jac else None
        best=None
        for allow_body in ([False] if fixed_motion or fixed_angles else ([True] if continuity is not None else ([False,True] if body_projection else [False]))):
            guess=encode(previous) if previous is not None else (encode(qwarm) if allow_body else desired.copy())
            p=np.clip(guess,low,high)
            if not allow_body:p[34:]=0.
            if fixed_motion:
                p[:2]=desired[:2];p[28:34]=desired[28:34]
            if fixed_angles:p[31:34]=desired[31:34]
            cols=np.arange(2,28) if fixed_motion else (np.arange(31) if fixed_angles else np.arange(38 if allow_body else 34))
            for _ in range(max_iterations):
                err,A=residual(p,allow_body,True);stepbox=np.r_[.005,.005,np.repeat(.05,26),np.repeat(.01,3),np.repeat(.03,3),np.repeat(.02,4)]
                step=np.zeros(38);step[cols]=lsq_linear(A[:,cols],err,bounds=(np.maximum(low-p,-stepbox)[cols]-1e-12,np.minimum(high-p,stepbox)[cols]+1e-12),method='bvls',tol=1e-10).x
                ok=False
                for rate in [1.,.5,.25,.125,.0625]:
                    trial=p+rate*step;ee,_=residual(trial,allow_body)
                    if ee@ee<=err@err+1e-10:p=trial;ok=True;break
                if not ok or np.linalg.norm(step)<1e-8:break
            err,_=residual(p,allow_body);q=fk.qpos.copy()
            pe=max(np.linalg.norm(err[k:k+3])/1000 for k in [0,6,12,18]);ae=max(np.linalg.norm(err[k:k+3])/300 for k in [3,9,15,21]);ce=np.linalg.norm(com-fk.subtree_com[0,:2]);bad=[(int(c.geom[0]),int(c.geom[1]),float(c.dist)) for c in fk.contact if self.forbidden(c) and c.dist<0.]
            score=max(pe/.0003,ae/.003,max(0.,ce-com_tube)/.001,1e4*max([-v[2] for v in bad] or [0.]))
            result=dict(q=q,parameters=p,position_error_m=float(pe),angle_error_rad=float(ae),COM_error_m=float(ce),COM_tube_m=com_tube,COM_tube_excess_m=float(max(0.,ce-com_tube)),forbidden_penetrations=bad,body_projection=allow_body,score=float(score))
            if best is None or score<best['score']:best=result
            if pe<.0002 and ae<.002 and ce<com_tube+.0001 and not bad:break
        return best
