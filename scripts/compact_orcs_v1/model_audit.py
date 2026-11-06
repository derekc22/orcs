"""Compare the actual compiled ORCS physical model with the accepted system."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def audit(path):
    old=mujoco.MjModel.from_binary_path(str(ROOT/'logs/compact_policy_learning_v1/complete_execution/rollout_model.mjb'))
    m=mujoco.MjModel.from_binary_path(str(path/'rollout_model.mjb'))
    # Scene assembly preserves body/joint/geom/actuator order. Names are
    # namespaced by entities; no physical permutation is inferred numerically.
    def name(model,kind,i):
        n=mujoco.mj_id2name(model,kind,i) or ''
        if n.startswith('robot/'):n=n[6:]
        if 'compact_geom_' in n:n=''
        return n
    names={}
    for kind,count in [(mujoco.mjtObj.mjOBJ_BODY,m.nbody),(mujoco.mjtObj.mjOBJ_JOINT,m.njnt),(mujoco.mjtObj.mjOBJ_ACTUATOR,m.nu)]:
        names[str(kind)]=all(name(old,kind,i)==name(m,kind,i) for i in range(count))
    errors={}
    for prefix,fields in {
        'body':['pos','quat','ipos','iquat','mass','inertia','invweight0','parentid'],
        'geom':['pos','quat','size','contype','conaffinity','condim','friction','solref','solimp','margin','gap','rbound','aabb','dataid'],
        'jnt':['pos','axis','range','margin','stiffness','solref','solimp','limited','type'],
        'dof':['damping','armature','frictionloss','M0','invweight0'],
        'actuator':['gainprm','biasprm','gear','ctrlrange','forcerange','ctrllimited','forcelimited','trnid','acc0'],
        'mesh':['vert','face','normal','facenormal','pos','quat','graph','polynormal','polyvert','polyvertadr','polyvertnum','polymap','polyadr','polynum','polymapadr','polymapnum'],
    }.items():
        for f in fields:
            field=prefix+'_'+f;a=getattr(old,field);b=getattr(m,field)
            errors[field]=float(np.max(abs(a-b),initial=0)) if a.shape==b.shape else None
    for field in ['qpos0','qpos_spring','exclude_signature']:
        errors[field]=float(np.max(abs(getattr(old,field)-getattr(m,field)),initial=0))
    options={}
    for k in dir(old.opt):
        if k.startswith('_'):continue
        a=getattr(old.opt,k);b=getattr(m.opt,k)
        if isinstance(a,np.ndarray):options[k]=bool(np.array_equal(a,b))
        elif isinstance(a,(int,float)):options[k]=a==b
    result=dict(pass_model_equivalence=all(v==0 for v in errors.values()) and all(names.values()) and all(options.values()),
                dimensions=dict(nq=m.nq,nv=m.nv,nu=m.nu,nbody=m.nbody,ngeom=m.ngeom,nmesh=m.nmesh),
                ordering_matches=names,physical_array_max_errors=errors,options_equal=options,
                equality_constraints=int(m.neq),mocap_bodies=int(m.nmocap),tendons=int(m.ntendon),
                note='Entity namespaces and rendering-only fields differ; physical parameters and compiled solver constants are compared explicitly.')
    (path/'model_equivalence.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('artifact',type=Path);a=p.parse_args();audit(a.artifact)
