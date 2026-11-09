"""Reproduce the bounded Goal 7 candidate through genuine ORCS and replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
G6=ROOT/'logs/compact_diffusion_root_v1'
TASK='Orcs-Uolm-Compact-Diffusion-Feedback'


def reproduce(out,resume=False,seeds=(42,43)):
    out=out.resolve();out.mkdir(parents=True,exist_ok=resume)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MUJOCO_GL='egl',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    def call(script,*args,log):
        cmd=[sys.executable,'-B',str(ROOT/script),*map(str,args)];print('EXECUTE',cmd,flush=True)
        with open(out/log,'w') as f:subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    if not (out/'preservation_before.json').exists():
        tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
        paths=[s for s in tracked if s.startswith(('scripts/compact_diffusion_root_v1/','scripts/compact_diffusion_body_v1/','scripts/compact_diffusion_authority_v1/','scripts/compact_diffusion_v1/','scripts/compact_orcs_v1/','src/orcs/tasks/uolm/compact/','logs/compact_diffusion_root_v1/'))]
        paths += ['logs/compact_diffusion_root_v1/feasible_42.npz','logs/compact_diffusion_root_v1/feasible_43.npz','logs/compact_diffusion_root_v1/disabled.npz','logs/compact_policy_learning_v1/selected_policy.npz']
        (out/'preservation_before.json').write_text(json.dumps({s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in paths},indent=2))
        (out/'working_tree_before.txt').write_text(subprocess.check_output(['git','status','--porcelain=v1','-uall'],cwd=ROOT,text=True))
    if not (out/'disabled.npz').exists():
        shutil.copyfile(G6/'disabled.npz',out/'disabled.npz')
        (out/'disabled.json').write_text(json.dumps(dict(source=str(G6/'disabled.npz'),sha256=hashlib.sha256((G6/'disabled.npz').read_bytes()).hexdigest()),indent=2))
    if not (out/'disabled/execution.json').exists():
        call('scripts/compact_orcs_v1/run.py','--task',TASK,'--reference',out/'disabled.npz','--out',out/'disabled',log='disabled.log')
    sys.path.insert(0,str(ROOT/'scripts/compact_diffusion_v1'));from evidence import physical
    assert physical(out/'disabled')['pass_physical_screen']
    for seed in seeds:
        target=out/f'safe_{seed}.npz';projection=out/f'projection_{seed}'
        feasible=out/f'feasible_{seed}.npz';execution=out/f'execution_{seed}';replay=out/f'replay_{seed}'
        if not target.exists():
            call('scripts/compact_diffusion_pose_v1/refine.py','--seed',seed,'--scale',.25,'--out',target,log=f'safe_{seed}.log')
        config=json.loads(target.with_suffix('.json').read_text());assert config['pass_cheap_gate'] and config['scale']==.25
        if not (projection/'execution.json').exists():
            call('scripts/compact_orcs_v1/run.py','--task',TASK,'--reference',target,'--out',projection,log=f'projection_{seed}.log')
        if not physical(projection)['pass_physical_screen']:raise RuntimeError(f'Sample {seed} projection failed; retain Goal 6 and diagnose recorded evidence')
        if not feasible.exists():
            call('scripts/compact_diffusion_v1/refine.py','realize','--run',projection,'--proposal',target,'--out',feasible,log=f'realize_{seed}.log')
        if not (execution/'execution.json').exists():
            call('scripts/compact_orcs_v1/run.py','--task',TASK,'--reference',feasible,'--out',execution,log=f'execution_{seed}.log')
        if not (replay/'execution.json').exists():
            call('scripts/compact_orcs_v1/run.py','--task',TASK,'--reference',feasible,'--out',replay,'--replay',execution,log=f'replay_{seed}.log')
        if not (execution/'video_metadata.json').exists():call('scripts/compact_orcs_v1/render.py',execution,log=f'video_{seed}.log')
        if not (execution/'task_assessment.json').exists():call('scripts/compact_orcs_v1/validate.py',execution,'--replay',replay,log=f'validation_{seed}.log')
        a=json.loads((execution/'task_assessment.json').read_text());assert a['status']=='SUCCESS'
    if len(seeds)==2:
        call('scripts/compact_diffusion_pose_v1/report.py',out,log='assessment.log')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--seed',type=int,choices=[42,43]);a=p.parse_args()
    reproduce(a.out,a.resume,(a.seed,) if a.seed else (42,43))
