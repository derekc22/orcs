"""Frozen diffusion body extension: actual ORCS execution, replay and evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
TASK='Orcs-Uolm-Compact-Diffusion-Feedback'


def reproduce(out,resume=False,seeds=(42,43),scale=.6):
    out=out.resolve();out.mkdir(parents=True,exist_ok=resume)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MUJOCO_GL='egl',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    def call(script,*args,log):
        cmd=[sys.executable,'-B',str(ROOT/script),*map(str,args)]
        print('EXECUTE',cmd,flush=True)
        with open(out/log,'w') as f:subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    if not (out/'preservation_before.json').exists():
        tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
        paths=[s for s in tracked if s.startswith(('scripts/compact_diffusion_authority_v1/','scripts/compact_diffusion_v1/','scripts/compact_orcs_v1/','src/orcs/tasks/uolm/compact/','logs/compact_diffusion_authority_v1/'))]
        paths += ['logs/compact_diffusion_authority_v1/feasible_42.npz','logs/compact_diffusion_authority_v1/feasible_43.npz','logs/compact_diffusion_authority_v1/disabled.npz','logs/compact_diffusion_primary_v1/goal3_assessment.json']
        (out/'preservation_before.json').write_text(json.dumps({s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in paths},indent=2))
        (out/'working_tree_before.txt').write_text(subprocess.check_output(['git','status','--porcelain=v1','-uall'],cwd=ROOT,text=True))
    if not (out/'disabled.npz').exists():
        call('scripts/compact_diffusion_body_v1/refine.py','--seed',42,'--disabled','--out',out/'disabled.npz',log='disabled_reference.log')
    if not (out/'inference_recheck/provenance.json').exists():
        call('scripts/compact_diffusion_v1/proposals.py','--out',out/'inference_recheck','--seeds','42,43,0',log='inference_recheck.log')
    if not (out/'disabled/execution.json').exists():
        call('scripts/compact_orcs_v1/run.py','--task',TASK,'--reference',out/'disabled.npz','--out',out/'disabled',log='disabled.log')
    sys.path.insert(0,str(ROOT/'scripts/compact_diffusion_v1'))
    from evidence import physical
    assert physical(out/'disabled')['pass_physical_screen'],'Disabled baseline failed'
    for seed in seeds:
        target=out/f'body_final_{seed}.npz';projection=out/f'body_projection_{seed}'
        feasible=out/f'feasible_{seed}.npz';execution=out/f'execution_{seed}';replay=out/f'replay_{seed}'
        if not target.exists():
            call('scripts/compact_diffusion_body_v1/refine.py','--seed',seed,'--scale',scale,'--out',target,log=f'body_final_{seed}.log')
        config=json.loads(target.with_suffix('.json').read_text())
        assert config['pass_cheap_gate'] and config['scale']==scale
        if not (projection/'execution.json').exists():
            call('scripts/compact_orcs_v1/run.py','--task',TASK,'--reference',target,'--out',projection,log=f'body_projection_{seed}.log')
        if not physical(projection).get('pass_physical_screen',False):
            raise RuntimeError(f'Physical projection {seed} failed; candidate retained for diagnosis')
        if not feasible.exists():
            call('scripts/compact_diffusion_v1/refine.py','realize','--run',projection,'--proposal',target,'--out',feasible,log=f'realize_{seed}.log')
        if not (execution/'execution.json').exists():
            call('scripts/compact_orcs_v1/run.py','--task',TASK,'--reference',feasible,'--out',execution,log=f'execution_{seed}.log')
        if not (replay/'execution.json').exists():
            call('scripts/compact_orcs_v1/run.py','--task',TASK,'--reference',feasible,'--out',replay,'--replay',execution,log=f'replay_{seed}.log')
        if not (execution/'video_metadata.json').exists():
            call('scripts/compact_orcs_v1/render.py',execution,log=f'video_{seed}.log')
        call('scripts/compact_orcs_v1/validate.py',execution,'--replay',replay,log=f'validation_{seed}.log')
        assessment=json.loads((execution/'task_assessment.json').read_text())
        if assessment['status']!='SUCCESS':raise RuntimeError(f'Sample {seed} failed unchanged validation gates')
    if len(seeds)==2:
        call('scripts/compact_diffusion_body_v1/plot.py',out,log='plot.log')
        call('scripts/compact_diffusion_body_v1/report.py',out,log='assessment.log')
        result=json.loads((out/'goal5_assessment.json').read_text())
        print(json.dumps(dict(status=result['status'],gates=result['gates']),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--seed',type=int,choices=[42,43]);p.add_argument('--scale',type=float,default=.6)
    a=p.parse_args();reproduce(a.out,a.resume,(a.seed,) if a.seed else (42,43),a.scale)
