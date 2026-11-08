"""Reproduce frozen inference, two managed physical executions and exact replays."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
TASK='Orcs-Uolm-Compact-Diffusion-Feedback'


def reproduce(out,resume=False,seeds=(42,43)):
    out=out.resolve()
    out.mkdir(parents=True,exist_ok=resume)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',MUJOCO_GL='egl',PYTHONUNBUFFERED='1')
    def call(script,*args,log):
        command=[sys.executable,'-B',str(ROOT/script),*map(str,args)]
        print('EXECUTE',command,flush=True)
        with open(out/log,'w') as f:
            subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    snapshot=out/'preservation_before.json'
    if not snapshot.exists():
        folders=['scripts/compact_diffusion_v1','scripts/compact_orcs_v1','src/orcs/tasks/uolm/compact','logs/compact_diffusion_v1','logs/compact_orcs_v1/complete_execution']
        paths=[p for d in folders for p in (ROOT/d).rglob('*') if p.is_file() and '__pycache__' not in str(p)]
        snapshot.write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))
        (out/'working_tree_before.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))
    disabled=out/'disabled.npz'
    if not disabled.exists():
        call('scripts/compact_diffusion_authority_v1/refine.py','--seed',42,'--disabled','--out',disabled,log='disabled_reference.log')
    if not (out/'inference_recheck/provenance.json').exists():
        call('scripts/compact_diffusion_v1/proposals.py','--out',out/'inference_recheck','--seeds','42,43,0',log='inference_recheck.log')
    if not (out/'disabled/execution.json').exists():
        call('scripts/compact_orcs_v1/run.py','--task',TASK,'--reference',disabled,'--out',out/'disabled',log='disabled.log')
    sys.path.insert(0,str(ROOT/'scripts/compact_diffusion_v1'))
    from evidence import physical
    if not physical(out/'disabled')['pass_physical_screen']:
        raise RuntimeError('Diffusion-disabled baseline did not reproduce physical success')
    for seed in seeds:
        target=out/f'dense_{seed}.npz';projection=out/f'projection_{seed}'
        feasible=out/f'feasible_{seed}.npz';execution=out/f'execution_{seed}';replay=out/f'replay_{seed}'
        if not target.exists():
            call('scripts/compact_diffusion_authority_v1/refine.py','--seed',seed,'--out',target,log=f'dense_{seed}.log')
        if not (projection/'execution.json').exists():
            call('scripts/compact_orcs_v1/run.py','--task',TASK,'--reference',target,'--out',projection,log=f'projection_{seed}.log')
        if not physical(projection).get('pass_physical_screen',False):
            raise RuntimeError(f'Sample {seed} failed physical projection; retained for diagnosis')
        if not feasible.exists():
            call('scripts/compact_diffusion_v1/refine.py','realize','--run',projection,'--proposal',target,'--out',feasible,log=f'realize_{seed}.log')
        if not (execution/'execution.json').exists():
            call('scripts/compact_orcs_v1/run.py','--task',TASK,'--reference',feasible,'--out',execution,log=f'execution_{seed}.log')
        if not (replay/'execution.json').exists():
            call('scripts/compact_orcs_v1/run.py','--task',TASK,'--reference',feasible,'--out',replay,'--replay',execution,log=f'replay_{seed}.log')
        if not (execution/'video_metadata.json').exists():
            call('scripts/compact_orcs_v1/render.py',execution,log=f'video_{seed}.log')
        call('scripts/compact_orcs_v1/validate.py',execution,'--replay',replay,log=f'validation_{seed}.log')
        if json.loads((execution/'task_assessment.json').read_text())['status']!='SUCCESS':
            raise RuntimeError(f'Sample {seed} failed one or more unchanged physical gates')
    if len(seeds)==2:
        call('scripts/compact_diffusion_authority_v1/plot.py',out,log='plot.log')
        call('scripts/compact_diffusion_authority_v1/report.py',out,log='assessment.log')
        result=json.loads((out/'goal4_assessment.json').read_text())
        print(json.dumps(dict(status=result['status'],gates=result['gates']),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--seed',type=int,choices=[42,43])
    a=p.parse_args();reproduce(a.out,a.resume,(a.seed,) if a.seed else (42,43))
