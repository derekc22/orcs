"""Reproduce the strongest verified Goal 7 pickup, without rejected variants."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
TASK='Orcs-Uolm-Compact-Diffusion-Feedback'


def reproduce(out,resume=False):
    out=out.resolve();out.mkdir(parents=True,exist_ok=resume)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MUJOCO_GL='egl',PYTHONDONTWRITEBYTECODE='1')
    def call(script,*args,log):
        with open(out/log,'w') as f:
            subprocess.run([sys.executable,'-B',str(ROOT/script),*map(str,args)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    for seed in [42,43]:
        source=out/f'pickup_{seed}.npz';projection=out/f'pickup_projection_{seed}'
        feasible=out/f'pickup_feasible_{seed}.npz';execution=out/f'pickup_execution_{seed}';replay=out/f'pickup_replay_{seed}'
        if not source.exists():call('scripts/compact_diffusion_pose_v1/primary_compile.py','--seed',seed,'--phase','pickup','--out',source,log=f'pickup_{seed}.log')
        if not (projection/'execution.json').exists():call('scripts/compact_orcs_v1/run.py','--task',TASK,'--reference',source,'--out',projection,log=f'pickup_projection_{seed}.log')
        call('scripts/compact_diffusion_v1/evidence.py','physical',projection,log=f'pickup_screen_{seed}.log')
        import json
        assert json.loads((projection/'physical_screen.json').read_text())['pass_physical_screen']
        if not feasible.exists():call('scripts/compact_diffusion_v1/refine.py','realize','--run',projection,'--proposal',source,'--out',feasible,log=f'pickup_realize_{seed}.log')
        if not (execution/'execution.json').exists():call('scripts/compact_orcs_v1/run.py','--task',TASK,'--reference',feasible,'--out',execution,log=f'pickup_execution_{seed}.log')
        if not (replay/'execution.json').exists():call('scripts/compact_orcs_v1/run.py','--task',TASK,'--reference',feasible,'--out',replay,'--replay',execution,log=f'pickup_replay_{seed}.log')
        if not (execution/'video_metadata.json').exists():call('scripts/compact_orcs_v1/render.py',execution,log=f'pickup_video_{seed}.log')
        if not (execution/'task_assessment.json').exists():call('scripts/compact_orcs_v1/validate.py',execution,'--replay',replay,log=f'pickup_validation_{seed}.log')
        assert json.loads((execution/'task_assessment.json').read_text())['status']=='SUCCESS'
    call('scripts/compact_diffusion_pose_v1/primary_report.py','--out',out,log='pickup_assessment.log')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();reproduce(a.out,a.resume)
