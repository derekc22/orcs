"""Reproduce the frozen, bounded diffusion/ORCS experiment without training.

Creates a new artifact directory. Existing accepted artifacts are read only.
Every physical rollout uses the registered ORCS environment and native runner.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from evidence import difference

ROOT=Path(__file__).resolve().parents[2]
TASK='Orcs-Uolm-Compact-Diffusion-Feedback'


def reproduce(out):
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1',MUJOCO_GL='egl')
    def call(script,*args,log):
        cmd=[sys.executable,str(ROOT/script),*map(str,args)]
        print('EXECUTE',cmd,flush=True)
        with open(out/log,'w') as f:subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    call('scripts/compact_diffusion_v1/proposals.py','--out',out/'proposals',log='proposals.log')
    call('scripts/compact_diffusion_v1/proposals.py','--out',out/'calibration','--seeds','0',log='calibration.log')
    call('scripts/compact_diffusion_v1/proposals.py','--out',out/'proposals_recheck','--seeds','42,43,0',log='proposals_recheck.log')
    for seed in [42,43]:
        target=out/f'spread_target_{seed}.npz';feasible=out/f'feasible_{seed}.npz'
        projection=out/f'spread_projection_{seed}';execution=out/f'execution_{seed}';replay=out/f'replay_{seed}'
        call('scripts/compact_diffusion_v1/refine.py','propose','--sample',out/f'proposals/sample_{seed}.npz',
             '--out',target,'--gain','.15','--sample-gain','4','--retime-max','.04',
             '--timing-calibration',out/'calibration/sample_0.npz',log=f'spread_target_{seed}.log')
    refs=[np.load(out/f'spread_target_{seed}.npz') for seed in [42,43]]
    static=difference(dict(qpos=refs[0]['qref'],ctrl_physics=refs[0]['uref']),dict(qpos=refs[1]['qref'],ctrl_physics=refs[1]['uref']))
    (out/'kinematic_causality.json').write_text(json.dumps(static,indent=2))
    if not all(v['pass_meaningful_motion'] for v in static.values()):raise RuntimeError('Cheap reference causality gate failed')
    for seed in [42,43]:
        target=out/f'spread_target_{seed}.npz';feasible=out/f'feasible_{seed}.npz';projection=out/f'spread_projection_{seed}'
        call('scripts/compact_orcs_v1/run.py','--task',TASK,'--reference',target,'--out',projection,log=f'spread_projection_{seed}.log')
        call('scripts/compact_diffusion_v1/refine.py','realize','--run',projection,'--proposal',target,'--out',feasible,log=f'realize_{seed}.log')
    call('scripts/compact_diffusion_v1/evidence.py','causal','--runs',out/'spread_projection_42',out/'spread_projection_43',
         '--baseline',ROOT/'logs/compact_orcs_v1/complete_execution','--references',out/'spread_target_42.npz',out/'spread_target_43.npz',
         '--out',out/'projection_causality.json',log='projection_causality.log')
    if not json.loads((out/'projection_causality.json').read_text())['pass_causality']:raise RuntimeError('Refinement erased diffusion contribution')
    for seed in [42,43]:
        feasible=out/f'feasible_{seed}.npz';execution=out/f'execution_{seed}';replay=out/f'replay_{seed}'
        call('scripts/compact_orcs_v1/run.py','--task',TASK,'--reference',feasible,'--out',execution,log=f'execution_{seed}.log')
        call('scripts/compact_orcs_v1/run.py','--task',TASK,'--reference',feasible,'--out',replay,'--replay',execution,log=f'replay_{seed}.log')
        call('scripts/compact_orcs_v1/render.py',execution,log=f'video_{seed}.log')
        call('scripts/compact_orcs_v1/validate.py',execution,'--replay',replay,log=f'validation_{seed}.log')
        if json.loads((execution/'task_assessment.json').read_text())['status']!='SUCCESS':raise RuntimeError('Physical validation failed')
    call('scripts/compact_orcs_v1/run.py','--out',out/'baseline_recheck','--replay',ROOT/'logs/compact_orcs_v1/complete_execution',log='baseline_recheck.log')
    call('scripts/compact_diffusion_v1/evidence.py','causal','--runs',out/'execution_42',out/'execution_43',
         '--baseline',ROOT/'logs/compact_orcs_v1/complete_execution','--references',out/'feasible_42.npz',out/'feasible_43.npz',
         '--out',out/'causality.json',log='causality.log')
    call('scripts/compact_diffusion_v1/finalize.py',out,log='verification.log')
    print('Complete:',out/'verification.json',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();reproduce(a.out)
