#!/usr/bin/env python3
"""End a continuous recorded expert at its first complete, quiet goal state.
No controls or physics states are changed; the full source and aftermath remain.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np

def terminate(source,target):
 a=np.load(source);trace=np.load(source.with_name(source.stem+'_verification_trace.npz'))['trace'];audit=json.loads(source.with_name(source.stem+'_acceptance_audit.json').read_text())
 failures={k for k,v in audit['gates'].items() if not v}
 if failures-{'stable_robot'}:raise RuntimeError(f'Source has unresolved task failures: {failures}')
 end_withdraw=int(float(a['phase_withdrawal_end_frame']));frame=None
 for end in range(end_withdraw+50,len(trace)//4+1):
  t=trace[(end-50)*4:end*4]
  quiet=(np.percentile(np.linalg.norm(t[:,36:39],axis=1),95)<.15 and np.percentile(np.linalg.norm(t[:,39:42],axis=1),95)<.25 and np.linalg.norm(t[:,20:23],axis=1).max()<.01 and np.linalg.norm(t[:,23:26],axis=1).max()<.03 and not t[:,43].any() and t[:,0].all() and (t[:,26:32]<-.025).all())
  if quiet:frame=end;break
 if frame is None:raise RuntimeError('No complete quiet final window in this rollout')
 if target.exists():raise FileExistsError(target)
 count=frame*4;arrays={k:(a[k][:count] if a[k].ndim>0 and a[k].shape[0]==len(trace) else a[k].copy()) for k in a.files};arrays['source_full_rollout']=np.asarray(str(source));arrays['goal_termination_frame']=np.asarray(frame);np.savez_compressed(target,**arrays);shutil.copy2(source.with_name(source.stem+'_model.mjb'),target.with_name(target.stem+'_model.mjb'))
 target.with_name(target.stem+'_termination.json').write_text(json.dumps({'source_full_rollout':str(source),'original_physics_steps':len(trace),'accepted_physics_steps':count,'termination_task_frame':frame,'termination_time_s':count*.005,'method':'first one-second quiet goal window after complete physical withdrawal','controls_or_states_modified':False,'source_aftermath_preserved':True},indent=2));print('GOAL_TERMINATION',frame,count*.005,target)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('target',type=Path);a=p.parse_args();terminate(a.source,a.target)
