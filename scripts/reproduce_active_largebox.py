#!/usr/bin/env python3
"""Replay a frozen active-grasp expert into a fresh directory, preserving evidence."""
import argparse,datetime,json,subprocess,sys
from pathlib import Path
from active_largebox_artifacts import materialize
ROOT=Path(__file__).resolve().parents[1]
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('artifact',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--video',action='store_true');args=parser.parse_args();artifact=args.artifact.resolve();model_name=artifact.stem+'_model.mjb';model=materialize(artifact.with_name(model_name))
 output=args.output or ROOT/'logs/active_largebox/reproductions'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');output=output.resolve();output.mkdir(parents=True,exist_ok=False)
 (output/artifact.name).symlink_to(artifact)
 (output/model_name).symlink_to(model)
 target=output/artifact.name
 command=[sys.executable,str(ROOT/'scripts/verify_active_largebox.py'),str(target)]
 if args.video:command.append('--video')
 subprocess.run(command,cwd=ROOT,check=True);subprocess.run([sys.executable,str(ROOT/'scripts/audit_active_largebox.py'),str(target)],cwd=ROOT,check=True)
 result=json.loads(target.with_name(target.stem+'_acceptance_audit.json').read_text());print('REPRODUCTION_DIRECTORY',output);print('ALL_MACHINE_GATES_PASS',result['all_machine_gates_pass'])
 return 0 if result['all_machine_gates_pass'] else 1
if __name__=='__main__':raise SystemExit(main())
