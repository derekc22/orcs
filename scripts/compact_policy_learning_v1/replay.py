"""Run the unchanged accepted verifier with eager NPZ caching only.

Its historical lazy access decompresses a full state trajectory on every
physics step. This read-only proxy loads each NPZ array once. No verifier
physics, contact attribution, rendering, or measurement code is changed.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/corner_hands_v1'),str(ROOT/'scripts/articulated_gripper_v1'),str(ROOT/'scripts')]
from verify_physics import verify_physics


class EagerNumpy:
    def __getattr__(self,name):return getattr(np,name)
    def load(self,path,*args,**kwargs):
        archive=np.load(path,*args,**kwargs)
        if isinstance(archive,np.lib.npyio.NpzFile):
            result={key:archive[key] for key in archive.files};archive.close();return result
        return archive


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('artifact',type=Path);p.add_argument('--video',action='store_true');args=p.parse_args()
    verify_physics.__globals__['np']=EagerNumpy()
    verify_physics(args.artifact,args.video)
