"""Separated compact common-frame model; baseline builder sources are read only."""
from pathlib import Path
import importlib.util
import numpy as np,mujoco
from compact_gripper_gate import OUT,BASE

def build(shorten=.30,tag='common'):
 source=(Path(__file__).parent/'active_largebox_clamp.py').read_text().split('\ndef run(tag):')[0]
 source=source.replace("config=[];points=[]", "config=[];points=[];delta=np.array([.937078432588943,-.0168575976376823,.3487116754261833])*SHORTEN")
 source=source.replace('local=W.T@(world-origin)', 'local=W.T@(world-origin)-delta').replace('hub=W.T@(top-origin)','hub=W.T@(top-origin)-delta')
 source=source.replace('bypass=W.T@(outward*.16+np.array([0,0,.04]))','bypass=.4*hub+W.T@(outward*.10)')
 source=source.replace('up=W.T@(upper-origin);lo=W.T@(p-origin)','up=W.T@(upper-origin)-delta;lo=W.T@(p-origin)-delta')
 ns={'__name__':'compact_common_builder','SHORTEN':shorten};exec(compile(source,'compact_common_builder','exec'),ns);ns['OUT']=OUT
 m,q,v=ns['build'](tag)
 # Return gate poses on the exact newly compiled model: matching native/finger DOFs.
 return m,q,v
if __name__=='__main__':build()
