"""Actual root/box paths, foot placements, sole clearance and force support plots."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(path):
 path=Path(path);a=np.load(path);report=json.loads(path.with_name(path.stem+'_sustained_metrics.json').read_text());l=np.load(path.with_name(path.stem+'_locomotion_trace.npz'))['trace'];t=np.load(path.with_name(path.stem+'_verification_trace.npz'))['trace'];s,e=map(int,report['metrics']['walking_interval_physics_steps']);sec=np.arange(len(l))*.005;feet=l[:,6:12].reshape(-1,2,3);fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True);ax=axes[0,0];ax.plot(l[s:e,0],l[s:e,1],color='seagreen',lw=3,label='Robot root');ax.plot(t[s:e,11],t[s:e,12],color='darkorange',label='Box COM')
 for side,color in enumerate(['royalblue','crimson']):ax.plot(feet[s:e,side,0],feet[s:e,side,1],color=color,alpha=.8,label=['Left foot','Right foot'][side])
 for n,event in enumerate(report['step_events'],1):
  xy=event['replant_xy'];ax.scatter(*xy,color='royalblue' if event['side']=='left' else 'crimson',s=45);ax.annotate(str(n),xy,xytext=(4,5),textcoords='offset points')
 ax.set_aspect('equal');ax.set_xlabel('World x (m)');ax.set_ylabel('World y (m)');ax.set_title('Actual walking paths and loaded replants');ax.legend();ax.grid(alpha=.25)
 ax=axes[0,1];k=np.clip(np.arange(len(l)),s,e-1);root=np.linalg.norm(l[k,:2]-l[s,:2],axis=1);box=np.linalg.norm(t[k,11:13]-t[s,11:13],axis=1);ax.plot(sec,root,color='seagreen',label='Root net from walking start');ax.plot(sec,box,color='darkorange',label='Object net from walking start');ax.axhline(2,color='black',ls='--',label='2 m root requirement');ax.axvspan(s*.005,e*.005,alpha=.15,color='seagreen');ax.set(xlabel='Time (s)',ylabel='Horizontal displacement (m)',title='Walking-only distance; held constant outside interval');ax.legend();ax.grid(alpha=.25)
 ax=axes[1,0]
 for side,color in enumerate(['royalblue','crimson']):ax.plot(sec,100*l[:,12+side],color=color,label=['Left minimum sole clearance','Right minimum sole clearance'][side])
 ax.axhline(2,color='gray',ls='--');ax.set(xlim=(s*.005-.2,e*.005+.2),ylim=(-1,10),xlabel='Time (s)',ylabel='Minimum collision clearance (cm)',title='Whole-foot clearance during actual swing');ax.legend();ax.grid(alpha=.25)
 ax=axes[1,1];states=np.array([t[:,0]>0,l[:,14]>=10,l[:,15]>=10]);labels=['Object terrain contact','Left foot loaded support','Right foot loaded support']
 for i,(state,label) in enumerate(zip(states,labels)):
  ax.fill_between(sec,i,i+.7,where=state,step='post',color=['darkorange','royalblue','crimson'][i],alpha=.8)
 ax.axvspan(s*.005,e*.005,alpha=.12,color='seagreen');ax.set_yticks(np.arange(3)+.35,labels);ax.set(xlabel='Time (s)',title='Measured contact and force support states');ax.grid(axis='x',alpha=.25)
 fig.suptitle('Original largebox: actual sustained walking with physical grasp',fontsize=16);fig.savefig(path.with_name(path.stem+'_foot_trajectories.png'),dpi=160);plt.close(fig)
 print('PLOT',path.with_name(path.stem+'_foot_trajectories.png'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('artifact',type=Path);plot(p.parse_args().artifact)
