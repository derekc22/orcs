"""Unfiltered physical walking traces and explicit phase-level regressions."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parents[2]/'logs/compact_smooth_integration_v2';tr=[np.load(OUT/p/'motion_trace.npz') for p in ['baseline','complete_execution']];r=json.loads((OUT/'comparison_metrics.json').read_text());s,e=3449,4721;t=np.arange(s,e)*.005;colors=['#b96635','#167e96'];labels=['Accepted baseline','Complete smoothed execution']
fig,ax=plt.subplots(3,1,figsize=(13,9),sharex=True)
for a,color,label in zip(tr,colors,labels):
 for x,key in zip(ax,['object_omega','object_alpha','relative_omega']):x.plot(t,np.linalg.norm(a[key][s:e],axis=1),label=label,color=color,lw=1,alpha=.85)
for x,title in zip(ax,['Object angular speed [rad/s]','Object angular acceleration [rad/s²]','Combined box/palm angular rate [rad/s]']):x.set_ylabel(title);x.grid(alpha=.2)
ax[0].legend();ax[-1].set_xlabel('Physical time [s]');fig.suptitle('Walking improvement preserved exactly in the complete execution');fig.tight_layout();fig.savefig(OUT/'walking_motion_comparison.png',dpi=150)
items=[('Walking\nobject acceleration','walking','object_alpha'),('Walking\npalm-relative rate','walking','relative_omega'),('Placement\nobject acceleration','placement','object_alpha'),('Full run\nobject speed','complete_execution','object_omega'),('Full run\nobject acceleration','complete_execution','object_alpha')];values=[r['phase_comparison'][p][k]['rms']['after']/r['phase_comparison'][p][k]['rms']['before'] for _,p,k in items]
fig,ax=plt.subplots(figsize=(12,4));bars=ax.bar(range(len(items)),values,color=['#167e96' if x<1 else '#b96635' for x in values]);ax.axhline(1,color='gray',ls='--');ax.set_xticks(range(len(items)),[a[0] for a in items]);ax.set_ylabel('RMS / accepted baseline');ax.set_ylim(0,1.45)
for bar,x in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,x+.03,f'{(x-1)*100:+.1f}%',ha='center')
ax.set_title('Placement object vibration regresses; carry and full-run measures improve');fig.tight_layout();fig.savefig(OUT/'smoothness_tradeoff.png',dpi=150)
