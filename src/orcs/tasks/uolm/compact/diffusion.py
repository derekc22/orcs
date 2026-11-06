"""Frozen-controller execution of externally refined diffusion references."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from . import env_cfg as baseline_env_cfg, agent_cfg
from .runtime import CompactRealizedCommand, CompactRealizedCommandCfg, reference

TASK_ID='Orcs-Uolm-Compact-Diffusion-Feedback'


class DiffusionRefinedCommand(CompactRealizedCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        path=Path(cfg.refined_reference_path)
        if not path.is_file():
            raise FileNotFoundError(f'Refined diffusion reference required: {path}')
        with np.load(path) as a:
            r={k:a[k].copy() for k in a.files}
        b=reference()
        for k in ['qref','vref','uref']:
            if r[k].shape!=b[k].shape or not np.isfinite(r[k]).all():
                raise ValueError(f'Invalid compact reference {k}: {r[k].shape}')
        if not np.array_equal(r['qref'][0].astype(np.float32),b['qref'][0].astype(np.float32)) or not np.array_equal(r['vref'][0],b['vref'][0]):
            raise ValueError('Diffusion must preserve the accepted physical initial state')
        for k in ['qref','vref','uref']:
            setattr(self,k,torch.as_tensor(r[k],device=self.device,dtype=torch.float64))
        self.fixed_goal_position=torch.as_tensor(r['goal_position'],device=self.device,dtype=torch.float64)
        self.fixed_goal_quaternion=torch.as_tensor(r['goal_quaternion'],device=self.device,dtype=torch.float64)
        if not np.array_equal(r['goal_position'],b['qref'][-1,44:47]) or not np.array_equal(r['goal_quaternion'],b['qref'][-1,47:51]):
            raise ValueError('Sample comparison must preserve the final task goal')

    @property
    def object_goal_pos_env(self):return self.fixed_goal_position.expand(self.num_envs,-1)

    @property
    def object_goal_quat(self):return self.fixed_goal_quaternion.expand(self.num_envs,-1)


@dataclass(kw_only=True)
class DiffusionRefinedCommandCfg(CompactRealizedCommandCfg):
    refined_reference_path:str=''
    def build(self,env):return DiffusionRefinedCommand(self,env)


def env_cfg(play=False):
    cfg=baseline_env_cfg(play)
    cfg.commands['motion']=DiffusionRefinedCommandCfg(motion_file='',resampling_time_range=(1e9,1e9),debug_vis=False)
    return cfg
