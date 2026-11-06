"""Full-task realized UOLM reference, native residual action and frozen actor."""
from dataclasses import dataclass
from functools import lru_cache

import mujoco
import numpy as np
import torch
from mjlab.managers.command_manager import CommandTerm
from orcs.tasks.uolm.mdp.commands import RealizedObjectMotionCommand, RealizedObjectMotionCommandCfg
from orcs.tasks.uolm.mdp.actions import RealizedResidualJointPositionAction, RealizedResidualJointPositionActionCfg
from rsl_rl.models import MLPModel

from .model import CHECKPOINT, source_model


@lru_cache(maxsize=1)
def reference():
    with np.load(CHECKPOINT) as a:
        return {k: a[k].copy() for k in a.files}


class CompactRealizedCommand(RealizedObjectMotionCommand):
    """Generalized realized command: physical pose conventions, named 37-DOF I/O.

    Existing RealizedObjectMotionCommand is restricted to the old 6 s flat-hand
    artifact. This subclass retains its manager lifecycle while using the full
    accepted reference. State assignments occur only in episode reset.
    """
    def __init__(self, cfg, env):
        CommandTerm.__init__(self, cfg, env)
        self.robot = env.scene['robot']
        self.object = env.scene['object']
        r = reference()
        self.horizon = len(r['uref'])
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_state_writes = 0
        self.qref = torch.as_tensor(r['qref'], dtype=torch.float64, device=self.device)
        self.vref = torch.as_tensor(r['vref'], dtype=torch.float64, device=self.device)
        self.uref = torch.as_tensor(r['uref'], dtype=torch.float64, device=self.device)
        m = source_model()
        names = [m.joint(int(j)).name for j in m.actuator_trnid[:, 0]]
        self.action_to_source = np.array([names.index(n) for n in self.robot.joint_names])
        self.joint_qpos = np.array([m.jnt_qposadr[m.joint(n).id] for n in self.robot.joint_names])
        self.joint_qvel = np.array([m.jnt_dofadr[m.joint(n).id] for n in self.robot.joint_names])
        self.metrics['reference_phase'] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self):
        return torch.cat((self.qref[self.time_steps], self.vref[self.time_steps]), dim=-1)

    @property
    def joint_pos(self):
        return self.qref[self.time_steps][:, self.joint_qpos]

    @property
    def joint_vel(self):
        return self.vref[self.time_steps][:, self.joint_qvel]

    def future_frames(self, steps, skip=1):
        return (self.time_steps[:, None] + torch.arange(steps, device=self.device)*skip).clamp(max=self.horizon-1)

    @property
    def motion_joint_pos_future(self):
        return self.qref[self.future_frames(self.cfg.future_steps)][..., self.joint_qpos]

    @property
    def motion_joint_vel_future(self):
        return self.vref[self.future_frames(self.cfg.future_steps)][..., self.joint_qvel]

    @property
    def realized_control_target(self):
        return self.uref[self.time_steps][:, self.action_to_source]

    @property
    def object_pos_w(self):
        return self.qref[self.time_steps, 44:47] + self._env.scene.env_origins

    @property
    def object_quat_w(self):
        return self.qref[self.time_steps, 47:51]

    @property
    def object_goal_pos_env(self):
        return self.qref[-1, 44:47].expand(self.num_envs, -1)

    @property
    def object_goal_quat(self):
        return self.qref[-1, 47:51].expand(self.num_envs, -1)

    def _resample_command(self, env_ids):
        self.time_steps[env_ids] = 0
        # Exact full model ordering is established by the model gate. Both
        # physical entities are initialized once through the command lifecycle.
        self._env.sim.data.qpos[env_ids] = self.qref[0].to(self._env.sim.data.qpos.dtype)
        self._env.sim.data.qvel[env_ids] = self.vref[0].to(self._env.sim.data.qvel.dtype)
        self.reset_state_writes += len(env_ids)
        if hasattr(self._env, 'compact_measurements'):
            del self._env.compact_measurements

    def _update_command(self, env_ids=None):
        self.time_steps[:] = self._env.episode_length_buf.clamp(max=self.horizon-1)

    def _update_metrics(self):
        self.metrics['reference_phase'][:] = self.time_steps / (self.horizon-1)


@dataclass(kw_only=True)
class CompactRealizedCommandCfg(RealizedObjectMotionCommandCfg):
    def build(self, env):
        return CompactRealizedCommand(self, env)


class CompactResidualAction(RealizedResidualJointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        cmd = env.command_manager.get_term(cfg.command_name)
        r = reference()
        self.bound = torch.as_tensor(r['action_bound'][cmd.action_to_source], device=self.device)
        self.ctrlrange = torch.as_tensor(source_model().actuator_ctrlrange[cmd.action_to_source], device=self.device)

    def apply_actions(self):
        cmd = self._env.command_manager.get_term(self.cfg.command_name)
        delta = torch.clamp(self._processed_actions - self._offset, -self.bound, self.bound)
        target = torch.clamp(cmd.realized_control_target + delta, self.ctrlrange[:, 0], self.ctrlrange[:, 1])
        self._entity.set_joint_position_target(target.to(self._entity.data.joint_pos_target.dtype), joint_ids=self._target_ids)


@dataclass(kw_only=True)
class CompactResidualActionCfg(RealizedResidualJointPositionActionCfg):
    def build(self, env):
        return CompactResidualAction(self, env)


def policy_observation(env):
    cmd = env.command_manager.get_term('motion')
    return torch.cat((cmd.time_steps[:, None], env.sim.data.qpos[:],
                      env.sim.data.qvel[:], cmd.command), dim=-1).double()


class CompactFeedbackActor(MLPModel):
    """The accepted learned K policy, evaluated through the native runner.

    Inputs come exclusively from the observation manager, including the motion
    command's current reference. The optional neural residual starts at zero;
    no training or replacement of the accepted feedback is needed.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for p in self.mlp.parameters():
            p.data.zero_()
        r = reference()
        self.K = r['K']
        self.stride = int(r['stride'])
        self.bound = r['action_bound']
        m = source_model()
        self.order = np.argsort(m.jnt_qposadr[m.actuator_trnid[:, 0]])
        self.delta = {}
        self.last_step = {}

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        data = obs[self.obs_groups[0]].detach().cpu().numpy()
        result = []
        m = source_model()
        for i, row in enumerate(data):
            step = int(row[0]); q = row[1:52]; v = row[52:101]
            qr = row[101:152]; vr = row[152:201]
            if step % self.stride == 0 or i not in self.delta or step <= self.last_step.get(i, -1):
                dq = np.empty(m.nv)
                mujoco.mj_differentiatePos(m, dq, 1., qr, q)
                self.delta[i] = np.clip(-self.K[step//self.stride] @ np.r_[dq, v-vr], -self.bound, self.bound)
            self.last_step[i] = step
            result.append(self.delta[i][self.order])
        # Keep the native model/distribution residual interface available.
        residual = super().forward(obs.to(dtype=torch.float32), masks, hidden_state, stochastic_output)
        return torch.as_tensor(np.asarray(result), device=residual.device, dtype=torch.float64) + residual


def tracking_reward(env):
    cmd = env.command_manager.get_term('motion')
    target = cmd.qref[(cmd.time_steps+1).clamp(max=cmd.horizon-1)]
    return torch.exp(-50 * (env.sim.data.qpos-target).square().mean(dim=-1)).float()


def invalid_state(env):
    return (~torch.isfinite(env.sim.data.qpos).all(-1)) | (env.sim.data.qpos[:, 2] < .32)


def complete(env):
    return env.episode_length_buf >= env.command_manager.get_term('motion').horizon
