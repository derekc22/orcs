"""Reference-centred actions for realized physical UOLM trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

__all__ = [
    "RealizedResidualJointPositionAction",
    "RealizedResidualJointPositionActionCfg",
]


@dataclass(kw_only=True)
class RealizedResidualJointPositionActionCfg(JointPositionActionCfg):
    """Policy residual around the actuator targets saved by physical rollout."""

    command_name: str = "motion"
    residual_bound: float = 0.15

    def build(self, env):
        return RealizedResidualJointPositionAction(self, env)


class RealizedResidualJointPositionAction(JointPositionAction):
    """Apply bounded policy residuals around a realized-reference control.

    A zero policy action exactly requests the actuator target recorded by the
    dynamics-aware generator.  The policy therefore learns contact-sensitive
    feedback corrections instead of reconstructing the entire feed-forward
    signal.  The final command still passes through the compiled position
    actuators and their nominal limits.
    """

    cfg: RealizedResidualJointPositionActionCfg

    def __init__(self, cfg: RealizedResidualJointPositionActionCfg, env):
        super().__init__(cfg, env)
        self._env = env

    def apply_actions(self) -> None:
        command = self._env.command_manager.get_term(self.cfg.command_name)
        if not hasattr(command, "realized_control_target"):
            raise TypeError(
                "RealizedResidualJointPositionAction requires a realized reference"
            )
        # JointPositionAction's processed value contains the default-pose
        # offset.  Only its scaled policy component is the residual here.
        residual = self._processed_actions - self._offset
        residual = torch.clamp(
            residual, min=-self.cfg.residual_bound, max=self.cfg.residual_bound
        )
        target = command.realized_control_target + residual
        self._entity.set_joint_position_target(target, joint_ids=self._target_ids)

