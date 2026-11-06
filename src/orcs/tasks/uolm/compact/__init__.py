"""Registered full-task compact-hand UOLM integration."""
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import SimulationCfg
from mjlab.rl import RslRlModelCfg
from orcs.core.rl import runner
from orcs.tasks.uolm.mdp.rewards import object_pos_tracking_reward, object_ori_tracking_reward
from .model import AcceptedPhysicsCfg, robot_cfg, object_cfg, add_terrain
from .runtime import (CompactRealizedCommandCfg, CompactResidualActionCfg,
                      policy_observation, tracking_reward,
                      invalid_state, complete)
from .measurements import record_physics, invalid_contacts

TASK_ID = "Orcs-Uolm-Compact-Realized-Feedback"


def env_cfg(play=False):
    group = lambda: ObservationGroupCfg(terms={"state_reference": ObservationTermCfg(func=policy_observation)},
                                       concatenate_terms=True, enable_corruption=False)
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(entities={"robot": robot_cfg(), "object": object_cfg()}, spec_fn=add_terrain),
        sim=SimulationCfg(mujoco=AcceptedPhysicsCfg(timestep=.005), nconmax=256, njmax=1000),
        decimation=1, episode_length_s=48.145, auto_reset=False, seed=42,
        commands={"motion": CompactRealizedCommandCfg(motion_file="", resampling_time_range=(1e9, 1e9), debug_vis=False)},
        actions={"joint_pos": CompactResidualActionCfg(entity_name="robot", actuator_names=(".*",),
                                                       scale=1., use_default_offset=False)},
        observations={"policy": group(), "critic": group()},
        rewards={"robot_tracking": RewardTermCfg(func=tracking_reward, weight=1.),
                 "object_tracking": RewardTermCfg(func=object_pos_tracking_reward, weight=1., params={"command_name": "motion", "std": .1}),
                 "object_orientation": RewardTermCfg(func=object_ori_tracking_reward, weight=1., params={"command_name": "motion", "std": .4})},
        metrics={"physical_contact_audit": MetricsTermCfg(func=record_physics, per_substep=True)},
        terminations={"invalid_state": TerminationTermCfg(func=invalid_state),
                      "invalid_contacts": TerminationTermCfg(func=invalid_contacts),
                      "complete": TerminationTermCfg(func=complete, time_out=True)},
    )


def agent_cfg():
    cfg = runner("orcs_uolm_compact_feedback")
    cfg.actor = RslRlModelCfg(class_name="orcs.tasks.uolm.compact.runtime:CompactFeedbackActor",
                             hidden_dims=(16,), obs_normalization=False,
                             distribution_cfg={"class_name": "GaussianDistribution", "init_std": .001, "std_type": "scalar", "learn_std": False})
    cfg.critic.hidden_dims = (16,)
    cfg.critic.obs_normalization = False
    cfg.clip_actions = None
    cfg.logger = "tensorboard"
    cfg.upload_model = False
    return cfg
