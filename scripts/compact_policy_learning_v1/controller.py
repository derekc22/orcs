"""Reference-conditioned controller learned from local physical transition data.

Only actuator commands are written during inference. The expert feedforward,
clock and native position servos remain engineered; K is optimized from fitted
robot AND object dynamics, rather than assigning synthetic PD action labels.
"""
from pathlib import Path
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'logs/compact_place_refinement_v1/complete_execution/rollout.npz'
MODEL = SOURCE.with_name('rollout_model.mjb')
OUT = ROOT / 'logs/compact_policy_learning_v1'


def error(m, qref, vref, q, v):
    dq = np.empty(m.nv)
    mujoco.mj_differentiatePos(m, dq, 1., qref, q)
    return np.r_[dq, v-vref]


class Policy:
    def __init__(self, m, checkpoint):
        c = np.load(checkpoint)
        self.m = m
        self.K = c['K']
        self.stride = int(c['stride'])
        self.qref = c['qref']
        self.vref = c['vref']
        self.uref = c['uref']
        self.bound = c['action_bound']
        self.delta = np.zeros(m.nu)
        self.last_error = np.zeros(2*m.nv)

    def control(self, step, d, feedback=True):
        if step % self.stride == 0:
            self.last_error = error(self.m, self.qref[step], self.vref[step], d.qpos, d.qvel)
            self.delta = np.clip(-self.K[step//self.stride] @ self.last_error,
                                 -self.bound, self.bound) if feedback else np.zeros(self.m.nu)
        return np.clip(self.uref[step] + self.delta,
                       self.m.actuator_ctrlrange[:, 0], self.m.actuator_ctrlrange[:, 1])
