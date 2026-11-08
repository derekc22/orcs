# Physically executable diffusion authority (Goal 4)

Run from `orcs-fork`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -B scripts/compact_diffusion_authority_v1/reproduce.py --out logs/compact_diffusion_authority_v1_reproduction
```

The output directory must be new. `--resume` continues an existing experiment.
The retained result is in `logs/compact_diffusion_authority_v1/goal4_assessment.json`.
The two final videos are `execution_42/rollout.mp4` and `execution_43/rollout.mp4`.
`executed_authority.png` and `.pdf` compare actual motion against the disabled
baseline and between the two samples. The JSON includes both the original
Goal 2 comparison windows and the smaller intervals where references change.

## Motion boundary

The successful controller's original 200 Hz reference is the base. A disabled
identity branch returns its arrays exactly. The frozen August HF-BPS model's
samples 42 and 43 are compared with the fixed calibration sample 0. The same
condition, coordinate transform, gains, limits, and refinement are used for both
samples. Fresh inference must reproduce the preserved samples exactly.

The editable variables are fourteen arm joints and the object's six-dimensional
pose. A bounded nonlinear IK projection preserves both hand/object transforms.
After smoothing only the additive innovation, a second projection at every
physics tick restores rigid grasp geometry and bounds added arm velocity.
Joint bounds preserve the baseline's existing soft-limit excursions without
enlarging them in the reference. Servo targets include the position and damping
offset plus an inverse-dynamics torque difference for the arms.

Root, leg, waist and gripper references, non-arm nominal feedforward, support
timing, footsteps, contact order, physical model, and learned feedback remain
unchanged. No reference is downsampled or reconstructed from a low-rate full-body
trajectory. Each phase returns to the expert reference with smooth transitions.

## Physical evidence

`projection_42/` and `projection_43/` contain genuine managed ORCS rollouts of the
retargeted references. Only a complete, physically successful projection can be
compiled into `feasible_42.npz` or `feasible_43.npz`. These contain measured states
and actual servo commands as a new reference, following the accepted Goal 2
refinement method. The states are never assigned during execution beyond reset.

`execution_42/` and `execution_43/` execute those references again through the
registered `Orcs-Uolm-Compact-Diffusion-Feedback` task and frozen actor.
`replay_42/` and `replay_43/` independently integrate the recorded actions through
the same genuine ORCS physics. The unchanged validator checks all 22 gates,
including exact state/control/contact replay and full-task video coverage.

`report.py` fails the overall authority criterion unless actual sample-dependent
motion is substantial in at least two phases and measurable in all three.
It reports root translation/orientation, fourteen arm joints, object
translation/orientation, and all actuator commands separately for each phase.
Its numeric authority criterion is an operational test, not a calibrated
one-to-ten control-authority scale.

## Files

- `refine.py`: frozen-sample retargeting and bounded contact refinement.
- `reproduce.py`: inference, physical executions, replay, videos, validation.
- `report.py`: causal comparisons, physical metrics, preservation hashes.
- `plot.py`: standalone figures of executed motion differences.

The successful Goal 1/2 sources and artifacts and Goal 3 diagnostics are retained.
This experiment does not train a model/controller, change the gripper or physical
parameters, commit, push, or access the separate original `orcs` repository.
