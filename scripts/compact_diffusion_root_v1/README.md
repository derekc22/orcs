# Goal 6: supported diffusion root XY

This extends the successful Goal 5 references from commit
`1aab1ee2a1c108630fa67ca3cccdf4a8523b34ba`. Goal 5 and all accepted
physics, controller, gripper, footstep and validation code stay unchanged.

The August HF-BPS checkpoint is frozen. Cached seeds 42 and 43, minus the same
seed-0 calibration, supply root XY innovations. The alignment comes from the
conditioning demonstration and fixed task goal, never from executed motion.
Independent fresh inference verifies the cached samples exactly.

`supported_refine.py` keeps the Goal 5 pelvis height/orientation, waist, object
and gripper trajectories. Bounded arm/leg IK preserves both soles and hands.
Only additive motion is smoothed; baseline 200 Hz contact details are retained.
Differential inverse dynamics supplies feedforward for the unchanged controller.
No simulator is stepped by the refiner.

XY motion is zero in flight and contact guards. Supported bouts must contain
both 300 ms quintic ramps and 50 ms guards (0.70 s minimum). A shorter carry
bout had insufficient contact-wrench reserve in both baselines and is excluded
by this common duration rule. The original model's pyramidal friction cone,
joint limits, contact geometry, and actuator limits remain unchanged.
Root roll/pitch remain Goal 5 references: preliminary counterlean candidates
failed kinematic/dynamic screening. They were never physically rolled out.
`root_refine.py` retains these diagnostic prototypes and the shared source,
support-clock and wrench-screen helpers; the accepted pipeline uses
`supported_refine.py` exclusively.

The actual ORCS UOLM manager, native runner, frozen feedback actor and simulator
establish feasibility. Successful projections are realized with the existing
Goal 2 helper, executed afresh, replayed independently, rendered completely,
and checked by the unchanged 22-gate validator. Realization stores actual
states and controls; it does not alter weights or attach the box virtually.

Run from `/home/learning/Documents/orcs-fork`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -B scripts/compact_diffusion_root_v1/reproduce.py --out logs/compact_diffusion_root_v1_reproduction
```

This requires the existing successful Goal 5 references and executions in
`logs/compact_diffusion_body_v1/`, original proposal/calibration and conditioning
files in `logs/compact_diffusion_v1/`, the existing frozen policy and original
model assets, the Goal 4 executions for inherited-body causal ablations, and
the read-only diffusion repository/checkpoint. These local
dependencies are reused rather than packaged or copied. Large arrays, model
snapshots and complete videos remain at their recorded local paths.

Evidence is written to the selected output directory: `goal6_assessment.json`,
`source_attribution.json`, `goal6_manifest.json`, phase plots, and each sample's
`execution_*/task_assessment.json`, `replay_*/execution.json`, `rollout.mp4` and
`video_metadata.json`. The report separates raw-source to constrained-reference
tracking from constrained-reference to physical execution, includes same-seed
Goal 5 ablations, and reports all components by fixed task and active phases.
Post-step execution is compared to the next reference tick without fitted
alignment. Any failed experiment stays separate from the accepted candidate.

Inherited pelvis yaw uses Goal 5's original world-Z rotation coordinate.
Euler yaw is reported separately because it couples to the added balance tilt
in the crouched pose. Pickup Euler-yaw sample difference falls by 37%; the
original world-Z rotation-component difference retains 85.5%, with both
samples still closely following the frozen yaw target. Root XY authority
does not imply independent diffusion control of roll/pitch.
