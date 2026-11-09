# Goal 7: full-source pickup and bounded carry/placement diagnosis

The strongest new candidate replaces the pickup interior with event-aligned
curves from the existing frozen August HF-BPS samples 42 and 43. It retains the
verified incremental Goal 7 candidate for the other phases. This is substantial
pickup authority, **not demonstrated 8/10 diffusion dominance across all phases**.
The original incremental implementation, assessment, references and videos are
preserved. No tracked baseline, model, gripper, learned controller or weight is
changed. No commit or push is made.

`primary_targets.py` aligns complete native proposals to shared contact events
and task endpoints, retaining the previously accepted sample-dependent motions.
`primary_kinematics.py` constrains the accepted sole poses and bilateral grasp,
with collision clearance, native workspace bounds, and a bounded COM region.
`primary_dynamics.py` includes loaded-foot Jacobian torques and nonnegative
generators of the actual pyramidal friction cone. `primary_compile.py` preserves
physics-rate control details through differential inverse dynamics. All are
offline necessary screens; genuine ORCS execution is the feasibility criterion.

Larger carry/placement references pass some geometry screens but fail dynamic
or workspace checks. The original dense projection introduced object angular
acceleration above 300 rad/s² and a predicted waist-yaw effort excess of 160 Nm.
Soft continuity introduced endpoint mismatch; holding smoothed poses/angles
fixed did not resolve the combination of workspace and dynamic constraints.
Rejected references were never executed. `primary_isolate.py` records the
screened pickup extraction used for the first seed-42 physical test.

Evidence is in `logs/compact_diffusion_pose_v1/primary/consistent/`: `pickup_assessment.json`,
`pickup_source_attribution.json`, `pickup_execution_{42,43}/task_assessment.json`,
`pickup_replay_{42,43}/execution.json`, and the full-task `rollout.mp4` files in
each pickup execution directory. Large arrays and videos remain local.
Both final references use the same float64 compiler path. The initial seed-42
phase extraction had float32 storage; its successful execution is preserved as
earlier evidence and is excluded from the final comparison. Seed-43 evidence is
reused through relative links, with exact reference-array equality checked.

From `/home/learning/Documents/orcs-fork`, reproduce the strongest candidate:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -B scripts/compact_diffusion_pose_v1/primary_reproduce.py --out logs/compact_diffusion_pose_v1_pickup_reproduction
```

This uses the existing verified incremental references under
`logs/compact_diffusion_pose_v1/`, Goal 6 and disabled physical evidence, frozen
proposals/calibration, accepted controller and model dependencies, and read-only
August checkpoint. It neither reruns the rejected variants nor trains anything.
The comparison separates native proposals, endpoint retargeting, constrained
references, and actual states. It does not interpret motion RMS ratios as causal
percentages or a calibrated authority score.

Feet, support/contact ordering and gripper mechanics remain expert controlled.
Carry and placement retain the accepted expert task-scale backbone and frozen
feedback stabilization. No claim is made that 8/10 is fundamentally impossible
under a different verified strategy.
