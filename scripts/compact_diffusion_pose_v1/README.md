# Goal 7 investigation: supported root tilt and waist coordination

The requested 8/10 diffusion-primary objective is not established by this
implementation. The implemented candidate is an incremental, source-derived
root-tilt/waist extension of the verified Goal 6 reference backbone. Its physical
task validity and quality are assessed separately from the authority objective.
See `logs/compact_diffusion_pose_v1/goal7_assessment.json` for final evidence.

No accepted source, controller, model, gripper, or diffusion weight is changed.
The baseline is commit `6e09e89fc547dc547582ece30a82c7e620c90cc7`.

## Method and bounded diagnosis

`refine.py` uses the original frozen August HF-BPS samples 42 and 43 and the
same seed-0 calibration. It extracts world-horizontal root relative rotation
and native waist roll/pitch innovations, with the existing deterministic world
alignment and common support clock. The source and checkpoint hashes are
recorded, and Goal 6's independent exact frozen-inference evidence is retained.
No new noise, fitted execution alignment, or learned parameters are introduced.

The constrained variables are two root-rotation coordinates and 28 dependent
native joints. Goal 6 root XYZ, waist yaw, object, gripper references and
foot/hand transforms are retained. Whole-system horizontal COM is preserved
locally; only additions are smoothed. Bounded projection and differential
inverse dynamics produce inputs to the unchanged frozen controller. FK and
RNE are offline screens; no standalone simulator is stepped.

Initial projection fit contact geometry but introduced root acceleration
corners. A root acceleration bound of 1.5 rad/s² fixed these. Sample 43 still
failed the exact pyramidal-contact wrench screen at tick 3920 (19.6 s), with
about 1.86 Nm added pitch moment. Its linearized feasible increment was 0.312.
Strict acceleration boxes on every native joint conflicted with closing joint
workspace constraints and were rejected before physical execution.

The final common source scale is 0.25 for both samples, below that diagnosed
wrench reserve. This is not a claim of maximal feasible authority. No larger
body-motion campaign is pursued, and no failed reference is called physical
success. Flight, contact guards, and short support bouts have zero additions.

The physical route is unchanged: registered ORCS UOLM managers and native
runner, successful physical projection, Goal 2 realization, fresh execution,
independent action replay, full video, and the unchanged 22-gate validator.

## Reproduction

From `/home/learning/Documents/orcs-fork`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -B scripts/compact_diffusion_pose_v1/reproduce.py --out logs/compact_diffusion_pose_v1_reproduction
```

This reproduces the bounded candidate and its assessment; it does not repeat
the rejected compiler variants or perform training. It reuses the existing
Goal 6 references/executions/disabled baseline, original frozen proposals and
calibration, accepted policy/model assets, earlier tracked verification tools,
and read-only checkpoint in `/home/learning/Documents/g1-diffusion`. These
existing local dependencies are not packaged or replaced.

`report.py` compares both samples, the diffusion-disabled expert, and same-seed
Goal 6 ablations by fixed phase windows. Raw innovations, mapped modes,
requested modes, projected references and executed changes remain distinct.
Motion-curvature RMS ratios describe counterfactual intervention magnitude;
they are not causal percentages or a calibrated authority rating.

`quality.py` measures actual walking torso oscillation, box angular velocity
and acceleration, loaded-foot surface-point slip, grasp forces and placement
smoothness against Goal 6. Video storyboards provide additional visual review.
Soft position-stop penetrations already present in the accepted model are
distinct from hard actuator command/force limits; physical parameters remain
unchanged. Large arrays, models and complete videos stay at local evidence
paths, with provenance recorded in `goal7_manifest.json`.

Remaining expert responsibilities include footstep/support/contact timing,
gripper mechanics, necessary stabilization, and the accepted task-scale motion
backbone. Adding a bounded tilt coordinate cannot by itself justify 8/10.
In particular, carrying waist roll/pitch is redirected into balance
compensation: its source-to-refined correlation is negative for both samples,
even though the robot closely executes that refined compensation. Root tilt
does follow the requested frozen-sample shape. This is not independent primary
diffusion control over every permitted body coordinate, and it does not prove
that 8/10 is fundamentally impossible under a different verified refinement.
