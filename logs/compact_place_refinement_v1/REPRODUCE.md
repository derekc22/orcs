Run in `/home/learning/Documents/orcs-fork`, using the existing `.venv`. Do not install dependencies or access the canonical checkout.

Independent full physics replay, contact attribution and complete video:

```bash
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python \
  scripts/corner_hands_v1/verify_physics.py \
  logs/compact_place_refinement_v1/complete_execution/rollout.npz --video
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/compact_integration_v2/audit.py \
  logs/compact_place_refinement_v1/complete_execution
```

This replays every raw control from the original ground-supported initial condition. Expected qpos/qvel discrepancies and forbidden/self/invalid-terrain impulses are exactly zero. The unchanged legacy verifier always reports `success:false` and also has a historical 45° torso-tilt gate. The separate existing task audit evaluates the actual requested physical gates, explicit grasp review and complete video inspection in `task_assessment.json`.

Regenerate the selected physical placement and recovery using fresh output tags:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/compact_place_refinement_v1/place.py \
  --source logs/compact_place_refinement_v1/complete_execution/rollout.npz --trim 33.505 \
  --tag placement_repro --forward .10 --fork-release-height .085 \
  --transfer-wait 0 --reference-ramp --arm-damping .7 --press-damping 6 --continuous-handoff \
  --plan-cache logs/compact_place_refinement_v1/continuous_transfer/rollout_placement_plan.npz
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/compact_place_refinement_v1/recover.py \
  --source logs/compact_place_refinement_v1/placement_repro/rollout.npz \
  --tag complete_repro
```

The source is the committed complete execution trimmed to its exact 6,701-control accepted pickup/walking/braking/support-transition prefix, avoiding a duplicate intermediate rollout/model. The cache is the unchanged accepted 15-pose placement plan; its hash must match the exact source controls. The script reconstructs solver history by replaying the entire prefix and then appends real actuator commands. It never transplants a saved post-pickup state. Placement generation uses the existing local `logs/corner_hands_v1/reach3.npz` seed even with the cache; this historical input remains local and is not included in this publication. Omitting `--plan-cache` reruns the existing planner. No policy inference or walking/pickup regeneration is needed.

Regenerate metrics and the equivalent-view side-by-side placement comparison:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/compact_smooth_integration_v2/motion_metrics.py \
  logs/compact_place_refinement_v1/complete_execution/rollout.npz \
  --out logs/compact_place_refinement_v1/complete_execution
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/compact_place_refinement_v1/compare_metrics.py \
  logs/compact_place_refinement_v1/complete_execution/rollout.npz \
  --output logs/compact_place_refinement_v1/selected_metrics
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python \
  scripts/compact_place_refinement_v1/compare_video.py \
  logs/compact_place_refinement_v1/complete_execution/rollout.npz \
  logs/compact_place_refinement_v1/selected_comparison_repro.mp4
```

Comparison uses accepted commit `b2673fdb`, equal camera parameters, simultaneous physical time, 50 Hz and 1× speed. It begins at the common 33.505 s placement start and ends with the shorter baseline at 47.610 s; no frozen frames or time alignment is added. The full candidate video covers all 48.145 s. Numerical placement windows use the existing placement-start/recovery-start convention, supplemented by the fixed baseline 10.105 s window and event-based windows. Recorded states, controls, metric inputs and video motion are unfiltered.

The published scripts contain the successful placement/recovery controllers and required analysis/rendering tools. Historical diagnosis and ablation summaries are included as JSON/figures; their diagnostic-only scripts and local-cache inputs remain local. The full audit also uses the existing historical `logs/corner_hands_v1/pickup3_model.mjb`, as documented by the accepted baseline; this duplicate is excluded. Independent raw-control replay needs only the published rollout/model and existing verifier sources. Failed variants, duplicate intermediate rollouts/models, frame atlases, pretrained weights, and unrelated research are excluded. The accepted baseline sources and artifacts remain unchanged. This publication adds one commit after one rerun of the existing accepted verification protocol, without additional experiments or controller refinement.
