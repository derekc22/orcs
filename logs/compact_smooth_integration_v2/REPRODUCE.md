Run from `/home/learning/Documents/orcs-fork` using the existing `.venv`. No installation, training, morphology change or CUDA runtime is needed to replay the final execution.

Replay the complete recorded actuator sequence and regenerate disjoint contact attribution and video:

```bash
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python \
  scripts/corner_hands_v1/verify_physics.py \
  logs/compact_smooth_integration_v2/complete_execution/rollout.npz --video
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/compact_integration_v2/audit.py \
  logs/compact_smooth_integration_v2/complete_execution
```

The first command reconstructs all solver history from the original initial state. It does not invoke SONIC or a generation controller. Expected replay discrepancies are exactly zero. The inherited raw verifier retains its unconditional `success:false` and legacy torso-tilt field; task acceptance is recorded separately in `task_assessment.json`, using the same physical gates and explicit video review as the accepted baseline.

Regenerate the integration with fresh tags, preserving the demonstrated smoothed walking and braking controls:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  scripts/compact_smooth_integration_v2/transition.py \
  --source logs/compact_smooth_v1/best_partial/rollout.npz \
  --tag repro_entry --align-yaw --seconds 4
OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  scripts/compact_smooth_integration_v2/place.py \
  --source logs/compact_smooth_integration_v2/repro_entry/rollout.npz \
  --tag repro_place --forward .10 --fork-release-height .085 --transfer-wait 0
OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  scripts/compact_smooth_integration_v2/recover.py \
  --source logs/compact_smooth_integration_v2/repro_place/rollout.npz \
  --tag repro_complete
```

Each stage replays the full control prefix before adding controls. No real robot or object state is transplanted. The transition reads `entry_comparison.json` to reproduce its measured 34.046 mm forward COM correction and accepted relative foot-yaw references. Its extra carry damping fades over one second after walking. The physical model and effort caps remain unchanged.

Optional local cache: the final generation used the cached placement plan from `canonical_placement_probe`, generated for that exact source prefix. To reproduce that same plan path, append:

```text
--plan-cache logs/compact_smooth_integration_v2/canonical_placement_probe/rollout_placement_plan.npz
```

Use this cache only for the same source control prefix and the same forward/fork parameters; the script checks the prefix control hash. Reference-plan caching affects neither state integration nor control feedback. Omitting it regenerates the plan using the accepted planner.

Regenerate physical motion metrics, the matched camera comparison, and timing evidence:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/compact_smooth_integration_v2/motion_metrics.py \
  logs/compact_smooth_integration_v2/complete_execution/rollout.npz \
  --out logs/compact_smooth_integration_v2/complete_execution
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python \
  scripts/compact_smooth_integration_v2/compare.py \
  logs/compact_smooth_integration_v2/complete_execution/rollout.npz \
  logs/compact_smooth_integration_v2/comparison_repro.mp4
OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  scripts/compact_smooth_integration_v2/transition_evidence.py
```

The comparison uses simultaneous physical time, equal camera parameters, equal 50 Hz sampling and 1× playback, ending at the shorter trajectory without frozen frames or motion filtering. The complete final video covers all of the longer execution.

Walking contact-local slip measurements are inherited from the exact independently replayed smoothed prefix; `prefix_identity.json` verifies identical controls, qpos and qvel. They can also be recomputed directly:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/compact_smooth_integration_v2/contact_motion.py \
  logs/compact_smooth_integration_v2/complete_execution/rollout.npz \
  --out logs/compact_smooth_integration_v2/walking_slip_repro
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/compact_smooth_integration_v2/contact_motion.py \
  logs/compact_smooth_integration_v2/complete_execution/rollout.npz \
  --out logs/compact_smooth_integration_v2/transition_slip_repro \
  --start-s 29.505 --end-s 33.505
```

The accepted baseline and all prior experiments remain in their original local directories. Publication excludes intermediate rollout/model duplicates, the `canonical_placement_probe` plan cache, failed branches, frame atlases, screenshots and pretrained weights. Regenerating the integration from scratch therefore uses the existing local `compact_smooth_v1/best_partial` prefix (and existing historical inputs documented by the accepted baseline). The optional cache command and failure-comparison plot regeneration also use excluded local research inputs. These inputs are not needed for independent raw-control replay, task auditing, motion/slip measurements or the matched final-video comparison.

The successful smoothing implementation is preserved in `scripts/compact_smooth_v1/walk.py`, `sonic_cpu.py` and `brake.py`. Existing local walking-prefix generation remains available using the accepted `corner_parallel` input with `--waist-mode yaw --abduction .04 --repetitions 2 --arm-damping .7 --press-damping 6`, followed by `brake.py`. No pretrained weights are included.

Before regenerating the comparison figures from their scripts, regenerate baseline motion traces as well:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/compact_smooth_integration_v2/motion_metrics.py \
  logs/compact_integration_v2/corner_recovered/rollout.npz \
  --out logs/compact_smooth_integration_v2/baseline
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/compact_smooth_integration_v2/comparison_plot.py
```

No reset, deletion or repository cleanup was performed. The verified result is being published in one subsequent commit at the user's explicit request.
