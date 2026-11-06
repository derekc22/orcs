# Reproduction in the existing environment

Work only in `/home/learning/Documents/orcs-fork` and use its `.venv`. No installs, model rebuilding, canonical checkout access, commit or push are needed. The protected baseline is `f65e57e648db0a0d8728c8650b42a597797da9b5`.

The [publication manifest and exclusions](PUBLICATION.md) identify which accepted artifacts are committed. The selected checkpoint, accepted nominal rollout/video/model, required scripts, reports, and modest-trial metric/contact traces are included. Training caches, duplicate runs, perturbed-trial raw rollouts/videos, and frame atlases remain local. Their absence does not prevent policy inference or fresh seeded execution. Exact replay of an omitted stored perturbation first requires its documented fresh seeded execution. The inherited full task audit also uses the existing local `logs/corner_hands_v1/pickup3_model.mjb`, as it did for the committed engineered baseline; standalone raw-control replay does not need that historical comparison model.

## Exact stored nominal replay and full video

```bash
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python \
  scripts/compact_policy_learning_v1/replay.py \
  logs/compact_policy_learning_v1/complete_execution/rollout.npz --video
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/compact_integration_v2/audit.py \
  logs/compact_policy_learning_v1/complete_execution
```

Expected: exact qpos/qvel errors 0/0, zero forbidden/self/invalid-terrain impulses, 2.750144 m retained robot-root walking, ten alternating replants, and `task_assessment.json` status `SUCCESS`. The legacy raw verifier's unconditional `success:false` remains unchanged. The separate accepted task audit uses the actual task criteria and stored explicit video/grasp review.

`replay.py` calls the unchanged accepted verifier with a read-only eager NPZ cache. It changes no simulation or measurement code. To use the historical entry point directly, substitute `scripts/corner_hands_v1/verify_physics.py`; its repeated lazy decompression is slower.

## Fresh closed-loop execution

Choose an output directory that does not exist; rollout creation refuses to overwrite it.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/compact_policy_learning_v1/rollout.py \
  --checkpoint logs/compact_policy_learning_v1/selected_policy.npz \
  --out logs/compact_policy_learning_v1/reproduced_nominal
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python \
  scripts/compact_policy_learning_v1/evaluate.py \
  logs/compact_policy_learning_v1/reproduced_nominal --video
```

The first command runs policy inference against current MuJoCo robot/object observations throughout the complete task. It records every raw actuator command and resulting state from the original ground-supported initial condition. Only initial qpos/qvel assignment and subsequent actuator commands are used. No expert mid-task state is transplanted. The second command performs a separate raw-control replay, attributes contacts and writes `evaluation.json` plus a full 50 Hz video.

Fresh outputs initially have `visual_review_complete:false`; a new video is not automatically declared inspected. The three planned stored nominal runs have exactly identical qpos, qvel, controls and contacts (`nominal_repeat_identity.json`). A deterministic reproduction can be compared with the reviewed complete execution before inheriting its visual evidence.

Direct use of the inference class with the standalone model:

```python
import sys
import mujoco
sys.path.insert(0, 'scripts/compact_policy_learning_v1')
from controller import Policy

model = mujoco.MjModel.from_binary_path(
    'logs/compact_policy_learning_v1/complete_execution/rollout_model.mjb')
policy = Policy(model, 'logs/compact_policy_learning_v1/selected_policy.npz')
# Initialize a fresh MjData with the intended valid initial state, mj_forward,
# then for each 5 ms step: data.ctrl[:] = policy.control(step, data);
# mujoco.mj_step(model, data). Construct a new Policy for each episode.
```

## Independently seeded modest trials

Stored local trials are `expanded_test_71106001` through `expanded_test_71106010`. Every one has an exact rollout, model link, full video, independent contact/locomotion traces, task/grasp/model audits, and explicitly completed visual review. Publication includes their metric/audit/contact traces and review records; raw perturbed rollouts/videos and frame panels remain local.

Regenerate a trial using the declared distribution, without changing it:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/compact_policy_learning_v1/rollout.py \
  --checkpoint logs/compact_policy_learning_v1/selected_policy.npz \
  --profile modest --seed 71106001 \
  --out logs/compact_policy_learning_v1/reproduced_modest_71106001
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python \
  scripts/compact_policy_learning_v1/evaluate.py \
  logs/compact_policy_learning_v1/reproduced_modest_71106001 --video
```

Repeat with seeds 71106002–71106010 for the full independent set. The `small` profile uses seeds 71105001–71105010 and the smaller magnitudes in `evaluation_protocol.json`; the expanded profile is recorded separately in `expanded_protocol.json`. The selected checkpoint is identical for both sets. Every trial's exact realized perturbation, initial contacts and validity are saved in `generation.json`.

To regenerate the matched ablation, use the same checkpoint, profile and seed, add `--feedback-off`, and choose a different fresh output directory. Expected result: pickup balance/terrain-contact failure, zero qualifying walking and complete-task failure. This is equivalent to frozen committed expert control replay with native servos; it does not regenerate all historical online stage controllers.

For saved trials, replay with `replay.py` and rerun the accepted audit as in the nominal commands, replacing the directory. To refresh the new wrapper's numerical assessment from already verified traces without another simulation, use `evaluate.py DIRECTORY --reuse-replay`.

## Training and retained data

`selected_policy.npz` contains double-precision K matrices, qpos/qvel/actuator references, correction bounds, cost arrays and stride. The selected checkpoint SHA-256 is `841a42fe5f4e2f163f6b279c090557293bd7b835b9a9f964a7881ed5511e3dbe`. The duplicate first fitted `policy.npz` is preserved locally with the same contents and excluded from Git.

The original full fit used 8 workers, seed 11052026, stride 4, 256 orthogonal probes per interval, and default tracking/action costs. `training_configuration.json` stores all costs, dimensions, bounds and timing; `training.json` includes prediction checks and reference/model hashes. The local, regenerable `reference.npz` retains exact expert states, controls and solver/integration states for generating probes. The local `dynamics.npz` retains all fitted transition models. These caches are excluded from Git and are never used to reset an evaluated execution. The training command below regenerates them from the already committed engineered expert.

Reproduce the pilot/full fit in **fresh directories**, preserving the selected artifacts:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/compact_policy_learning_v1/train.py --mode pilot \
  --run-dir logs/compact_policy_pilot_reproduction \
  --workers 8 --stride 4 --samples 256 --seed 11052026
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/compact_policy_learning_v1/train.py --mode train \
  --run-dir logs/compact_policy_training_reproduction \
  --workers 8 --stride 4 --samples 256 --seed 11052026
```

The full fit takes about 107 seconds on the tested CPU. It optimizes one finite-horizon linear residual policy from empirically learned physical dynamics; no neural actor, GPU training, hyperparameter sweep or checkpoint selection against held-out trials is involved.

## Visual coverage and result tables

The original full-resolution videos are unmodified. `video_sheets.py VIDEO FRESH_DIRECTORY` creates full-view every-frame sheets as used for the nominal review. `task_video_panels.py VIDEO` creates five chronological every-frame robot/box/nearby-terrain panels plus a manifest, as used for all ten modest trials. Panel generation does not automatically mark a review complete. Physical contact/retention claims also require the independent 200 Hz traces and grasp audit.

`ROBUSTNESS.md`, `all_trials.csv`, `all_trials.json`, `SUMMARY.json`, `expert_comparison.json`, `feedback_sensitivity.json`, and `paired_ablation_integrity.json` provide the full evaluation and causal-feedback evidence. `provenance.json` identifies the exact model, expert, checkpoint, nominal rollout, video and implementation hashes. No artifact replaces or modifies the accepted engineered expert.
