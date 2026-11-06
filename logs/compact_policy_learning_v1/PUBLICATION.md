# Selective publication of the verified learned-policy success

The user explicitly authorized one descriptive commit and a non-force push of the current `main` branch to its existing `origin` remote. The protected parent is `f65e57e648db0a0d8728c8650b42a597797da9b5`, confirmed against remote main before publication. No training, new experiments, controller changes, baseline changes or unrelated cleanup are part of publication.

The existing accepted verification is run once: independent raw-control MuJoCo replay with full nominal video, followed by the unchanged accepted task audit. `publication_verification.json` records the outcome, commands and exact artifact hashes. The research-phase `provenance.json` is retained as a historical snapshot; its pre-publication commit/push fields describe that phase.

## Included

- Selected learned checkpoint `selected_policy.npz`, containing all inference gains and references. It is the newly learned policy, not pretrained weights.
- Inference, training, rollout, independent replay, evaluation and review-generation scripts required by the reproduction instructions. Existing baseline dependencies are already tracked and remain unchanged.
- Training configuration and diagnostics, methodology, selection record, both frozen perturbation protocols, summary, every trial's metrics and failure classifications, feedback sensitivity and matched-ablation checks.
- Accepted nominal `complete_execution/rollout.npz`, full `rollout.mp4`, standalone byte-identical `rollout_model.mjb`, independent physics/contact traces, model/grasp/task audits, motion metrics and explicit visual-review records.
- Ten accepted modest-trial metric/audit/review records and supporting contact, locomotion and grasp traces. Their frame manifests preserve review coverage without committing the frame atlases.

`publication_manifest.json` lists the exact staged paths, sizes and SHA-256 hashes.

## Preserved locally and excluded from Git

- Regenerable training caches: `reference.npz` (about 46 MB) and `dynamics.npz` (about 231 MB). They are unnecessary for inference or exact nominal replay; the published training script regenerates them from the committed engineered expert.
- Duplicate checkpoints `policy.npz` and `reference_only.npz`, pilot dynamics and raw pilot caches, duplicate nominal/development recordings and intermediate diagnostics.
- Raw perturbed-trial rollouts, model links and videos. All trial outcomes remain in `all_trials.csv` / `all_trials.json`; accepted modest-trial audits and contact traces are committed individually. Fresh seeded execution with the selected checkpoint recreates the omitted recordings without training.
- Failed feedback-disabled recordings and other failed variants. Their comparison outcomes and classifications remain in the published metrics.
- Every frame atlas, task-view panel, comparison sheet and montage image, including the reviewed nominal sheets. Review records, coverage manifests, video hashes and the accepted full nominal video are retained.
- Diagnostic-only assembly/comparison scripts `report.py`, `compare_videos.py` and `video_coverage.py`; they consume preserved local research intermediates and are not needed for inference, training or the documented verification.
- Pretrained weights and all unrelated historical experiments.

The existing historical audit comparison model `logs/corner_hands_v1/pickup3_model.mjb` remains local and untracked, matching the engineered baseline's existing reproduction scope. The full task audit needs it in the existing environment; standalone raw-control replay and learned-policy inference use the published models and do not need that comparison cache. No new execution-essential checkpoint, reference or physical model is omitted.

No excluded file is deleted, reset or cleaned. The original engineered success remains committed and unchanged.
