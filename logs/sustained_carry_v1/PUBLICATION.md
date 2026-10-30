# Selected sustained-carry result

This commit preserves `complete2.npz`, its standalone MuJoCo model and full video,
the generation/verification scripts, small input poses, walking reference, and
validation evidence. See `VALIDATION.md` for the existing-environment reproduction
commands. Its statement that no commit/push occurred describes the engineering
run before this separately authorized publication.

The existing verifier was run once for publication. All nine sustained-carry gates
and fifteen inherited physical-task gates passed; qpos/qvel replay errors and
forbidden impulse were zero. Output is in `commit_verification_console.txt`.
Machine-only reports deliberately leave visual acceptance unset; `metrics.json`
and `visual_review.json` record the completed full-video inspection.

Files left intact locally and excluded from this commit:

- Duplicate/intermediate models and rollouts in this directory (`complete1*`,
  `pickup_fast1*`, `walk1*`, `walk2*`) and unrelated research.
- Atlas image pages in `complete2_every_frame/` and `complete2_walking_detail/`;
  their small review indexes are included, along with the full video and montage.
- The duplicate generator input `logs/compact_gripper_v1/static6_model.mjb`.
  Generation uses this existing local file; the selected `complete2_model.mjb`
  contains the unchanged physical model and is included for exact replay.
- Existing pretrained weights, cached native motion, and other compact-gripper
  experiments. Generation uses the existing environment and caches.

No previous baseline implementation or artifact was changed.
