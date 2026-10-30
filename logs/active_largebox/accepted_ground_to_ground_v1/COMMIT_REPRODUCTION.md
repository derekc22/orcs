# Committed success and local artifact dependencies

This is **not a self-contained fresh-clone distribution**. The successful
implementation, accepted controls/states, compressed accepted model, full video,
contact trace, lightweight metrics/configuration and validation report are in Git.
Git LFS is not configured. The accepted model is stored as lossless
`expert_model.mjb.gz` (about 43 MB), with raw/compressed hashes in
`storage_manifest.json`; replay restores verified bytes into a temporary cache.
All original large artifacts and useful packaging work remain preserved locally.

`LOCAL_ARTIFACTS.json` lists their exact paths, sizes and SHA-256 checksums.
In particular, **not stored remotely** are the uncompressed model working copies,
three intermediate/calibration models, pretrained SONIC weights, the original
20 s source/aftermath rollout, exhaustive frame atlases, and archival source
snapshots. Existing earlier research is also not added by this commit.

## Existing-environment verification

From `/home/learning/Documents/orcs-fork`:

```bash
.venv/bin/python scripts/reproduce_active_largebox.py \
  logs/active_largebox/accepted_ground_to_ground_v1/expert.npz
```

Add `--video` and `MUJOCO_GL=egl` for a fresh render. The wrapper creates a new
output directory and replays all 3,828 physics-rate controls before the 15-gate
audit. The full audit **requires** the preserved local reference model at
`generation_inputs/original_largebox_vertical_rail_model_v1.mjb.gz` in this bundle.
Restore it from the documented local archive before running that audit on another
checkout. The standalone physics verifier can replay the versioned model without
this reference model, but cannot by itself certify original-geometry equivalence.

Python 3.11 packages for replay are pinned in `requirements-replay.txt`. They
must be installed separately. GPU controller regeneration additionally requires
the standard project dependencies/assets pinned in `deps.lock` and `uv.lock`,
CUDA, and the local intermediate models/checkpoint listed above. Small motion,
posture and geometry calibration inputs used by the generator are versioned.
Pretrained weights retain their upstream terms; they are not bundled in Git.
No new controller training or research is part of this commit.

`GIT_SHA256SUMS` covers this versioned bundle subset. Historical validation and
original archive hashes describe the original validation session, before the
subsequent user-authorized commit/push. The original 450a875 baseline remains
unchanged and all unrelated local experimental scripts are excluded.
