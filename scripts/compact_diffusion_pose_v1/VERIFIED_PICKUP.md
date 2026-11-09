# Preserved Goal 7 two-sample pickup

This preservation commit extends Goal 6 commit
`6e09e89fc547dc547582ece30a82c7e620c90cc7` with the verified pickup result.
It does not claim 8/10 diffusion authority. Goal 8/9 implementations and all
experiment-only Goal 7 carry/placement refiners remain local and uncommitted.

Both frozen August HF-BPS samples pass the unchanged 22 physical gates, carry
the object about 2.75 m with 10 alternating replants, place it, and physically
release it. Forbidden impulses and actuator-limit excess are zero; independent
replay is exact and walking/placement quality is retained. Executed pickup
object-path RMS differences from the disabled expert are 113.26/113.97 mm,
and pelvis-height RMS differences are 194.77/192.31 mm for seeds 42/43.

## Implementation dependencies

The existing sources are preserved byte-for-byte:

- `primary_targets.py`, `primary_kinematics.py`, `primary_dynamics.py`, and
  `primary_compile.py` generate the verified pickup-only proposal. These shared
  modules retain their original phase definitions and optional arguments so
  their hashes continue to match the recorded successful compiler provenance.
- `primary_reproduce.py` runs the two-sample physical projection, realization,
  fresh execution, independent replay, rendering, unchanged validation and report.
- `primary_report.py`, `report.py`, `quality.py`, and `plot.py` supply the
  existing comparisons, source attribution and quality measurements.
- `refine.py` and `reproduce.py` preserve the accepted quarter-scale incremental
  tilt baseline used as the pickup compiler's input. `README.md` describes that
  accepted prerequisite; `PRIMARY_README.md` is the historical pickup report.

The ORCS runtime, model, controller, validator and frozen source helpers are
already tracked in Goal 6. No changes to them are included here. Historical
research descriptions in the original documents are not additional experiments
included in this commit.

## Existing local assets and reproduction

From `/home/learning/Documents/orcs-fork`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv/bin/python -B scripts/compact_diffusion_pose_v1/primary_reproduce.py \
  --out logs/compact_diffusion_pose_v1_pickup_reproduction
```

This command requires the existing local accepted Goal 6 references/executions,
the incremental Goal 7 `feasible_{42,43}.npz` references and executions, disabled
expert evidence, frozen proposals/calibration, and accepted policy/model assets.
The reporting path also uses the preserved historical assessments and their
workspace-preservation records. The read-only August checkpoint remains under
`/home/learning/Documents/g1-diffusion`.

Large arrays, models, checkpoints and videos are deliberately not added. This
is not an asset bundle or a promise to reconstruct those assets from a text-only
checkout. Their existing paths and recorded hashes are documented in
`logs/compact_diffusion_pose_v1/verified_pickup_preservation.json`.

## Evidence and preservation checks

`logs/compact_diffusion_pose_v1/primary/consistent/pickup_assessment.json`,
`pickup_source_attribution.json`, and `pickup_comparison.csv` separate native
source, constrained request and executed motion by phase. The two execution
directories contain physical validation and video metadata; the replay
directories contain the exact independent replay records. Seed-43 relative
links and their lightweight targets are both preserved. Large linked artifacts
remain local.

The historical top-level `status: FAILURE` refers to the unmet all-phase 8/10
objective. `candidate_status: SUCCESS` and the two 22/22 physical assessments
describe the pickup result being preserved.

Before committing, only static dependency inspection, recorded gate/quality/
replay checks, and file/provenance hashes were checked. No simulation, frozen
inference, optimization or reproduction was rerun.
