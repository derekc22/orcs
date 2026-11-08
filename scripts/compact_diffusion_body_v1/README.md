# Goal 5: selected body motion on the verified Goal 4 execution

From `orcs-fork`, with the retained Goal 4 artifacts and read-only diffusion
repository available:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -B scripts/compact_diffusion_body_v1/reproduce.py --out logs/compact_diffusion_body_v1_reproduction
```

Use a new output directory. `--resume` continues existing output; `--seed 42`
or `--seed 43` runs only the selected verification branch. The retained run is
`logs/compact_diffusion_body_v1/goal5_assessment.json`. Its manifest contains
source and artifact hashes. No commit, push, training or physical changes occur.

## Added motion freedoms

Three independent targets come from genuine frozen August HF-BPS samples 42/43
relative to the fixed calibration sample 0: pelvis height, pelvis world yaw,
and waist yaw. Samples, conditions, gains, bounds and support clock are common.
The native 300-frame curves receive the same low-pass filter, phase mapping,
smooth envelope and bounds. No noise, fitted alignment or post-execution
perturbation is introduced. Fresh inference must exactly reproduce all three
preserved samples.

The base of each enabled sample is its successful Goal 4 `feasible_*.npz`.
Goal 4's hand transforms and object trajectory remain fixed during the offline
body refinement. Both ankle transforms also remain on their Goal 4 trajectories.
Leg and arm joint changes compensate for the body targets. Root xy compensates
to preserve whole-system horizontal COM; these dependent corrections are
identified separately from the independent diffusion inputs. Root roll/pitch
and waist roll/pitch retain the expert reference. All eight compact-hand
coordinates and their nominal feedforward remain unchanged.

## Dynamics and physics-rate details

Every original physics-rate sample is retained. Only additive pose changes,
their velocity corrections, and native-joint servo feedforward differences are
added. Root angular velocity is transported between body frames. Joint targets
include position and damping offsets plus the difference in recursive
Newton-Euler efforts. The controller and its learned feedback matrix are frozen.

The common body clock pauses near the accepted trajectory's low-support/flight
intervals. A centroidal constraint preserves the added whole-system angular
momentum and vertical COM offset while support is unavailable. Its one-tick
extension covers the centered velocity stencil; it releases for subsequent
supported landing dynamics. This avoids Goal 3's unavailable external moments
without reconstructing its failed full-body reference compiler.

Cheap screens reject geometric/contact inconsistency, added joint-limit
excursion, excessive velocity, feedforward or torque increments, and excessive
flight momentum defects. These screens are local approximations. Successful
actual ORCS execution is the final criterion; no standalone MuJoCo simulation
or virtual attachment is used.

## Verification and attribution

`body_final_42.npz` and `body_final_43.npz` are the screened proposals.
`body_projection_*/` executes them in the registered ORCS UOLM task. Only a full
successful physical projection can produce `feasible_*.npz`, using the unchanged
Goal 2 physical-realization helper. `execution_*/` executes the realized reference
again. `replay_*/` independently integrates recorded actions through genuine
ORCS physics. The original validator applies all 22 unchanged gates, including
full videos, model integrity, physical contacts, effort limits and exact replay.

The report compares the two enabled samples, each sample against disabled
diffusion, and each against its own Goal 4 execution. It provides RMS/peak root
translation and rotation, torso orientation, waist joints, fourteen arm joints,
object translation and rotation, and actuator commands in each phase. Both the
original task windows and the active refinement windows are included. Correlation
with the projected source-derived body modes uses a fixed one-step convention;
there is no fitted time shift. Arm/object retention, grasp forces, walking and
placement smoothness are compared with Goal 4.

`executed_body_authority.png` and `.pdf` show actual sample-dependent motion.
The 6–7/10 authority label remains qualitative; numerical thresholds alone do
not establish that rating. Footstep locations/timing, support transitions,
grasp/release mechanics, and the fixed physical model remain expert constraints.

Screened-out candidates and the initial smaller successful sample-42 rollout
are retained as diagnostics. The selected result is always identified by the
final assessment and its explicit paths; rejected references are never promoted.
