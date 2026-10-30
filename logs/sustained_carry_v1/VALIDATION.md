# Sustained original-largebox walking carry — verified success

**G1 moves its root 2.526 m during the explicit airborne walking interval, then brakes, places, opens and withdraws cleanly.** The complete 24.90 s execution independently replays with zero qpos/qvel error. All nine locomotion/task gates and all fifteen inherited physical task gates pass. Every one of the 1245 full-video frames, plus every walking frame at larger detail scale, was visually inspected.

## What changed and why

The earlier reference contained about three seconds of near-standing motion, less than one second of translation, then a braking/hold tail. Independent diagnosis found only 0.825 m net root displacement for the accepted long clamp and 0.868 m for the previous compact execution during their longest airborne intervals. Their object displacement was larger. Those earlier object-distance reports did not establish sustained robot locomotion.

This execution uses the **unchanged proven active-grasp model**. It replaces the short reference with a continuous local native walking clip, aligns it to the physical carrying stance, applies actual-state SONIC feedback, and keeps the grasp stabilized through translation and yaw. No gripper appearance or morphology work was performed.

The first long walk already achieved 2.504 m root movement and valid alternating steps, but late set-down exhausted finger travel and landed too quickly (0.345 m/s vertical, 0.302 m/s horizontal). It was rejected. The final change removes avoidable airborne idle time and speeds the physical pickup transition while preserving the **same 3.66 s walking segment**. Lowering begins shortly after braking instead of after a long stationary hold. The final landing is gentle and release is earned by actual terrain weight support.

## Root distance is separate from object distance

The required walking interval is **5.20–8.86 s** (physics indices 1040–1771). Its phase bounds are defined by the reference builder before generation. It excludes pickup, carry preparation, braking, lowering, release and withdrawal.

| Strict walking-only measurement | Result |
|---|---:|
| Root horizontal Euclidean net displacement | **2.526 m** |
| Root horizontal path length (not used for 2 m gate) | 2.599 m |
| Object COM horizontal net displacement in same interval | 2.569 m |
| Actual walking interval duration | 3.66 s |
| Mean net root speed | 0.690 m/s |
| Fraction with measured forward root speed >0.15 m/s | 99.73% |
| Root net / path ratio | 0.972 |
| Whole-foot net displacement, left/right | 2.387 / 2.708 m |
| Complete qualifying alternating replants wholly within interval | **5: R–L–R–L–R** |
| Peak whole-foot sole clearance, left/right | 7.95 / 5.77 cm |
| Object–terrain contact steps during walking | **0** |
| Minimum measured clamp contact force per opposing jaw | 2.920 / 3.373 N |

Footsteps require >=60 ms off loaded support, >=20 mm whole-foot collision clearance, >=120 mm same-foot relocation, and a loaded replant. Loaded foot support uses measured normal force >=10 N. Preparatory/partial and braking steps are conservatively excluded from the five-step count. The 10 N force threshold can produce brief unloaded intervals near support exchange; the full raw force/contact states are retained rather than hidden by a gait label.

| Step | Foot | Swing interval (s) | Same-foot relocation (m) | Peak sole clearance (cm) |
|---|---|---:|---:|---:|
| 1 | right | 5.640–6.125 | 0.895 | 5.49 |
| 2 | left | 6.290–6.845 | 0.955 | 6.13 |
| 3 | right | 6.940–7.545 | 0.981 | 5.77 |
| 4 | left | 7.660–8.200 | 0.885 | 5.27 |
| 5 | right | 8.280–8.760 | 0.739 | 5.32 |

The forward distances between successive alternating replants are **0.571, 0.466, 0.490, 0.311 m**. These are distinct from the longer same-foot relocation/stride measurements above. Foot positions, minimum collision clearances, normal loads, and root states are recorded at 200 Hz in `complete2_locomotion_trace.npz`, with an embedded schema; event positions/times are in `complete2_step_events.json`. See `complete2_foot_trajectories.png`.

For comparison over each execution's **entire longest uninterrupted airborne interval**, using net displacement for both robot and box:

| Execution | Root net (m) | Object net (m) | Terrain-free duration (s) |
|---|---:|---:|---:|
| Committed active clamp | 0.825 | 1.122 | 4.820 |
| Previous compact execution | 0.868 | 0.952 | 11.415 |
| New sustained carry | 2.599 | 2.614 | 10.370 |

The new full airborne interval is 2.135–12.505 s, lasting **10.370 s**. This includes pickup/settling/lowering time and must not be confused with the shorter explicit walking interval. The 2 m gate uses the stricter walking-only measurement.

## Pickup, placement and physical integrity

The box retains original 0.473572 × 0.461602 × 0.409559 m collision geometry, 0.050 kg mass, accepted finite friction [2.5, 0.1, 0.003] and contact settings. Mesh vertices and geom poses match the reference exactly. Maximum physical COM lift is **32.81 cm**.

First terrain entry occurs at 12.505 s with **0.0085 m/s horizontal, 0.0532 m/s vertical**, and 0.0185 rad/s angular speed. Terrain accepts >=70% of the payload weight while the gripper remains engaged for **2.820 s**. Deliberate physical opening starts at 15.405 s; full upward withdrawal finishes at 23.205 s. Post-opening slide is 4.11 mm, COM height range 1.04 mm, and post-withdrawal gripper/object contact steps **0**. Final robot and box are stable.

Strict non-clamp robot–object impulse, robot self-contact impulse, and invalid robot–terrain impulse are **0 / 0 / 0 N·s**. Native hands are also excluded from allowed clamp contacts in the strict attribution. Exact replay qpos/qvel errors are **0 / 0**. All native motor/gripper force caps are respected; equality count, disabled physics flags, and external applied forces are zero. The frozen model comparison confirms geometry, masses, inertias, collision masks, friction/contact parameters, joint ranges, and actuator settings are unchanged from the proven mechanism.

The actual simulation receives only finite actuator commands. There is no object actuator, weld, kinematic attachment, fixture, external support force, body/foot propulsion of the box, or post-initialization actual object-state edit. Phase continuation replays the complete prefix controls to preserve solver state. Separate planning and SONIC observation data never transport the physical object.

## Video and exact reproduction

- Full execution: `complete2.mp4` (24.90 s, 1,245 frames, 50 Hz).
- Left panel: fixed world camera, 1 m ruler ticks, 0.5 m grid and actual green root trail. Right panel: contact/foot-detail following view. Ruler/grid/trail are rendering decorations absent from the physics model.
- Entire execution atlas: `complete2_every_frame/index.json`; every walking frame at larger detail scale: `complete2_walking_detail/index.json`.
- Standalone MuJoCo model: `complete2_model.mjb`; exact 200 Hz controls/states: `complete2.npz`.
- Independent acceptance: `complete2_sustained_metrics.json`, `complete2_acceptance_audit.json`, `complete2_verification.json`, `complete2_verification_trace.npz`, `complete2_locomotion_trace.npz`.
- Final reviewed acceptance: `metrics.json`, `visual_review.json`, `contact_attribution.json`.

From `/home/learning/Documents/orcs-fork`, using the existing environment:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/verify_sustained_carry.py \
  logs/sustained_carry_v1/complete2.npz
```

This independently steps the complete model from the initial anchor through all 4,980 controls and exits nonzero if any task or sustained-walking machine gate fails. It uses no generation controller, observation simulator or pretrained weights. The original-object geometry audit uses the already preserved accepted reference model. Regenerate the video after verification with:

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/render_sustained_carry.py \
  logs/sustained_carry_v1/complete2.npz
```

Generator chain (use fresh tags to preserve selected artifacts):

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/sustained_carry_pickup.py --tag pickup_repro
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python scripts/sustained_carry_run.py \
  --source logs/sustained_carry_v1/pickup_repro.npz --prefix 880 --tag walk_repro
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/sustained_carry_place.py \
  --source walk_repro --tag full_repro --trim-seconds 10.4
```

The 10.4 s handoff is after all walking and braking (walking ends at 8.86 s); it removes only a stationary carry hold. Generation uses existing small posture inputs and cached native motion/checkpoint. No pretrained models or large research archives were repackaged, and no new training or gripper redesign was performed. One expert execution is verified; disturbance robustness is not claimed.

Both the committed `6ed4dcec` accepted bundle and the uncommitted compact-gripper primary artifact hashes still match. HEAD is unchanged. No commit or push was performed.
