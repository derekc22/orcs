# Verified original-largebox ground-to-ground execution

**Complete physical execution achieved and independently replayed.** The continuous
19.14 s expert contains ground reach-down, powered acquisition, free pickup,
walking transport, braking and lowering, terrain load transfer, deliberate opening,
withdrawal, and final stabilization. All machine acceptance gates pass. Every one
of the 957 rendered 50 Hz frames was visually inspected, including the aftermath.

The box retains the exact repository collision decomposition and normal resting
orientation: **0.473572 × 0.461602 × 0.409559 m, 0.050 kg**. Mesh vertices and
object geom poses match the original model exactly.

| Physical measurement | Verified result |
|---|---:|
| Actual reach-down before acquisition | 8.65 cm |
| Maximum object COM lift | 14.53 cm |
| Terrain-free task frames | 241 (4.82 s) |
| Consecutive bilateral-supported free lift above 5 cm | 167 (3.34 s) |
| Terrain-free supported carry | 1.081 m |
| Transport in the principal uninterrupted airborne interval | 1.145 m |
| Left / right foot motion during airborne carry | 0.775 / 0.895 m |
| Left-only / right-only support during airborne carry | 25 / 36 frames |
| First-terrain-entry horizontal / vertical speed | 0.127 / 0.102 m/s |
| First-terrain-entry angular speed | 0.269 rad/s |
| Terrain accepting ≥70% of weight while the gripper still controls the box | 2.58 s measured overlap |
| Placement movement from first terrain contact | 12.0 mm |
| Movement after opening starts | 5.69 mm |
| Post-opening COM height range / maximum tip change | 1.58 mm / 0.66° |
| Forbidden robot-object / invalid terrain / robot-self impulse | **0 / 0 / 0 N·s** |
| Maximum replay qpos / qvel error | **0 / 0** |

The grasp uses a **0.301 kg wrist-mounted opposing clamp**, six genuinely driven
pistons (270 mm stroke, 2 N force caps), and one additional wrist axis capped at
5 Nm. The native limbs, waist, feet, and native motor torque limits are preserved.
The clamp is substantial: its initial grasp centroid is 0.757 m from the wrist,
and its farthest pad is 0.943 m away. These dimensions are explicitly disclosed;
the result relies on this long distal mechanism. Arm servo gains are increased
within the original motor force limits. The full configuration and relaxation
rationale are in [engineering_manifest.json](engineering_manifest.json).

The common jaw frame solves the failed split-wrist retention mode. Initial
130 mm fingers exhausted travel on the irregular mesh; the longer, collidable
pistons retain physical pressure. Finite-force closure and near-ground feedback
precede measured terrain weight transfer. The controller then holds the measured
clamp pose, opens all fingers outward at 5 cm/s, and raises the open frame 8 cm.
All end-effector geometry is free of object contact after withdrawal.

No object actuator, weld, equality constraint, external support, artificial object
force, post-initialization object state write, floor elevation, object handle,
collision loophole, or kinematic object transport is used. Object manipulation
comes from actual end-effector contacts and finite robot/gripper actuator forces.
Original-object friction remains the accepted finite `[2.5, 0.1, 0.003]`.

Mean vertical end-effector support during the uninterrupted airborne interval is 0.4889 N against a 0.4905 N payload weight.

The model runs at 200 Hz. `initial_qpos` / `initial_qvel` are the pre-step anchor;
`ctrl_physics` contains every physics-rate control and recorded states are
post-step. Landing speeds use the pre-contact state, not impact-reduced velocity.
Fresh-process CPU MuJoCo replay matches all 3,828 recorded physics states exactly,
without the generation controller or its separate SONIC observation adapter.

The expert stops at the **first one-second quiet goal window**, after 957 task frames (last zero-based video frame 956).
`full_source_20s.npz` preserves the original uninterrupted 20 s rollout and its
remaining aftermath. No controls or states were edited to obtain the goal prefix.
The later source aftermath remains upright, object-stable and contact-valid.

Reproduce from `/home/learning/Documents/orcs-fork`:

```bash
MUJOCO_GL=egl .venv/bin/python scripts/reproduce_active_largebox.py \
  logs/active_largebox/accepted_ground_to_ground_v1/expert.npz --video
```

This creates a fresh output directory, replays the complete model, renders the
full video, and reruns the independent audit. It exits successfully only when all
machine gates pass. Visual review is recorded separately in [metrics.json](metrics.json).

Artifacts: [exact controls and states](expert.npz), [standalone model, lossless gzip](expert_model.mjb.gz),
[full video](full_execution.mp4), [contact attribution](contact_attribution.json),
[physics contact trace](physics_contact_trace.npz), [trace schema](trace_schema.json),
[metrics and visual audit](metrics.json), and [every-frame review](every_frame_review_com/index.json).
Exact generation and verification sources accompany this directory. The accepted
`450a875` baseline and all previous research evidence remain intact. No commit or
push was performed. This validates one expert execution; policy distillation and
statistical robustness were outside this completed execution audit.

Checkout storage and dependency instructions: [COMMIT_REPRODUCTION.md](COMMIT_REPRODUCTION.md).
