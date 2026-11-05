**SUCCESS — the demonstrated smoothed walking execution now completes controlled placement, physical release, clean withdrawal and stable recovery in one continuous, independently replayed execution.**

The selected [complete video](complete_execution/rollout.mp4), [exact rollout](complete_execution/rollout.npz) and [standalone MuJoCo model](complete_execution/rollout_model.mjb) contain 9,522 controls at 200 Hz, lasting **47.610 s**. The video contains 2,380 frames at 50 Hz. The [side-by-side comparison](complete_comparison.mp4) uses simultaneous physical time, equal camera parameters and 1× playback; the final video covers the additional 0.305 s beyond the baseline. No state, video or playback smoothing is used.

The complete execution preserves the previous **26.4% reduction in walking object angular-acceleration RMS and 36.4% reduction in palm-relative angular-rate RMS**. All 5,901 pickup/walk/brake controls and their robot/object states are exactly identical to `compact_smooth_v1/best_partial`. This phase changes only the physical support transition and placement release timing. It does not rebuild or retune the walking controller.

**What prevented integration, and what changed.**

Heading-aligned terminal-state analysis found a changed support geometry rather than a large robot/object pose or velocity jump. At the smoothed brake, the right ankle is 46 mm ahead of the left, versus about 1 mm in the accepted brake. The COM is approximately 34 mm farther behind the ankle midpoint. Both feet initially carry load near the heels; the bounded allocator clips the right heel-moment request. The first bounded control jump is actually smaller than in the successful baseline, so command discontinuity alone does not explain the failed stationary hold. Robot, object, hand, foot, actuator and wrench comparisons are preserved in [entry_comparison.json](entry_comparison.json).

The selected transition uses the existing LegIK and bounded native-torque controller to move the COM **34.046 mm forward over one second**. It then slowly pivots the foot-yaw references toward the accepted relative headings and settles, lasting four seconds in total. The feet and object move only through actual actuator/contact dynamics. Extra carry damping fades over the first second; the static arm preload is captured without freezing a velocity-dependent damping term into it. A short control with the same fade/blend but no COM correction still tipped into self-contact at 2.22 s. The corrected transition retained the grasp and settled without forbidden contacts. See [support comparison](support_transition_comparison.png).

Lowering then exposed a second, distinct mismatch. The corrected foot-yaw branch had **210 ms of continuous terrain support above 70% of box weight, object speed below 0.08 m/s, angular speed below 0.3 rad/s, and object height error below 8 mm**. This satisfies the accepted 150 ms physical-load gate. However, a separate fixed 700 ms wait prevented opening during that valid window. The controller continued holding the grounded box until a left toe struck terrain.

The selected placement removes only that additional fixed wait. It retains the sustained load, height, motion and fork-opening gates and the existing physical opening/withdrawal trajectories. Opening begins at **39.605 s**, after the measured support gate, with **0.624876 N** terrain load against **0.4905 N** box weight. With the fixed wait, opening could not become eligible before 40.200 s; that valid gentle-support window ended at 39.665 s. The two physical runs have **identical controls, qpos and qvel through all 7,921 controls before opening**. The fixed-wait run has 0.007375 N·s invalid terrain impulse; the event-gated run has zero. [Timing ablation](release_timing_ablation.json) and [force/motion plot](release_timing_comparison.png) document this causal comparison.

The accepted bounded balance, placement IK/feedback, grip/terrain-transfer regulation, physical opening, withdrawal and recovery controllers are reused. The selected hand IK has its original behavior: an experimental terrain-clearance penalty was rejected and is disabled. No morphology, mass, object dimensions, contact/friction properties, actuator gains, force caps or physics settings changed. The final model is byte-identical to the accepted model.

**Full physical validation.**

| Measurement | Complete smoothed execution |
|---|---:|
| Initial original-box terrain support | Verified; unchanged pickup prefix |
| Maximum COM lift | 0.787750 m |
| Counted airborne walking interval | 17.245–23.605 s |
| Actual root displacement in airborne walking carry | **2.750144 m** |
| Object COM displacement during walking | 2.793699 m |
| Initial-to-final horizontal object COM transport | 2.776320 m |
| Consecutive alternating qualifying replants | **10, L–R repeated five times** |
| Measured qualifying swing clearance | 52.4–91.1 mm |
| Minimum walking left/right hand force | 0.5976 / 0.3753 N, continuously nonzero |
| Minimum walking COM lift | 0.569751 m |
| Sustained terrain support above 70% weight begins | 39.090 s |
| Physical opening command begins | 39.605 s |
| Terrain load at opening | 0.624876 N, 127.4% of weight |
| Last hand contact | 40.035 s |
| Terrain support after last hand contact | Continuous through the end |
| Recovery begins | 43.610 s |
| Final 1 s maximum root linear/angular speed | 0.002665 m/s / 0.010944 rad/s |
| Final 1 s maximum object COM linear/angular speed | 0.00009595 m/s / 0.00063735 rad/s |
| Minimum final 1 s left/right foot load | 200.24 / 130.86 N |
| Final root height / torso tilt | 0.560398 m / 17.61° |
| Forbidden object / robot self / invalid terrain impulse | **0 / 0 / 0 N·s** |
| Independent exact qpos / qvel replay discrepancy | **0 / 0** |

The box is continuously airborne from 3.120 to 38.920 s. A brief separation during settling at 39.015–39.090 s occurs while physically retained. Opening follows terrain load transfer; terrain contact remains continuous after the last hand contact. Final carriages are approximately −25 mm, forks approximately 0.016 rad and thumbs approximately zero. Both hands remain clear. Walking distance excludes braking, the COM correction, foot pivot and placement.

The unchanged [independent raw-control verifier](complete_execution/rollout_verification.json), [contact/force trace](complete_execution/rollout_verification_trace.npz), [footstep trace](complete_execution/rollout_locomotion_trace.npz), [task metrics](complete_execution/task_metrics.json) and [task assessment](complete_execution/task_assessment.json) pass every requested physical gate. [Physical integrity](complete_execution/physical_integrity.json) verifies preserved geometry, contact properties and effort limits. External forces, object actuators and equality constraints are zero. The raw verifier retains its legacy unconditional `success:false` and 45° torso-tilt field, as documented for the baseline; the explicit task assessment is **SUCCESS**.

**Preserved smoothness, slip, and regressions.**

All walking comparisons use the same 6.36 s physical interval, identical sampling and raw velocities/accelerations. Full-run comparisons use the complete 47.305 s baseline and 47.610 s refined execution. See [walking traces](walking_motion_comparison.png), [phase tradeoff plot](smoothness_tradeoff.png), [complete motion metrics](complete_execution/motion_metrics.json), and [before/after JSON](comparison_metrics.json).

| RMS physical measure | Committed baseline | Complete smoothed execution | Change |
|---|---:|---:|---:|
| Walking object angular acceleration | 35.401 rad/s² | 26.060 rad/s² | **−26.4%** |
| Walking combined palm-relative angular rate | 1.2532 rad/s | 0.7969 rad/s | **−36.4%** |
| Walking object angular speed | 1.5143 rad/s | 1.2511 rad/s | −17.4% |
| Walking root angular speed | 1.0602 rad/s | 1.0302 rad/s | −2.8% |
| Braking object angular acceleration | 15.166 rad/s² | 11.074 rad/s² | −27.0% |
| Placement object angular acceleration | 22.423 rad/s² | 27.462 rad/s² | **+22.5% regression** |
| Placement object angular speed | 0.4720 rad/s | 0.5622 rad/s | +19.1% regression |
| Placement root angular acceleration | 13.679 rad/s² | 11.593 rad/s² | −15.3% |
| Full-run object angular speed | 0.6873 rad/s | 0.5826 rad/s | −15.2% |
| Full-run object angular acceleration | 19.303 rad/s² | 18.008 rad/s² | −6.7% |
| Full-run root angular speed | 0.6361 rad/s | 0.5593 rad/s | −12.1% |
| Full-run root angular acceleration | 17.326 rad/s² | 15.641 rad/s² | −9.7% |

Placement retains visible compliant rocking, especially near the floor. Its object acceleration is worse even though it completes safely. This result preserves the full demonstrated walking gains and improves complete-execution motion measures; it does not claim uniform smoothness improvement across every phase. Final root linear-speed maximum increases from 0.001947 to 0.002665 m/s, remaining below the unchanged 0.01 m/s stability gate. Final recovery is about 28 mm lower than the baseline crouch. Minimum grip loads are lower, as already measured in the smoothed prefix.

| Walking grasp measurement, left/right | Baseline | Complete smoothed execution |
|---|---:|---:|
| Peak palm-relative rocking | 14.52° / 9.68° | 9.09° / 7.02° |
| Peak fixed-material-point tangential excursion | 13.14 / 5.31 mm | 8.01 / 5.03 mm |
| Final fixed-material-point tangential excursion | 7.19 / 1.17 mm | 1.70 / 2.19 mm |
| Loaded-contact tangential-speed RMS | 69.72 / 44.66 mm/s | 43.53 / 34.77 mm/s |

The right final material-point excursion increases; zero slip is not claimed. During the added support transition, left/right peak relative rotation is only 1.77°/1.80°, local tangential excursion 0.53/1.87 mm, and loaded-contact tangential-speed RMS 2.14/2.28 mm/s. [Walking slip](complete_execution/contact_slip.json), [transition slip](complete_execution/transition_contact_slip.json) distinguish material-point excursions from force-weighted relative velocity at actual loaded contacts. The latter is calculated using point Jacobians in the contact tangent plane and the solver's pre-integration velocity. The walking measurements transfer exactly because [prefix identity](complete_execution/prefix_identity.json) verifies identical complete physical states and controls.

**Inspection, reproduction, and preserved evidence.**

All 2,380 final-video frames are covered: frames 0–1,407 are pixel-identical to the previously completely inspected smoothed video; frames 1,408–2,379 were directly inspected in chronological sheets 22–37. [Pixel comparison](complete_execution/video_prefix_identity.json) records zero error for every inherited frame. [Visual review](complete_execution/visual_review.json) covers acquisition, real walking, support transition, lowering, terrain transfer, opening, separation, withdrawal and final stability. The existing carry comparison and the new full comparison retain the visible walking improvement; remaining placement rocking is explicitly reported.

Short failed tests remain preserved: damping fade alone failed stationary support; COM correction fixed it; direct/lateral/foot-yaw lowerings with the fixed wait hit a toe on terrain; the terrain-clearance penalty failed; alternative deep-plan initialization did not remove the issue. Deep ankle/COM residuals alone were not a valid distinguishing explanation because the successful baseline also has comparable residuals. The decisive late-stage evidence is the paired fixed-wait/event-gated-release comparison. No broad gain search, policy training, new policy inference or morphology exploration was run in this integration phase.

[Reproduction instructions](REPRODUCE.md) cover exact replay, the local transition, reused placement/recovery generation, metrics and matched video rendering. [Integration assessment](complete_execution/integration_assessment.json) combines full physical acceptance with retained meaningful smoothing and explicit regressions. [Provenance](provenance.json) records artifact hashes, identical model bytes, unchanged HEAD and no tracked worktree differences.

The accepted baseline, prior smoothing results, previous dirty-tree research and all diagnostic branches are preserved. No reset, deletion, commit or push was performed during the engineering phase. At the user's subsequent request, this verified result is selectively published in one commit after one additional run of the existing verification protocol. Failed variants, duplicate intermediate artifacts, frame atlases, screenshots and pretrained weights remain local and unstaged; optional generation commands using those historical inputs are identified in `REPRODUCE.md`.
