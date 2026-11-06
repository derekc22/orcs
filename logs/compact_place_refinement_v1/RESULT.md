**SUCCESS — smoother controlled placement in a complete, independently replayed ground-to-ground G1 execution.**

The [complete video](complete_execution/rollout.mp4), [exact 9,629-control rollout](complete_execution/rollout.npz) and [standalone MuJoCo model](complete_execution/rollout_model.mjb) cover 48.145 s at 200 Hz. The [side-by-side placement comparison](selected_comparison.mp4) uses accepted commit `b2673fdb`, simultaneous physical time, equal cameras, 50 Hz and 1× playback. Its shared interval is 33.505–47.610 s; the complete candidate video includes its additional 0.535 s.

The dominant roughness was generated during descent, after a nearly motionless stationary entry. Sampled arm position references excited the compliant grasp, and the carriage force law canceled native damping. Interpolating arm position references at 200 Hz and adding placement-only arm/carriage damping removes much of that excitation. Preserving the arm reference at terrain-load confirmation prevents a second opening-step disturbance. [Diagnosis](DIAGNOSIS.md), [intervention data](intervention_evidence.json), [handoff ablation](handoff_ablation.json) and the [controller implementation](../../scripts/compact_place_refinement_v1/arms.py) document the physical evidence and rejected variants. No walking, pickup, hand, box, contact, force-cap or model changes were made.

All main RMS measurements use the existing unfiltered 200 Hz method and the existing placement-start/recovery-start window convention. Baseline duration is 10.105 s; candidate duration is 10.640 s. A fixed 10.105 s comparison and integrated acceleration energy establish that the improvement is not caused by extra settling time.

| Physical measurement | Accepted b2673fdb | Selected refinement | Change |
|---|---:|---:|---:|
| Placement object angular-acceleration RMS | 27.462 rad/s² | 7.344 rad/s² | -73.3% |
| Fixed 10.105 s object angular-acceleration RMS | 27.462 rad/s² | 7.536 rad/s² | -72.6% |
| Placement integrated squared angular acceleration | 7620.727 rad²/s³ | 573.902 rad²/s³ | -92.5% |
| Placement object angular-speed RMS | 0.562 rad/s | 0.252 rad/s | -55.2% |
| Initial-contact angular speed | 0.882 rad/s | 0.328 rad/s | -62.8% |
| Initial-contact downward COM speed | 0.172 m/s | 0.048 m/s | -72.3% |
| Initial-contact tilt | 7.818 ° | 6.065 ° | -22.4% |
| Maximum tilt during load transfer | 7.818 ° | 7.119 ° | -8.9% |
| Tilt range during load transfer | 4.679 ° | 3.847 ° | -17.8% |
| Terrain-load peak through hand separation | 3.545 N | 1.409 N | -60.3% |
| Peak COM speed in first 0.5 s after opening command | 0.208 m/s | 0.116 m/s | -44.0% |
| Peak angular speed in first 0.5 s after opening command | 0.686 rad/s | 0.581 rad/s | -15.3% |

Tilt uses the original terrain-supported box up axis; it is not a camera-dependent angle. Raw world angular velocity is differentiated exactly as in the accepted implementation. [Complete before/after JSON](selected_metrics/placement_comparison.json), [physical traces](complete_execution/motion_trace.npz), and [comparison plot](selected_metrics/placement_comparison.png) retain all windows and peaks.

The free settling after the last hand contact is a remaining tradeoff. In the same 0.5 s window, peak angular speed increases from 0.258 to 0.581 rad/s and peak COM speed from 0.050 to 0.110 m/s. Maximum orientation departure from the final settled pose increases from 0.33° to 0.61°. Time from separation to sustained linear/angular settling below 0.001 m/s and 0.003 rad/s is 0.200/0.205 s versus 0.230/0.240 s. The final free rock is small but not improved; uniform improvement is not claimed. The larger descent and touchdown disturbances are substantially reduced, terrain remains continuously supporting the released box, and both executions settle stably.

| Event / preservation check | Selected execution |
|---|---:|
| Initial terrain contact | 39.105 s |
| Start of sustained >70% weight support | 39.145 s |
| Compliant load transfer | 39.425 s |
| Deliberate opening | 40.140 s |
| Last hand contact | 40.440 s |
| Recovery begins | 44.145 s |
| Terrain load at opening / box weight | 0.588592 / 0.490500 N |
| Actual root walking displacement | 2.750144 m |
| Alternating qualifying foot replants | 10 |
| Walking object angular-acceleration / palm-relative angular-rate RMS | 26.060 rad/s² / 0.796893 rad/s, exactly unchanged |
| Minimum walking left/right hand force | 0.597582 / 0.375330 N, exactly unchanged |
| Final 1 s maximum object COM linear/angular speed | 0.00009904 m/s / 0.00063735 rad/s |
| Final 1 s maximum root linear/angular speed | 0.002863 m/s / 0.010467 rad/s |
| Minimum final 1 s foot loads, left/right | 202.53 / 128.31 N |
| Forbidden / self / invalid-terrain impulses | 0 / 0 / 0 N·s |
| Independent exact qpos / qvel replay error | 0 / 0 |

All 6,701 pre-placement controls, qpos, qvel and contacts are exactly identical to the accepted execution; the model is byte-identical. Pickup, the 2.750144 m genuine airborne carry, ten alternating replants with 52.4–91.1 mm clearance, and the prior 26.4%/36.4% walking gains are preserved. Walking palm-relative rocking remains 9.09°/7.02° and material-point tangential excursions 8.01/5.03 mm. These are bounded compliant retention, not zero slip. Final root linear speed is slightly higher than the baseline (0.00286 versus 0.00267 m/s), still below the unchanged 0.01 m/s stability gate.

The unchanged [independent replay/contact attribution](complete_execution/rollout_verification.json), [200 Hz contact trace](complete_execution/rollout_verification_trace.npz), [footstep/hand-force trace](complete_execution/rollout_locomotion_trace.npz), [task metrics](complete_execution/task_metrics.json), [task assessment](complete_execution/task_assessment.json) and [model integrity](complete_execution/physical_integrity.json) pass every requested physical gate. The legacy verifier retains its unconditional `success:false` and historical torso-tilt Boolean; the separate unchanged task audit records **SUCCESS** against the actual requested criteria. No equality constraints, object actuators, external forces or disabled physics are used.

Complete visual coverage is recorded in [visual_review.json](complete_execution/visual_review.json), with decoded-prefix identity and direct chronological review of every remaining frame. The same-view comparison shows reduced descent wobble and gentler touchdown, deliberate support-gated opening, clean separation and stable recovery. The residual free-settling rock is visible and quantified above.

[Reproduction instructions](REPRODUCE.md) use the existing environment and the preserved exact source/cache. Four focused continuations and their failures are retained. The accepted baseline and intentionally preserved research are unchanged. The engineering phase made no commit or push; at the user’s subsequent request this verified result is selectively published in one commit after one rerun of the existing accepted verification protocol. No further experiments or controller changes were made. Failed variants, duplicate intermediate rollouts/models, frame atlases and pretrained weights remain local and unstaged. [Provenance and hashes](provenance.json) identify the selected artifacts.

The [publication verification](publication_verification.json) reran the accepted raw-control replay with full video and task audit once. It reproduced all task gates, zero replay errors and invalid impulses, 2.750144 m carry and ten alternating replants. The rollout, model and rerendered video hashes are unchanged from the reviewed refinement.
