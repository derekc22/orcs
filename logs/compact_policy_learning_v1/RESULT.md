# SUCCESS — learned closed-loop compact-hand G1 locomomanipulation

The frozen [learned checkpoint](selected_policy.npz) completes the full ground-to-ground task in **3/3 planned nominal repetitions** and **10/10 valid, independently seeded modest perturbation trials**. An additional small-perturbation set passes **10/10 independent numerical evaluations**. No test seed selected a checkpoint. One additional nominal timing diagnostic also passed with exactly identical controls and states.

The [complete nominal video](complete_execution/rollout.mp4), [exact rollout](complete_execution/rollout.npz), [standalone MuJoCo model](complete_execution/rollout_model.mjb), [accepted task assessment](complete_execution/task_assessment.json), and [independent replay/contact attribution](complete_execution/rollout_verification.json) preserve the original physical task. All ten modest-trial videos and replay/contact traces are saved in `expanded_test_71106001` through `expanded_test_71106010`; each has an accepted `task_assessment.json` with **SUCCESS** and explicit video/grasp review.

## Tested perturbations and every outcome

Each scalar is sampled independently from a bounded uniform distribution. The distributions were frozen before their independent trials. No invalid initial trial was discarded or replaced; all 20 learned and all 20 matched ablation initial states were physically valid.

| Initial-state component | Initial small set | Expanded modest set |
|---|---:|---:|
| Robot root x and y, independently | ±1 mm | ±5 mm |
| Robot root yaw | ±0.2° | ±1° |
| All 29 native robot joint positions | ±0.001 rad | ±0.005 rad |
| Native joint velocities | ±0.005 rad/s | ±0.02 rad/s |
| Root linear velocities, each axis | ±0.002 m/s | ±0.005 m/s |
| Root angular velocities, each axis | ±0.002 rad/s | ±0.005 rad/s |
| Box x and y, independently | ±1 mm | ±5 mm |
| Box yaw | ±0.2° | ±1° |
| Independent seeds | 71105001–71105010 | 71106001–71106010 |
| Learned full numerical passes | **10/10** | **10/10** |
| Full video review and unchanged accepted task audit | Supplemental numerical set | **10/10 SUCCESS** |
| Matched frozen expert replay with learned feedback disabled | **0/10** | **0/10** |

Root/box height and roll/pitch, the box's initial velocity, and the open hand initial states are unchanged. Yaw preserves the box's vertical support geometry. Initial contacts and exact sampled qpos/qvel changes are recorded in every `generation.json`. Native initial joint positions also remain inside their limits; the minimum margin in the modest set is 0.17476 rad.

The first successful small set justified one wider evaluation of the **same unchanged checkpoint**. The wider distribution was declared before its three development seeds and ten new independent seeds. No gains, costs, correction limits, training data or checkpoint changed for expansion. Development seeds are disjoint from both independent sets.

[ROBUSTNESS.md](ROBUSTNESS.md) lists every independent outcome, walking displacement, step count, grip force and paired failure classification. [all_trials.csv](all_trials.csv) and [all_trials.json](all_trials.json) include all development, nominal, held-out and ablation results, actual perturbations, placement/release/stability gates and replay errors. The small set's complete physical replay is verified; only the modest set is promoted through the full per-trial video/accepted-audit protocol.

## What is learned

[Inference](../../scripts/compact_policy_learning_v1/controller.py) applies a **learned 98-state → 37-actuator residual at 50 Hz**, using current robot root, joints, hands, box pose and velocities relative to the expert. Exact expert feedforward advances at 200 Hz. The residual is bounded and added only to native actuator commands. Native position servos, finite effort limits, the reference and the phase clock remain engineered.

The policy is learned through empirical local dynamics identification and finite-horizon optimal feedback optimization. [Training](../../scripts/compact_policy_learning_v1/train.py) collects 256 orthogonal randomized physical transitions at each of 2,408 reference intervals, fits local state/action transition matrices by least squares, then optimizes the state-to-action gains by Bellman/Riccati recursion. There are **616,448 training transitions**, one full fit, and **zero checkpoint/gain searches or robustness retraining runs**. Training takes 106.98 s with eight CPU workers in the existing environment. No GPU or dependency installation was used.

This learns the phase-specific coupling between balance, feet, hands and box from actual simulator responses. It does not assign synthetic fixed-PD action labels, the failed historical approach. The expert references are honestly stored as reference arrays. All runtime state changes after initialization arise from continuous MuJoCo stepping; offline saved solver states are used only to generate local training probes.

The learned outputs matter. With identical perturbed initial states and exactly the committed expert control prefix, disabling learned feedback causes all 20 matched independent trials to lose balance or make non-foot terrain contact during pickup, before qualifying walking. This comparator is **frozen committed control replay with native servos**, not a regeneration of the historical expert's full online stage controllers. [Paired integrity checks](paired_ablation_integrity.json) verify equal initial states and exact reference controls. This establishes the effect of learned feedback without claiming an untested comparison against every engineered online controller.

[Fixed-phase observation sensitivity](feedback_sensitivity.json) separately demonstrates current-object dependence: at 18 s, changing only observed box y by 1 mm changes actuator targets by up to 0.001657 rad; changing root x by 1 mm changes targets by up to 0.004853 rad. The policy continues to use observations through carry and placement. Recorded online residuals reach the 0.15 rad native-joint correction bound on perturbed trials. No object force or state is commanded directly.

[Training configuration](training_configuration.json), [full methodology](METHODOLOGY.md), [training diagnostics](training.json), exact reference/solver dataset `reference.npz`, fitted dynamics `dynamics.npz`, the first checkpoint `policy.npz`, and selected checkpoint `selected_policy.npz` are preserved. Local prediction is imperfect at nonsmooth contacts: median held-out normalized prediction error is 0.131 and the worst sampled initial-contact error is 0.937. Successful closed-loop rollouts, rather than regression accuracy alone, establish the tested capability.

## Physical execution and expert comparison

| Measurement | Committed engineered nominal | Learned nominal / modest trials |
|---|---:|---:|
| Complete physical duration | 48.145 s | 48.145 s each |
| Maximum object COM lift | 0.787750 m | 0.787750 m nominal |
| Retained robot-root walking displacement | 2.750144 m | 2.750144 m nominal; **2.749127–2.750994 m** modest |
| Consecutive alternating qualifying replants | 10 | **10 in every trial** |
| Nominal swing clearance | 52.4–91.1 mm | 52.4–91.1 mm |
| Nominal minimum walking COM lift | 0.569751 m | 0.569751 m |
| Minimum walking left/right hand force | 0.597582 / 0.375330 N | At least **0.596691 / 0.375230 N** across modest trials |
| Placement object angular-acceleration RMS | 7.344258 rad/s² | 7.344258 rad/s² nominal |
| Walking object angular-acceleration RMS | 26.060155 rad/s² | 26.060155 rad/s² nominal |
| Final 1 s maximum root linear/angular speed | 0.002863 m/s / 0.010467 rad/s | Same to reported precision |
| Final 1 s maximum object linear/angular speed | 0.00009904 m/s / 0.00063735 rad/s | Same to reported precision |
| Forbidden object / robot self / invalid terrain impulse | 0 / 0 / 0 N·s | **0 / 0 / 0 in every learned trial** |
| Independent exact qpos / qvel replay error | 0 / 0 | **0 / 0 in every trial** |

The counted walking interval is 17.245–23.605 s. Root displacement is measured from actual robot positions during terrain-free, physically retained carry; object displacement is reported separately and never substituted for walking. Footsteps require support loss, at least 20 mm swing clearance, at least 120 mm same-foot relocation, and sustained foot loading before/after replant. All learned trials retain hand support throughout the entire pre-release airborne interval, including pickup and braking, not only the walking window.

Nominal sustained terrain support above 70% of box weight begins at 39.145 s; deliberate opening begins at 40.140 s with 0.588592 N terrain load against 0.490500 N box weight. Last hand contact is at 40.440 s. Ground support remains continuous thereafter; recovery begins at 44.145 s and both robot and released box settle. Every modest trial passes the same support-before-opening, physical-opening, clean-withdrawal and final-stability gates.

Grasp motion remains bounded and compliant. Nominal palm-relative rocking is 9.094° / 7.015°, with material-point tangential excursions 8.015 / 5.029 mm. Across modest trials the largest excursions are 8.964 / 5.441 mm. Zero slip is not claimed. [Nominal grasp audit](complete_execution/grasp_audit.json), per-trial force/geometric traces and explicit visual review support retention acceptance.

[expert_comparison.json](expert_comparison.json) retains the numerical comparison, including raw motion diagnostics. Nominal walking and refined placement smoothness are preserved to reported precision; this phase optimizes feedback capability rather than nominal smoothness.

## Independent validation, preservation and scope

The existing accepted raw-control verifier and task audit are reused. [replay.py](../../scripts/compact_policy_learning_v1/replay.py) only eagerly caches NPZ arrays to avoid repeatedly decompressing whole trajectories; it leaves the verifier's physics, contact attribution, rendering and metrics unchanged. The legacy verifier still writes its historical unconditional `success:false` and torso-tilt field. The unchanged accepted `task_assessment.json` evaluates the requested physical task and reports **SUCCESS** for the nominal and all ten modest trials.

All 2,407 nominal video frames were directly reviewed in 17 full-view chronological sheets, plus its larger montage. For **each** modest trial, all 2,407 frames were directly reviewed in five chronological task-view panels, supplemented by its larger full-view montage and independent 200 Hz contact/grasp traces. The panels use a fixed robot/box/nearby-terrain crop and 64×80 thumbnails with no temporal subsampling. These thumbnails check complete motion and gross retention; fine contact and sliding claims rely on force/material-point measurements. All original full-resolution videos remain unmodified. Each trial's `visual_review.json` and panel manifest records the coverage and video hash. Exploratory pixel-identity/difference diagnostics are preserved but do not substitute for the completed direct review.

Every physical model is byte-identical to the protected baseline: original box dimensions/mass/contact properties, compact hand morphology, effort limits, MuJoCo contacts and physics settings are unchanged. Model integrity audits confirm zero equality constraints, mocap supports, object actuators, applied external forces and disabled physics flags. The nominal model copy is standalone. The expert is preserved and remains available.

This establishes a **local, reference-conditioned simulation capability** in the documented xy/yaw/joint/velocity envelope. Current robot/object state is privileged simulator observation. Roll/pitch/height perturbations, new goals, changed physical parameters, runtime disturbances and hardware deployment are outside this test. No broader robustness is claimed.

The research phase stopped after meeting the target with one fit and one evidence-supported range expansion. [Reproduction instructions](REPRODUCE.md) use the existing `.venv`. The [research provenance snapshot](provenance.json) records protected commit `f65e57e648db0a0d8728c8650b42a597797da9b5`, remote-main confirmation before work, unchanged tracked files and index, and no commit or push during research. At the user's subsequent explicit request, the verified success is selectively published with one rerun of the accepted verification and no retraining or additional experiments. [Publication scope](PUBLICATION.md) distinguishes committed artifacts from preserved local caches and recordings. All historical untracked experiments remain preserved.
