# Per-trial robustness results

All trials and actual perturbations are preserved in [all_trials.csv](all_trials.csv) and [all_trials.json](all_trials.json). Every listed evaluation independently replays raw controls. No invalid initial trial was discarded.

## small independent set

| Seed | Learned result | Retained root walk (m) | Alternating steps | Min L/R grip (N) | Place / release / stable | Frozen-replay ablation |
|---|---|---:|---:|---|---|---|
| 71105001 | PASS | 2.750451 | 10 | 0.5978 / 0.3754 | pass / pass / pass | fall at 1.475s |
| 71105002 | PASS | 2.750104 | 10 | 0.5976 / 0.3753 | pass / pass / pass | fall at 2.250s |
| 71105003 | PASS | 2.750425 | 10 | 0.5972 / 0.3752 | pass / pass / pass | fall at 1.695s |
| 71105004 | PASS | 2.750269 | 10 | 0.6016 / 0.3761 | pass / pass / pass | invalid_robot_terrain_contact at 1.705s |
| 71105005 | PASS | 2.750132 | 10 | 0.6016 / 0.3761 | pass / pass / pass | fall at 2.160s |
| 71105006 | PASS | 2.750085 | 10 | 0.5975 / 0.3753 | pass / pass / pass | fall at 1.895s |
| 71105007 | PASS | 2.750308 | 10 | 0.5973 / 0.3753 | pass / pass / pass | invalid_robot_terrain_contact at 1.980s |
| 71105008 | PASS | 2.750340 | 10 | 0.5974 / 0.3753 | pass / pass / pass | fall at 1.960s |
| 71105009 | PASS | 2.750307 | 10 | 0.6016 / 0.3761 | pass / pass / pass | fall at 2.730s |
| 71105010 | PASS | 2.750654 | 10 | 0.5979 / 0.3754 | pass / pass / pass | invalid_robot_terrain_contact at 1.650s |

All learned trials have zero forbidden-object, self and invalid-terrain impulses, exact qpos/qvel replay errors 0/0, full airborne grasp retention, support before opening, clean separation and final stability. Full details and seed-specific initial contacts are in each trial directory.

## modest independent set

| Seed | Learned result | Retained root walk (m) | Alternating steps | Min L/R grip (N) | Place / release / stable | Frozen-replay ablation |
|---|---|---:|---:|---|---|---|
| 71106001 | PASS | 2.749743 | 10 | 0.5967 / 0.3752 | pass / pass / pass | invalid_robot_terrain_contact at 1.465s |
| 71106002 | PASS | 2.750361 | 10 | 0.5973 / 0.3754 | pass / pass / pass | invalid_robot_terrain_contact at 1.515s |
| 71106003 | PASS | 2.750994 | 10 | 0.5975 / 0.3753 | pass / pass / pass | fall at 1.510s |
| 71106004 | PASS | 2.750498 | 10 | 0.5972 / 0.3753 | pass / pass / pass | invalid_robot_terrain_contact at 1.325s |
| 71106005 | PASS | 2.750286 | 10 | 0.5974 / 0.3753 | pass / pass / pass | invalid_robot_terrain_contact at 1.515s |
| 71106006 | PASS | 2.750454 | 10 | 0.6016 / 0.3761 | pass / pass / pass | fall at 1.570s |
| 71106007 | PASS | 2.750555 | 10 | 0.5973 / 0.3753 | pass / pass / pass | robot_self_contact at 1.730s |
| 71106008 | PASS | 2.749127 | 10 | 0.5978 / 0.3753 | pass / pass / pass | fall at 1.460s |
| 71106009 | PASS | 2.750369 | 10 | 0.6015 / 0.3760 | pass / pass / pass | invalid_robot_terrain_contact at 1.460s |
| 71106010 | PASS | 2.750030 | 10 | 0.6014 / 0.3760 | pass / pass / pass | invalid_robot_terrain_contact at 1.425s |

All learned trials have zero forbidden-object, self and invalid-terrain impulses, exact qpos/qvel replay errors 0/0, full airborne grasp retention, support before opening, clean separation and final stability. Full details and seed-specific initial contacts are in each trial directory.

## Development and nominal runs

| Trial | Result | Root walk (m) | Steps |
|---|---|---:|---:|
| development_v1_1105101 | PASS | 2.750292 | 10 |
| development_v1_1105102 | PASS | 2.749718 | 10 |
| development_v1_1105103 | PASS | 2.750149 | 10 |
| expanded_development_1105201 | PASS | 2.750310 | 10 |
| expanded_development_1105202 | PASS | 2.750378 | 10 |
| expanded_development_1105203 | PASS | 2.750778 | 10 |
| nominal_repeat_2 | PASS | 2.750144 | 10 |
| nominal_repeat_3 | PASS | 2.750144 | 10 |
| nominal_v1_cached | PASS | 2.750144 | 10 |

The additional frozen-replay development diagnostic fails during pickup; its outcome is included in the CSV/JSON. An extra nominal timing diagnostic also completed successfully with identical states and controls, beyond the three planned nominal repetitions. A provisional interruption note in `nominal_v1/ABORTED.json` was corrected after its complete artifact appeared. Eager array caching avoids its repeated NPZ decompression overhead.

Frozen-replay failures are pickup balance loss/fall or non-foot terrain contact, before qualifying walking. They are feedback-ablation failures, not mechanical infeasibility or unsuccessful learned checkpoints. No learned trial failed within either tested envelope.
