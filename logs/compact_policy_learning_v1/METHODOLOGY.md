# Bounded compact-hand policy-learning phase

Protected accepted commit: `f65e57e648db0a0d8728c8650b42a597797da9b5`, confirmed against `git ls-remote origin refs/heads/main` before implementation. The research phase did not authorize a commit or push. The user's subsequent instruction authorizes selective publication of the verified success; see [publication scope](PUBLICATION.md). All new work is confined to `scripts/compact_policy_learning_v1` and `logs/compact_policy_learning_v1`.

## Selection rationale and gates

The earlier `logs/engineering_policy/FINAL_ROBUSTNESS_RESULT.md` identifies coupled support/contact divergence: stronger generic locomotion feedback preserved robot survival but destroyed payload placement. Its spectral action interpolation and synthetic fixed PD labels did not learn this coupling. Settling preludes failed the nominal gate. Those strategies are not repeated here.

The new reference has a different compact physical grasp and an exact full ground-to-ground demonstration. The existing environment runs a full 48.145 s CPU simulation in seconds; no functioning NVIDIA GPU is available. The repository's full neural imitation/RL paths depend on a different action/observation adapter or GPU environment. Reusing the exact native MuJoCo and accepted reference/replay interfaces avoids changing the physical system or spending a large training budget before establishing feasibility.

Selected approach: empirical local system identification followed by finite-horizon optimal feedback learning. It learns a phase-specific state-to-actuator residual, including coupling between floating robot base, native joints, eight hand DOFs and the floating box. This is model-based policy optimization, not a neural actor or a learned replacement for the whole expert.

Gate 1 pilot: reconstruct the full expert exactly, restore saved solver/integration history exactly, and fit/predict real perturbed simulator transitions at representative phases. `pilot.json` records the result, including poor local approximation at some nonsmooth contact/settling samples. A single full fit is justified by good prediction in most grasped carry/descent samples and low CPU cost. Local validity is not assumed to imply rollout robustness.

Gate 2: run the full online learned controller from the original initial state, then independently replay its raw controls. Nominal success requires the unchanged physical task gates, not just imitation error. No expensive robustness learning before this gate.

Gate 3: use only the development seeds in `evaluation_protocol.json` for diagnosing recurring physical failures. At most one targeted dynamics/cost update is initially justified, based on a common failure with an actionable cause. Preserve the first fitted policy regardless of outcome. Further work requires new evidence of progress; stop at a plateau.

Gate 4: freeze the selected checkpoint before evaluating the ten independent seeds. Record every trial, invalid initial state and partial failure. Never select a checkpoint against those outcomes. Compare the engineered feedforward execution on the same initial states. A failing robustness target is an evidenced limitation, not success.

## Learned and engineered components

Engineered: exact accepted expert state/action reference, fixed phase clock, 200 Hz feedforward commands, unchanged position servos and native contact dynamics, actuator clipping, bounded correction magnitudes, task gates. Reference conditioning is permitted by the task. The feedforward actions are honestly stored as a trajectory rather than relabeled as neural outputs.

Learned: a 98-dimensional tangent state (49 generalized position errors including quaternion tangent errors, plus 49 velocity errors) to 37-actuator residual matrix at each 20 ms reference interval. The matrix is optimized using empirically regressed physical transition models. Current root, joint, hand and box observations enter inference every 20 ms. Learned residuals are held for four physical steps while feedforward advances at 200 Hz.

For each reference interval, 256 orthogonal randomized joint/base/box/action perturbations produce actual MuJoCo transitions over the next four 5 ms steps. An ordinary least-squares regression fits local A and B matrices. The directions come from randomized columns of a Hadamard design, with scales stored in `train.py`. They identify local dynamics; they are distinct from the frozen initial-state distribution. Twelve independent Gaussian probes at regularly sampled intervals measure prediction error. The final shortened interval is one physical step.

Finite-horizon Bellman/Riccati recursion minimizes quadratic tracking plus action-change cost on these fitted dynamics, yielding the residual gain matrices K. Native actuator effort saturation is already present in every training transition. Deployment applies only bounded actuator commands, never generalized/external forces, object state edits, supports, equality constraints or hidden attachments. Saved integration states are used only for offline local training probes; evaluation always starts at the original or explicitly perturbed ground-supported initial state and advances continuously.

`reference.npz` preserves the complete exact expert and solver states for reproducing local data. `dynamics.npz` preserves the fitted transition dataset. `training.json` records seed, sample count, validation, model/reference hashes and timing. The checkpoint preserves K, references, correction bounds and optimization costs in double precision.

## Perturbations and independent evaluation

`evaluation_protocol.json` is frozen before any learned checkpoint evaluation. Each scalar is sampled independently from a bounded uniform distribution. Robot xy: ±1 mm; robot yaw: ±0.2 degrees; all 29 native robot joints: ±0.001 rad; native joint velocities: ±0.005 rad/s; root linear/angular velocities: ±0.002 m/s and rad/s. Box xy: ±1 mm; box yaw: ±0.2 degrees. Root/box height and roll/pitch, box initial velocity, and open hand initial states are unchanged. Yaw changes preserve the box's vertical support geometry. These are deliberately small perturbations; no larger robustness is implied.

Initial physical validity requires terrain-supported box and no meaningful forbidden object, self, or non-foot terrain contacts under the unchanged model. Every actual qpos/qvel delta and initial contact is recorded. No state adjustment or silent seed replacement is allowed. Development seeds are 1105101–1105103. Independent test seeds are 71105001–71105010.

Independent validation uses the existing `corner_hands_v1/verify_physics.py` for exact raw-control replay, force/contact attribution, complete video and walking measurements, plus the existing task measurement functions. The new assessment wrapper can report partial failures without assuming that grasp/placement events exist. Its numeric result does not imply visual acceptance. Complete video and explicit grasp review are separately required for a claimed successful controller.
