"""Fail-closed evidence manifest for the two frozen diffusion ORCS executions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from evidence import ROOT,WINDOWS


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def finalize(out):
    baseline=ROOT/'logs/compact_orcs_v1/complete_execution'
    old=json.loads((baseline/'verification_manifest.json').read_text())
    sources={p:sha(ROOT/p)==h for p,h in old['source_sha256'].items()}
    allowed={'src/orcs/tasks/uolm/__init__.py','scripts/compact_orcs_v1/run.py'}
    artifacts={p:sha(ROOT/p)==h for p,h in old['artifact_sha256'].items()}
    recheck=json.loads((out/'baseline_recheck/execution.json').read_text())
    exact_keys=['exact_qpos_error','exact_qvel_error','exact_control_error','exact_contact_trace_error','exact_locomotion_trace_error']
    generated=json.loads((out/'proposals/provenance.json').read_text())
    calibrated=json.loads((out/'calibration/provenance.json').read_text())
    conditions=[np.load(out/f'proposals/sample_{seed}.npz')['condition'] for seed in [42,43]]
    conditions.append(np.load(out/'calibration/sample_0.npz')['condition'])
    inference_recheck={}
    for seed in [42,43,0]:
        first=np.load(out/('calibration' if seed==0 else 'proposals')/f'sample_{seed}.npz')
        second=np.load(out/'proposals_recheck'/f'sample_{seed}.npz')
        inference_recheck[str(seed)]={k:bool(np.array_equal(first[k],second[k])) for k in first.files}
    causal=json.loads((out/'causality.json').read_text())
    projection_causal=json.loads((out/'projection_causality.json').read_text())
    with np.load(ROOT/'logs/compact_policy_learning_v1/selected_policy.npz') as a:expert_u=a['uref'].copy()
    details={};shared_ff=True;integrity=True
    for seed in [42,43]:
        p=out/f'execution_{seed}';a=np.load(p/'rollout.npz');ref=np.load(out/f'feasible_{seed}.npz')
        target=np.load(out/f'spread_target_{seed}.npz');projected=np.load(out/f'spread_projection_{seed}/rollout.npz')
        execution=json.loads((p/'execution.json').read_text());assessment=json.loads((p/'task_assessment.json').read_text())
        replay=json.loads((out/f'replay_{seed}/execution.json').read_text())
        contacts=np.load(p/'contact_attribution.npz');replay_contacts=np.load(out/f'replay_{seed}/contact_attribution.npz')
        exact_contacts=all(np.array_equal(contacts[k],replay_contacts[k]) for k in contacts.files)
        shared_ff &= np.array_equal(target['uref'],expert_u)
        fixed=json.loads((p/'baseline_preservation.json').read_text())
        model=json.loads((p/'model_equivalence.json').read_text())
        integrity &= bool(execution['reset_state_writes']==1 and execution['physics_step_counter']==9629
            and execution['episode_step_counter']==9629 and execution['metric_substeps']==9629
            and not execution['training_performed'] and all(v['unchanged'] for v in fixed.values())
            and model['pass_model_equivalence'] and execution['command_class']=='DiffusionRefinedCommand')
        details[str(seed)]=dict(physical_assessment=assessment,execution=execution,
            replay_errors={k:replay[k] for k in exact_keys},
            full_contact_attribution_exact=exact_contacts,attributed_contact_rows=len(contacts['contacts']),
            final_vs_physical_projection={k:float(abs(a[k]-projected[k]).max()) for k in ['qpos','qvel','ctrl_physics']},
            reference_tracking_error={k:float(abs(a[k][:-1]-ref[r][1:]).max()) for k,r in [('qpos','qref'),('qvel','vref')]},
            refined_reference_sha256=sha(out/f'feasible_{seed}.npz'),video=str(p/'rollout.mp4'))
    gates=dict(two_complete_physical_successes=all(d['physical_assessment']['status']=='SUCCESS' and all(d['physical_assessment']['numeric_gates'].values()) for d in details.values()),
        two_exact_independent_replays=all(all(v==0 for v in d['replay_errors'].values()) for d in details.values()),
        full_contact_attribution_replays=all(d['full_contact_attribution_exact'] for d in details.values()),
        diffusion_survives_physical_refinement=projection_causal['pass_causality'],
        diffusion_changes_final_reference_commands_and_motion=causal['pass_causality'],
        identical_diffusion_conditions=all(np.array_equal(c,conditions[0]) for c in conditions),
        identical_diffusion_sampler=generated['sampling']['steps']==calibrated['sampling']['steps']==50,
        frozen_checkpoint=sha(Path(generated['checkpoint']))==generated['checkpoint_sha256']==calibrated['checkpoint_sha256'],
        exact_frozen_inference_reproduction=all(v for sample in inference_recheck.values() for v in sample.values()),
        physics_projection_uses_shared_expert_feedforward=bool(shared_ff),
        physical_control_and_model_integrity=bool(integrity),
        accepted_goal1_artifacts_unchanged=all(artifacts.values()),
        accepted_goal1_core_code_unchanged=all(v for k,v in sources.items() if k not in allowed),
        accepted_goal1_exact_recheck=bool(recheck['steps']==9629 and all(recheck[k]==0 for k in exact_keys)))
    paths=['src/orcs/tasks/uolm/__init__.py','src/orcs/tasks/uolm/compact/diffusion.py','scripts/compact_orcs_v1/run.py']
    paths += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'scripts/compact_diffusion_v1').glob('*.py'))]
    artifact_paths=[p for p in out.rglob('*') if p.is_file() and p.suffix in ['.json','.npz','.mjb','.mp4'] and p.name not in ['verification.json','verification_manifest.json']]
    result=dict(status='SUCCESS' if all(gates.values()) else 'FAILURE',gates=gates,task='Orcs-Uolm-Compact-Diffusion-Feedback',
        diffusion=generated,calibration_seed=0,refinement=dict(mean_gain=.15,sample_spread_gain=4.,max_retime_s=.04,
        method='Phase alignment, bounded diffusion-derived innovation and differential IK, then deterministic frozen-feedback projection through actual ORCS physics; compiled feasible reference executed again through ORCS.'),
        causal_windows_physics_steps=WINDOWS,causality=causal,executions=details,
        goal1_artifact_preservation=artifacts,goal1_source_preservation=sources,
        baseline_recheck_errors={k:recheck[k] for k in exact_keys},
        frozen_inference_recheck=inference_recheck,
        limitations=['Validated for this fixed task, contact sequence and accepted 50 g original-sized box only.',
            'The frozen checkpoint rejects direct compact-task conditioning; matched native largebox proposals are retargeted to the fixed physical initial state and goal.',
            'The expert supplies contact topology, gripper schedule, support and nominal feedforward. Diffusion supplies bounded motion variation in all three phases, not a new contact planner.'],
        files_changed=paths,source_sha256={p:sha(ROOT/p) for p in paths},
        reproduction_command='.venv/bin/python scripts/compact_diffusion_v1/reproduce.py --out logs/compact_diffusion_v1_reproduction')
    (out/'verification.json').write_text(json.dumps(result,indent=2))
    (out/'verification_manifest.json').write_text(json.dumps(dict(source_sha256=result['source_sha256'],artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in artifact_paths}),indent=2))
    print(json.dumps(dict(status=result['status'],gates=gates),indent=2))
    if not all(gates.values()):raise RuntimeError('Goal 2 verification failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();finalize(a.out.resolve())
