"""Frozen checkpoint proposals and explicit compact-task coordinate boundary."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time

import mjlab
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from orcs.tasks.uolm.diffusion_runtime import BatchedDiffusionGenerator, DiffusionGeneratorCfg
from orcs.tasks.uolm.compact.model import source_model, CHECKPOINT as CONTROLLER

ROOT=Path(__file__).resolve().parents[2]
DIFFUSION=ROOT.parent/'g1-diffusion'
CHECKPOINT=DIFFUSION/'logs/object_goal_single_stage_init_goal_hf_bps_e10000_b16_lr5e-06_ts1000_w300_s10_transformer_2026Aug05_21-21-42/checkpoints/object_goal_single_stage_init_goal_epoch_000999.pt'
SOURCE=DIFFUSION/'data/hf_bps_preprocessed/omomo_sub3_largebox_003_sample1.pkl'


def columns6(R):
    return np.concatenate([R[..., :,0],R[..., :,1]],axis=-1)


def matrix6(x):
    a=x[...,:3];b=x[...,3:];a=a/np.linalg.norm(a,axis=-1,keepdims=True)
    b=b-(a*b).sum(-1,keepdims=True)*a;b=b/np.linalg.norm(b,axis=-1,keepdims=True)
    return np.stack([a,b,np.cross(a,b)],axis=-1)


def decode(raw):
    """Historical object component correction exactly once; root is ordinary."""
    root=Rotation.from_matrix(matrix6(raw[:,3:9])).as_quat(scalar_first=True)
    encoded=Rotation.from_matrix(matrix6(raw[:,41:47])).as_quat(scalar_first=True)
    physical=np.roll(encoded,-1,axis=-1)
    return root,physical


def centroid_offset():
    m=source_model();g=m.geom('object/largebox_visual').id;mesh=int(m.geom_dataid[g])
    vertices=m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]]
    return m.geom_pos[g]+Rotation.from_quat(m.geom_quat[g],scalar_first=True).apply(vertices.mean(0))


def generate(out,inference_steps=50,seeds=(42,43)):
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    generator=BatchedDiffusionGenerator(DiffusionGeneratorCfg(source_path=str(DIFFUSION),checkpoint_path=str(CHECKPOINT),
        sampler='ddim',num_inference_steps=inference_steps,eta=0.,horizon=300,precision='fp32'),device='cpu')
    with open(SOURCE,'rb') as f:data=pickle.load(f)
    module=__import__('scripts.sample_object_goal_single_stage_init_goal_hf_bps',fromlist=['condition_from_source'])
    g,demonstration=module.condition_from_source(data,300)
    normalized=generator.pipeline.normalize_global_cond(g)[0].cpu().numpy()
    m=source_model();c=np.load(CONTROLLER);q0=c['qref'][0];qgoal=c['qref'][-1]
    native=[m.jnt_qposadr[j] for j in range(m.njnt) if m.jnt_type[j]==3 and not m.joint(j).name.startswith('hand_')]
    assert len(native)==29
    offset=centroid_offset()
    def object_frame(q):
        physical=Rotation.from_quat(q[47:51],scalar_first=True)
        centroid=q[44:47]+physical.apply(offset)
        encoded=Rotation.from_quat(np.roll(q[47:51],1),scalar_first=True)
        return np.r_[centroid,columns6(encoded.as_matrix())]
    task_g=np.r_[object_frame(qgoal),q0[:3],columns6(Rotation.from_quat(q0[3:7],scalar_first=True).as_matrix()),q0[native],object_frame(q0)]
    task_z=generator.pipeline.normalize_global_cond(task_g)[0].cpu().numpy()
    # The checkpoint's terminal orientation has nearly-zero training variance.
    # Use a matched native proposal condition, then enforce the actual compact
    # initial state and goal at the refinement boundary. Do not extrapolate a
    # condition rejected by the checkpoint's own distribution guard.
    meta=dict(checkpoint=str(CHECKPOINT),checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        model_type='object_goal_single_stage_init_goal',state_dim=47,condition_dim=56,horizon=300,fps=50,
        sampling=dict(sampler='ddim',steps=inference_steps,eta=0.,precision='fp32',device='cpu',seeds=list(seeds)),
        conditioning_source=str(SOURCE),normalized_condition_absmax=float(abs(normalized).max()),
        actual_task_condition_absmax=float(abs(task_z).max()),
        conditioning_contract='Matched native largebox motion prior; identical condition for both seeds. Actual compact initial state, final task goal and contacts are imposed by refinement, not claimed as diffusion outputs.',
        native_joint_names=[m.joint(j).name for j in range(m.njnt) if m.jnt_type[j]==3 and not m.joint(j).name.startswith('hand_')],
        object_centroid_offset_body=offset.tolist(),historical_object_correction_count=1,
        causal_requirements=dict(per_phase_object_or_root_rms_m=.0005,per_phase_native_joint_rms_rad=.002,
            criterion='Both sample-vs-baseline and sample-vs-sample must show >=0.5 mm RMS object/root motion OR >=2 mrad RMS native posture in every phase; no success from numerical differences alone.'))
    np.savez_compressed(out/'conditioning.npz',condition=g,normalized_condition=normalized,task_condition=task_g,task_condition_z=task_z,demonstration=demonstration)
    for seed in seeds:
        torch.manual_seed(seed);start=time.time()
        raw=generator.generate(torch.as_tensor(g).unsqueeze(0))[0].cpu().numpy()
        assert raw.shape==(300,47) and np.isfinite(raw).all()
        root,physical=decode(raw)
        body=raw[:,38:41]-Rotation.from_quat(physical,scalar_first=True).apply(offset)
        np.savez_compressed(out/f'sample_{seed}.npz',raw=raw,root_quaternion=root,object_quaternion=physical,object_body_position=body,condition=g)
        print('sample',seed,'seconds',time.time()-start,'root_span',np.ptp(raw[:,:3],axis=0),'object_span',np.ptp(body,axis=0),flush=True)
    (out/'provenance.json').write_text(json.dumps(meta,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--inference-steps',type=int,default=50);p.add_argument('--seeds',default='42,43');a=p.parse_args();generate(a.out,a.inference_steps,tuple(map(int,a.seeds.split(','))))
