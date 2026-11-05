"""Task measurements from the independent raw-control physics replay.

Run verify_physics.py first. This script never advances or modifies a real
rollout: FK/contact reconstructions below are diagnostic data only.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/compact_integration_v1'))
from metrics import measure,runs

def audit(p):
    a=np.load(p/'rollout.npz');m=mujoco.MjModel.from_binary_path(str(p/'rollout_model.mjb'))
    v=json.loads((p/'rollout_verification.json').read_text());r=measure(p)
    t=np.load(p/'rollout_verification_trace.npz')['trace'];l=np.load(p/'rollout_locomotion_trace.npz')['trace']
    s,e=int(a['walking_start_physics_step']),int(a['walking_end_physics_step']);op=round(float(a['phase_opening_frame'])*4)
    ob=m.body('object/object').id;aq=m.jnt_qposadr[m.actuator_trnid[:,0]]
    old=mujoco.MjModel.from_binary_path(str(ROOT/'logs/sustained_carry_v1/complete2_model.mjb'));bo=old.body('object/object').id
    gs=np.flatnonzero(m.geom_bodyid==ob);ogs=np.flatnonzero(old.geom_bodyid==bo)
    errors={f:float(np.max(abs(getattr(m,f)[gs]-getattr(old,f)[ogs]))) for f in ['geom_pos','geom_quat','geom_size','geom_condim','geom_friction','geom_solref','geom_solimp','geom_margin','geom_gap','geom_contype','geom_conaffinity']}
    errors.update({f:float(np.max(abs(getattr(m,f)[ob]-getattr(old,f)[bo]))) for f in ['body_mass','body_inertia','body_ipos','body_iquat']})
    errors['mesh_vertices']=float(np.max(abs(m.mesh_vert-old.mesh_vert)))
    src=mujoco.MjModel.from_binary_path(str(ROOT/'logs/corner_hands_v1/pickup3_model.mjb'))
    same={f:bool(np.array_equal(getattr(m,f),getattr(src,f))) for f in ['body_mass','body_inertia','body_pos','body_quat','body_parentid','geom_bodyid','geom_type','geom_pos','geom_quat','geom_size','geom_friction','geom_solref','geom_solimp','geom_contype','geom_conaffinity','jnt_type','jnt_bodyid','jnt_pos','jnt_axis','jnt_range','actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_forcerange']}
    limited=m.actuator_forcelimited.astype(bool)
    integrity=dict(object_errors_vs_accepted=errors,physical_fields_identical_corner_pickup=same,
        unchanged_box=all(x==0 for x in errors.values()),unchanged_hands=all(same.values()),
        effort_caps_respected=bool(np.all(np.array(v['maximum_abs_actuator_force'])[limited]<=np.max(abs(m.actuator_forcerange),axis=1)[limited]+1e-9)),
        equalities=int(m.neq),disabled_flags=int(m.opt.disableflags),external_force_max=v['maximum_applied_external_force'],
        no_object_actuator=bool(not np.any(m.jnt_bodyid[m.actuator_trnid[:,0]]==ob)))
    (p/'physical_integrity.json').write_text(json.dumps(integrity,indent=2))
    # Follow a fixed box material point at the first loaded palm contact.
    # COM displacement alone also includes rocking about that contact.
    d=mujoco.MjData(m);d.qpos[:]=a['qpos'][s];d.qvel[:]=a['qvel'][s];d.ctrl[:]=a['ctrl_physics'][s];mujoco.mj_forward(m,d)
    plates=[m.body('hand_'+side+'_plate').id for side in ['left','right']];anchors=[]
    for i,side in enumerate(['left','right']):
        ps=[];weights=[]
        for k,c in enumerate(d.contact):
            if not any(m.geom(int(g)).name.startswith('hand_'+side+'_pad') for g in c.geom) or not any(m.geom_bodyid[g]==ob for g in c.geom):continue
            f=np.zeros(6);mujoco.mj_contactForce(m,d,k,f);w=np.linalg.norm(f[:3])
            if w>.001:ps.append(c.pos.copy());weights.append(w)
        b=plates[i];anchors.append(d.xmat[b].reshape(3,3).T@(np.average(ps,axis=0,weights=weights)-d.xpos[b]))
    rows=[]
    for q in a['qpos'][s:e]:
        d.qpos[:]=q;mujoco.mj_kinematics(m,d);row=[]
        for b in plates:
            H=d.xmat[b].reshape(3,3);row.extend(H.T@(d.xpos[ob]-d.xpos[b]));row.extend(Rotation.from_matrix(H.T@d.xmat[ob].reshape(3,3)).as_quat())
        rows.append(row)
    x=np.array(rows);grasp={}
    for i,side in enumerate(['left','right']):
        T=x[:,7*i:7*i+3];R=Rotation.from_quat(x[:,7*i+3:7*i+7]);P=anchors[i];O=R[0].inv().apply(P-T[0]);delta=R.apply(O)+T-P
        grasp[side]=dict(max_COM_motion_in_plate_m=float(np.linalg.norm(T-T[0],axis=1).max()),max_relative_rotation_deg=float(np.degrees((R*R[0].inv()).magnitude().max())),final_relative_rotation_deg=float(np.degrees((R[-1]*R[0].inv()).magnitude())),
            initial_contact_anchor_plate_m=P.tolist(),max_material_anchor_tangential_motion_m=float(np.linalg.norm(delta[:,:2],axis=1).max()),
            final_material_anchor_tangential_motion_m=float(np.linalg.norm(delta[-1,:2])),max_material_anchor_normal_motion_m=float(abs(delta[:,2]).max()))
    grasp['method']='Initial contact center weighted by palm-pad forces reconstructed at the first walking state. The same BOX MATERIAL POINT is followed in the actual moving plate frame at every walking state. These geometric diagnostics are not extra real-state edits or supports. They measure bounded relative motion, not total sliding path length.'
    (p/'grasp_audit.json').write_text(json.dumps(grasp,indent=2));np.savez_compressed(p/'grasp_audit_trace.npz',plate_object_pose=x)
    weight=float(m.body_mass[ob]*9.81);ground_loaded=(t[:,0]>0)&(t[:,42]>.7*weight)
    terrain_starts=[b for b,c in runs(ground_loaded) if c-b>=30 and b>e]
    transfer=min(terrain_starts);last_ee=int(np.flatnonzero(t[:,43]>0)[-1]);rec=int(a['recovery_start_physics_step']);final=slice(-200,None)
    release=dict(terrain_70percent_support_start_s=transfer*.005,opening_command_start_s=op*.005,last_EE_contact_s=last_ee*.005,
        recovery_start_s=rec*.005,ground_force_at_opening_N=float(t[op,42]),box_weight_N=weight,
        terrain_support_before_opening=bool(transfer<op),after_last_hand_contact_ground_continuous=bool(t[last_ee+1:,0].all()),
        final_ground_load_mean_N=float(t[final,42].mean()),final_root_linear_speed_max_1s_mps=float(np.linalg.norm(t[final,36:39],axis=1).max()),
        final_root_angular_speed_max_1s_radps=float(np.linalg.norm(t[final,39:42],axis=1).max()),
        final_hand_joints=dict(zip(['left_press','right_press','left_fork','right_fork','left_thumb','right_thumb'],a['qpos'][-1,aq[[29,30,33,34,35,36]]].tolist())))
    r.update(placement=release,object_COM_horizontal_walk_displacement_m=float(np.linalg.norm(t[e-1,11:13]-t[s,11:13])),
        minimum_walking_hand_force_N=l[s:e,16:18].min(0).tolist(),minimum_walking_COM_lift_m=float(t[s:e,5].min()),
        recorded_physics_controls=len(a['ctrl_physics']),initial_ground_supported=bool(t[0,0]),
        physical_integrity=integrity,grasp=grasp)
    (p/'task_metrics.json').write_text(json.dumps(r,indent=2))
    # Numerical gates consume the independent raw-physics traces. The visual
    # and grasp acceptance decisions remain explicit reviewed evidence, never
    # an automatic assumption for a newly generated directory.
    visual=json.loads((p/'visual_review.json').read_text()) if (p/'visual_review.json').exists() else {}
    footmin=l[-200:,14:16].min(0);handq=release['final_hand_joints']
    air_start=r['airborne_intervals_s'][0][0];air_end=r['airborne_intervals_s'][0][1]
    airborne=(np.arange(len(t))>=round(air_start/.005))&(np.arange(len(t))<round(air_end/.005))
    gates=dict(original_box_and_contact_properties=integrity['unchanged_box'],unchanged_existing_hand_mechanism=integrity['unchanged_hands'],
        initial_ground_support=bool(t[0,0] and t[0,42]>.7*weight),
        terrain_free_COM_lift_over_5cm=bool(r['maximum_COM_lift_m']>.05 and r['minimum_walking_COM_lift_m']>.05),
        actual_airborne_root_walk_over_2m=bool(r['bilateral_retained']['root_net_displacement_m']>=2),
        at_least_five_consecutive_alternating_qualifying_replants=bool(r['bilateral_retained']['longest_consecutive_alternating_replants']>=5),
        no_object_terrain_contact_while_walking=bool(not t[s:e,0].any()),
        continuous_bilateral_walking_support=bool((l[s:e,16:18]>1e-4).all()),
        continuous_EE_support_through_initial_airborne_carry=bool((l[airborne,16:18].sum(1)>1e-4).all()),
        terrain_load_transfer_before_full_hand_opening=bool(release['terrain_support_before_opening'] and release['ground_force_at_opening_N']>.7*weight),
        physical_hand_opening=bool(handq['left_press']<-.02 and handq['right_press']<-.02 and max(handq['left_fork'],handq['right_fork'])<.12 and max(abs(handq['left_thumb']),abs(handq['right_thumb']))<.02),
        clean_withdrawal_and_continuous_ground_support=bool(release['after_last_hand_contact_ground_continuous'] and (t[last_ee+1:,43]==0).all()),
        stable_final_robot=bool(release['final_root_linear_speed_max_1s_mps']<.01 and release['final_root_angular_speed_max_1s_radps']<.03 and (footmin>10).all()),
        stable_final_object=bool(r['final_1s_object_linear_speed_max_mps']<.001 and r['final_1s_object_angular_speed_max_radps']<.003),
        zero_forbidden_object_impulse=v['forbidden_object_impulse_Ns']==0.,
        zero_robot_self_and_invalid_terrain_impulse=bool(r['robot_self_impulse_Ns']==0 and r['invalid_terrain_impulse_Ns']==0),
        finite_actuation_with_preserved_effort_limits=integrity['effort_caps_respected'],
        no_virtual_grasp_or_hidden_force=bool(integrity['equalities']==0 and integrity['disabled_flags']==0 and integrity['external_force_max']==0 and integrity['no_object_actuator'] and m.nmocap==0 and m.ntendon==0),
        independent_exact_raw_control_physics_replay=bool(v['exact_qpos_error']==0 and v['exact_qvel_error']==0 and np.isfinite(a['qpos']).all() and np.isfinite(a['qvel']).all()),
        complete_video_inspection=bool(visual.get('full_sequence_visual_review_complete',False)))
    review=dict(accepted_as_bounded_compliant_retention=bool(visual.get('bounded_compliant_grasp_accepted',False)),zero_slip_claim=False,
        peak_contact_local_tangential_motion_mm=[1000*grasp[side]['max_material_anchor_tangential_motion_m'] for side in ['left','right']],
        final_contact_local_tangential_motion_mm=[1000*grasp[side]['final_material_anchor_tangential_motion_m'] for side in ['left','right']],
        peak_relative_palm_rocking_degrees=[grasp[side]['max_relative_rotation_deg'] for side in ['left','right']],
        basis=visual.get('grasp_review_note','Not yet visually reviewed.'))
    assessment=dict(status='SUCCESS' if all(gates.values()) and review['accepted_as_bounded_compliant_retention'] else 'INCOMPLETE',
        numeric_and_integrity_gates=gates,grasp_retention_review=review,
        stability_thresholds=dict(final_1s_root_linear_mps=.01,final_1s_root_angular_radps=.03,final_1s_object_linear_mps=.001,final_1s_object_angular_radps=.003),
        raw_verifier_note='Unconditional success:false and legacy 45-degree torso-tilt full_valid fields are preserved. This assessment uses the requested physical criteria and explicit visual review; independent raw replay measurements are unchanged.',
        physics_controls=len(a['ctrl_physics']),duration_seconds=len(a['ctrl_physics'])*.005,minimum_final_1s_foot_loads_N=footmin.tolist(),
        mocap_count=int(m.nmocap),tendon_count=int(m.ntendon),gravity_mps2=m.opt.gravity.tolist())
    (p/'task_assessment.json').write_text(json.dumps(assessment,indent=2))
    print(json.dumps(dict(walk=r['bilateral_retained']['root_net_displacement_m'],alternations=r['bilateral_retained']['longest_consecutive_alternating_replants'],placement=release,grasp=grasp),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);audit(p.parse_args().directory)
