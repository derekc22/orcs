"""Physical collision dimensions, masses, actuation, and measured contact reach."""
import argparse,json,itertools
import numpy as np,mujoco
from mechanism import OUT,indices

def bounds(m,d,geoms,wrist):
    W=d.xmat[wrist].reshape(3,3);o=d.xpos[wrist];lo=np.full(3,np.inf);hi=-lo;reach=0.
    for g in geoms:
        p=W.T@(d.geom_xpos[g]-o);R=W.T@d.geom_xmat[g].reshape(3,3);s=m.geom_size[g];typ=int(m.geom_type[g])
        if typ==2:extent=np.ones(3)*s[0];r=np.linalg.norm(p)+s[0]
        elif typ==3:
            v=s[1]*R[:,2];extent=np.abs(v)+s[0];r=max(np.linalg.norm(p-v),np.linalg.norm(p+v))+s[0]
        elif typ==5:
            axis=R[:,2];extent=s[0]*np.sqrt(np.maximum(0,1-axis*axis))+s[1]*np.abs(axis);par=p@axis;perp=np.sqrt(max(0,p@p-par*par));r=np.sqrt((abs(par)+s[1])**2+(perp+s[0])**2)
        elif typ==6:
            extent=np.abs(R)@s;corners=np.array(list(itertools.product([-1,1],repeat=3)))*s;r=np.linalg.norm(corners@R.T+p,axis=1).max()
        else:raise ValueError((m.geom(g).name,typ))
        lo=np.minimum(lo,p-extent);hi=np.maximum(hi,p+extent);reach=max(reach,float(r))
    return dict(low_m=lo.tolist(),high_m=hi.tolist(),dimensions_m=(hi-lo).tolist(),farthest_collision_surface_from_wrist_m=reach)

def run(tag):
    path=OUT/(tag+'.npz');a=np.load(path);m=mujoco.MjModel.from_binary_path(str(OUT/(tag+'_model.mjb')));d=mujoco.MjData(m);aj,aq,av,arm,pads=indices(m)
    report=dict(model=tag+'_model.mjb',native_actuator_count=29,palm_press_actuator_count=2,finger_curl_actuator_count=2,opposition_thumb_actuator_count=2,added_wrist_actuator_count=2,total_actuator_count=m.nu,hand_mass_kg={},geometry={},mechanism='Two independent wrist-mounted palms; each has one 60 mm force-driven palm carriage, one hinged fork with two short fingers, and one powered swivel. No common object-spanning frame.',contact_geometries={})
    for side in ['left','right']:
        gs=[g for g in range(m.ngeom) if m.geom(g).name.startswith('hand_'+side+'_')];bs=[b for b in range(m.nbody) if m.body(b).name.startswith('hand_'+side)]
        report['hand_mass_kg'][side]=float(m.body_mass[bs].sum());report['contact_geometries'][side]=[dict(name=m.geom(g).name,type=int(m.geom_type[g]),size_m=m.geom_size[g].tolist()) for g in gs]
        wrist=m.body(side+'_wrist_yaw_link').id;rows={}
        for name,press,curl in [('neutral',0.,0.),('open',-.025,0.),('closed',.035,1.57)]:
            d.qpos[:]=a['initial_qpos'];d.qpos[aq[29:31]]=press;d.qpos[aq[31:33]]=0.;d.qpos[aq[33:35]]=curl;d.qpos[aq[35:37]]=0 if name!='closed' else 1.10
            mujoco.mj_kinematics(m,d);rows[name]=bounds(m,d,gs,wrist)
        observed=[]
        for q in a['qpos'][::4]:
            d.qpos[:]=q;mujoco.mj_kinematics(m,d);observed.append(bounds(m,d,gs,wrist))
        rows['observed_wrist_frame_union']=dict(low_m=np.min([x['low_m'] for x in observed],axis=0).tolist(),high_m=np.max([x['high_m'] for x in observed],axis=0).tolist(),farthest_collision_surface_from_wrist_m=max(x['farthest_collision_surface_from_wrist_m'] for x in observed))
        rows['observed_wrist_frame_union']['dimensions_m']=(np.array(rows['observed_wrist_frame_union']['high_m'])-rows['observed_wrist_frame_union']['low_m']).tolist();report['geometry'][side]=rows
    report['total_hand_mass_kg']=sum(report['hand_mass_kg'].values());report['actuators']=[dict(name=m.actuator(i).name,joint=m.joint(int(aj[i])).name,joint_type=('slide' if m.jnt_type[aj[i]]==2 else 'hinge'),joint_units=('m' if m.jnt_type[aj[i]]==2 else 'rad'),force_units=('N' if m.jnt_type[aj[i]]==2 else 'Nm'),joint_range=m.jnt_range[aj[i]].tolist(),effort_limit=m.actuator_forcerange[i].tolist(),position_gain=float(m.actuator_gainprm[i,0]),measured_position_range=[float(a['qpos'][:,aq[i]].min()),float(a['qpos'][:,aq[i]].max())]) for i in range(29,m.nu)]
    distances=[[],[]];weights=[[],[]];ob=m.body('object/object').id;d=mujoco.MjData(m);d.qpos[:]=a['initial_qpos'];d.qvel[:]=a['initial_qvel'];mujoco.mj_forward(m,d)
    for u in a['ctrl_physics']:
        d.ctrl[:]=u;mujoco.mj_step(m,d)
        for ci,c in enumerate(d.contact):
            b1,b2=m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]
            if ob not in [b1,b2]:continue
            n=m.geom(c.geom2 if b1==ob else c.geom1).name
            if not n.startswith('hand_'):continue
            f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f);mag=np.linalg.norm(f[:3])
            if mag<1e-6:continue
            side=0 if n.startswith('hand_left_') else 1;w=m.body(['left','right'][side]+'_wrist_yaw_link').id
            distances[side].append(float(np.linalg.norm(c.pos-d.xpos[w])));weights[side].append(mag)
    report['actual_wrist_to_contact_m']={side:dict(min=float(np.min(distances[i])),mean=float(np.mean(distances[i])),max=float(np.max(distances[i])),force_weighted_mean=float(np.average(distances[i],weights=weights[i])),samples=len(distances[i])) for i,side in enumerate(['left','right'])}
    (OUT/(tag+'_dimensions.json')).write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ['hand_mass_kg','total_hand_mass_kg','actual_wrist_to_contact_m','geometry']},indent=2));return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('tag');run(p.parse_args().tag)
