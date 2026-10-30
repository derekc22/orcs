"""Full verified rollout, fixed world camera plus detailed following camera.
Rendering-only rulers/grid never enter the MuJoCo physics model.
"""
import argparse,json
from pathlib import Path
import mujoco,numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont

def render(path):
 path=Path(path);a=np.load(path);m=mujoco.MjModel.from_binary_path(str(path.with_name(path.stem+'_model.mjb')));d=mujoco.MjData(m);l=np.load(path.with_name(path.stem+'_locomotion_trace.npz'))['trace'];t=np.load(path.with_name(path.stem+'_verification_trace.npz'))['trace'];report=json.loads(path.with_name(path.stem+'_sustained_metrics.json').read_text());assert report['all_machine_gates_pass'],'Render selected candidate only after all machine gates pass'
 s,e=map(int,report['metrics']['walking_interval_physics_steps']);oq=m.jnt_qposadr[m.joint('object/floating_base_joint').id];q=a['qpos'];start=q[s,:2];end=q[e-1,:2];direction=(end-start)/np.linalg.norm(end-start);perp=np.array([-direction[1],direction[0]]);opening=float(a['phase_opening_frame'])*4;withdraw=float(a['phase_withdrawal_end_frame'])*4
 m.vis.global_.offwidth=800;m.vis.global_.offheight=600;r=mujoco.Renderer(m,width=800,height=600);fixed=mujoco.MjvCamera();fixed.distance=5.8;fixed.azimuth=float(np.degrees(np.arctan2(direction[1],direction[0]))+90);fixed.elevation=-21;fixed.lookat[:]=np.r_[.5*(q[0,:2]+q[-1,:2])+.15*direction,.65];follow=mujoco.MjvCamera();follow.distance=3.;follow.azimuth=130;follow.elevation=-20;font=ImageFont.load_default(size=20);small=ImageFont.load_default(size=16);writer=imageio.get_writer(str(path.with_suffix('.mp4')),fps=50,codec='libx264',quality=8,macro_block_size=None);montage=[]
 def line(p1,p2,color,width=1):
  g=r.scene.geoms[r.scene.ngeom];r.scene.ngeom+=1;mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_LINE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(color));mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_LINE,width,np.array(p1),np.array(p2))
 def view(camera,step):
  r.update_scene(d,camera=camera);center=.5*(q[0,:2]+q[-1,:2]);lo=np.floor(center-4);hi=np.ceil(center+4)
  for x in np.arange(lo[0],hi[0]+.1,.5):line([x,lo[1],.001],[x,hi[1],.001],[.34,.34,.34,1])
  for y in np.arange(lo[1],hi[1]+.1,.5):line([lo[0],y,.001],[hi[0],y,.001],[.34,.34,.34,1])
  p0=start-.6*perp
  line(np.r_[p0,.004],np.r_[p0+3*direction,.004],[.9,.9,.9,1],2)
  for metre in np.arange(0,3.01,.5):
   p=p0+metre*direction;line(np.r_[p-.1*perp,.005],np.r_[p+.1*perp,.005],[.9,.9,.9,1],3 if metre%1==0 else 1)
  for idx in range(s,min(step,e-1),10):
   j=min(idx+10,min(step,e-1));line(np.r_[q[idx,:2],.008],np.r_[q[j,:2],.008],[.1,.9,.4,1],3)
  return Image.fromarray(r.render().copy())
 frames=len(q)//4
 for f in range(frames):
  step=f*4+3;d.qpos[:]=q[step];d.qvel[:]=a['qvel'][step];mujoco.mj_forward(m,d);follow.lookat[:]=.5*d.qpos[:3]+.5*d.qpos[oq:oq+3];follow.lookat[2]=.65;left=view(fixed,step);right=view(follow,step);im=Image.new('RGB',(1600,600));im.paste(left,(0,0));im.paste(right,(800,0));dr=ImageDraw.Draw(im);k=min(max(step,s),e-1);rootdist=np.linalg.norm(q[k,:2]-start);boxdist=np.linalg.norm(t[k,11:13]-t[s,11:13]);phase='PICKUP / CARRY PREPARATION' if step<s else 'SUSTAINED WALKING CARRY' if step<e else 'BRAKING / LOWERING' if step<opening else 'PHYSICAL OPENING' if step<withdraw else 'WITHDRAWN / STABLE'
  dr.rectangle((0,0,1600,76),fill='black');dr.text((12,7),f'{f/50:.2f} s | {phase}',font=font,fill='white');dr.text((12,36),f'Walking root net {rootdist:.3f} m | box net {boxdist:.3f} m | measured interval {s*.005:.2f}-{e*.005:.2f} s',font=font,fill='white');dr.text((812,7),f'Terrain contact: {bool(t[step,0])} | COM lift {t[step,5]:+.3f} m',font=font,fill='white');dr.text((812,36),f'Foot load L/R {l[step,14]:.0f}/{l[step,15]:.0f} N | sole clearance {100*l[step,12]:.1f}/{100*l[step,13]:.1f} cm',font=small,fill='white');dr.rectangle((0,572,1600,600),fill='black');dr.text((10,578),'Fixed world view | ruler major ticks 1 m, grid 0.5 m | green = actual root path during walking',font=small,fill='white');dr.text((910,578),'Detail view | original geometry | finite physical actuators',font=small,fill='white');writer.append_data(np.array(im))
  if f in np.linspace(0,frames-1,16).astype(int):montage.append(im.resize((640,240)))
 writer.close();r.close();sheet=Image.new('RGB',(2560,960))
 for i,im in enumerate(montage):sheet.paste(im,((i%4)*640,(i//4)*240))
 sheet.save(path.with_name(path.stem+'_montage.png'));print('VIDEO_FRAMES',frames,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('artifact',type=Path);render(p.parse_args().artifact)
