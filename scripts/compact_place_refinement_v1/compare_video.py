"""Same-rate, same-scale render comparison of raw MuJoCo states."""
import argparse,json
from pathlib import Path
import mujoco,numpy as np,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2]
def run(args):
 paths=[ROOT/'logs/compact_smooth_integration_v2/complete_execution/rollout.npz',args.candidate]
 data=[np.load(p) for p in paths];models=[mujoco.MjModel.from_binary_path(str(p.with_name(p.stem+'_model.mjb'))) for p in paths];ds=[mujoco.MjData(m) for m in models]
 rs=[]
 for m in models:m.vis.global_.offwidth=800;m.vis.global_.offheight=600;rs.append(mujoco.Renderer(m,height=600,width=800))
 writer=imageio.get_writer(str(args.output),fps=50,codec='libx264',quality=7,macro_block_size=None);font=ImageFont.load_default(size=21)
 start=round(args.start/.005);end=min(min(len(a['qpos']) for a in data),round(args.end/.005))
 count=0
 for step in range(start+3,end,4):
  im=Image.new('RGB',(1600,600))
  for i,(a,m,d,r) in enumerate(zip(data,models,ds,rs)):
   k=min(step,len(a['qpos'])-1);d.qpos[:]=a['qpos'][k];d.qvel[:]=a['qvel'][k];mujoco.mj_forward(m,d)
   camera=mujoco.MjvCamera();camera.distance=3;camera.azimuth=130;camera.elevation=-20;camera.lookat[:]=.5*d.qpos[:3]+.5*d.xpos[m.body('object/object').id];camera.lookat[2]=.6
   r.update_scene(d,camera=camera);tile=Image.fromarray(r.render());draw=ImageDraw.Draw(tile);draw.rectangle([0,0,800,62],fill='black');draw.text((10,8),['Accepted b2673fdb',args.label][i],font=font,fill='white');draw.text((10,34),f'{(step+1)*.005:.2f} s | raw physical states, 1x speed',font=font,fill='white');im.paste(tile,(i*800,0))
  writer.append_data(np.array(im));count+=1
  if count%500==0:print('rendered',count,flush=True)
 writer.close()
 for r in rs:r.close()
 args.output.with_suffix('.json').write_text(json.dumps(dict(sources=[str(x) for x in paths],start_s=args.start,end_s=end*.005,frames=count,fps=50,playback_speed=1.,same_camera_parameters=True,unfiltered_states=True,final_frame_hold_if_lengths_differ=False),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('candidate',type=Path);p.add_argument('output',type=Path);p.add_argument('--label',default='Placement refinement');p.add_argument('--start',type=float,default=33.505);p.add_argument('--end',type=float,default=100);run(p.parse_args())
