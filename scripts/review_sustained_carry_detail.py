"""Every walking frame at a readable manipulation/foot-detail scale."""
import argparse,json
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont

def main(path):
 a=np.load(path);s=int(a['walking_start_physics_step'])//4;e=(int(a['walking_end_physics_step'])+3)//4;out=path.with_name(path.stem+'_walking_detail');out.mkdir(exist_ok=False);reader=imageio.get_reader(path.with_suffix('.mp4'));font=ImageFont.load_default(size=14);pages=[]
 for n,f in enumerate(range(s,e)):
  slot=n%20
  if slot==0:page=Image.new('RGB',(1600,1040),(22,22,22));first=f
  frame=Image.fromarray(reader.get_data(f));frame=frame.crop((800,0,1600,600)).resize((320,240));x=(slot%5)*320;y=(slot//5)*260;page.paste(frame,(x,y));ImageDraw.Draw(page).text((x+5,y+240),f'frame {f} | {f/50:.2f} s',font=font,fill='white')
  if slot==19 or f==e-1:
   name=f'frames_{first:04d}_{f:04d}.png';page.save(out/name);pages.append(name)
 reader.close();(out/'index.json').write_text(json.dumps(dict(video=path.with_suffix('.mp4').name,frame_range=[s,e-1],count=e-s,every_walking_frame_included_once=True,pages=pages),indent=2));print('WALK_DETAIL',e-s,len(pages))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('artifact',type=Path);main(p.parse_args().artifact)
