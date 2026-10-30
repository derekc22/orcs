#!/usr/bin/env python3
"""Make an exhaustive frame atlas for visual anti-cheat review of a full replay."""
import argparse,json
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont

def main():
 p=argparse.ArgumentParser();p.add_argument('video',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);reader=imageio.get_reader(a.video);fps=float(reader.get_meta_data()['fps']);font=ImageFont.load_default(size=14);page=None;count=0;pages=[]
 for index,frame in enumerate(reader):
  slot=index%100
  if slot==0:page=Image.new('RGB',(1920,1640),(30,30,30))
  im=Image.fromarray(frame).resize((192,144),Image.Resampling.LANCZOS);x=(slot%10)*192;y=(slot//10)*164;page.paste(im,(x,y));ImageDraw.Draw(page).text((x+4,y+144),f'f{index:04d}  {index/fps:.2f}s',font=font,fill='white');count=index+1
  if slot==99:
   name=f'frames_{index-99:04d}_{index:04d}.png';page.save(a.output/name);pages.append(name);page=None
 if page is not None:
  name=f'frames_{count-count%100:04d}_{count-1:04d}.png';page.save(a.output/name);pages.append(name)
 reader.close();(a.output/'index.json').write_text(json.dumps({'video':str(a.video),'fps':fps,'frame_count':count,'duration_s':count/fps,'every_frame_included_once':True,'pages':pages},indent=2));print(count,pages)
if __name__=='__main__':main()
