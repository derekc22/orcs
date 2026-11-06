"""Decode every frame into indexed chronological contact sheets for review."""
import argparse
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont

def sheets(video,out):
    out.mkdir(parents=True,exist_ok=False)
    reader=imageio.get_reader(video);fps=reader.get_meta_data()['fps'];font=ImageFont.load_default(size=10)
    width,height,cols,rows=160,132,12,12
    sheet=None;files=[];count=0
    for index,frame in enumerate(reader):
        slot=index%(cols*rows)
        if slot==0:sheet=Image.new('RGB',(cols*width,rows*height),'white')
        im=Image.fromarray(frame).resize((width,120));tile=Image.new('RGB',(width,height),'white');tile.paste(im,(0,12))
        ImageDraw.Draw(tile).text((2,0),f'{index:04d}  {index/fps:.2f}s',fill='black',font=font)
        sheet.paste(tile,((slot%cols)*width,(slot//cols)*height));count=index+1
        if slot==cols*rows-1:
            name=f'frames_{index-slot:04d}_{index:04d}.jpg';sheet.save(out/name,quality=90);files.append(name);sheet=None
    if sheet is not None:
        name=f'frames_{index-slot:04d}_{index:04d}.jpg';sheet.save(out/name,quality=90);files.append(name)
    reader.close()
    meta=dict(source=str(video),sha256=hashlib.sha256(video.read_bytes()).hexdigest(),frames=count,fps=fps,
              every_frame_included_once=True,files=files,inspection_complete=False)
    (out/'manifest.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('video',type=Path);p.add_argument('out',type=Path);args=p.parse_args();sheets(args.video,args.out)
