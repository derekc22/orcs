"""Every-frame task-view panels, supplemented by the full-frame replay montage.

The fixed crop contains robot, box and all nearby terrain in this unchanged
camera view. Original full videos remain untouched. Every frame is included,
in row-major order; no time subsampling or averaging is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont


def panels(video):
    out=video.parent/'task_video_panels';out.mkdir(exist_ok=False)
    reader=imageio.get_reader(video);sheet=None;files=[];font=ImageFont.load_default(size=11)
    for index,frame in enumerate(reader):
        slot=index%500
        if slot==0:
            sheet=Image.new('RGB',(1600,1640),'white')
            ImageDraw.Draw(sheet).text((4,3),f'{video.parent.name}: every frame {index} onward; 25 columns x 20 rows; 50 Hz; crop x230:730 y80:690',fill='black',font=font)
        tile=Image.fromarray(frame).crop((230,80,730,690)).resize((64,80))
        sheet.paste(tile,((slot%25)*64,40+(slot//25)*80))
        if slot==499:
            name=f'frames_{index-slot:04d}_{index:04d}.jpg';sheet.save(out/name,quality=95);files.append(name);sheet=None
    if sheet is not None:
        name=f'frames_{index-slot:04d}_{index:04d}.jpg';sheet.save(out/name,quality=95);files.append(name)
    reader.close()
    meta=dict(video=str(video),sha256=hashlib.sha256(video.read_bytes()).hexdigest(),frames=index+1,fps=50,
              files=files,every_frame_included_once=True,task_crop=[230,80,730,690],
              crop_thumbnail_size=[64,80],full_frame_context='rollout_montage.png and unchanged full 960x720 video',inspection_complete=False)
    (out/'manifest.json').write_text(json.dumps(meta,indent=2));print(video.parent.name,index+1,'frames',len(files),'panels',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('videos',type=Path,nargs='+');args=p.parse_args()
    for video in args.videos:panels(video)
