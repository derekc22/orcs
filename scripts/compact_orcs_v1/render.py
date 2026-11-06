"""Render every recorded ORCS physics state at its original 50 Hz video rate."""
import argparse
import json
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image,ImageDraw
import mujoco
import numpy as np


def render(path, preview_only=False):
    a=np.load(path/'rollout.npz');m=mujoco.MjModel.from_binary_path(str(path/'rollout_model.mjb'));d=mujoco.MjData(m)
    # Display contrast only; the saved physical model is never rewritten.
    for g in range(m.ngeom):
        name=m.geom(g).name or ''
        if name=='terrain':m.geom_rgba[g]=[.22,.24,.27,1.]
        elif name.startswith('robot/') and m.geom_group[g]==2:m.geom_rgba[g]=[.70,.78,.87,1.]
        elif name.endswith('largebox_visual'):m.geom_rgba[g]=[.64,.43,.22,1.]
    m.vis.global_.offwidth=960;m.vis.global_.offheight=720
    renderer=mujoco.Renderer(m,height=720,width=960);camera=mujoco.MjvCamera();camera.distance=3.;camera.azimuth=130;camera.elevation=-20
    writer=None if preview_only else imageio.get_writer(str(path/'rollout.mp4'),fps=50,codec='libx264',quality=7,macro_block_size=None)
    video_steps=range(3,len(a['qpos']),4)
    milestones=np.linspace(3,video_steps[-1],36).astype(int);pictures=[]
    steps=milestones if preview_only else video_steps
    for step in steps:
        d.qpos[:]=a['qpos'][step];d.qvel[:]=a['qvel'][step]
        mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        camera.lookat[:]=.5*d.qpos[:3]+.5*d.qpos[44:47];camera.lookat[2]=.6
        renderer.update_scene(d,camera=camera)
        for axis in [0,1]:
            center=camera.lookat[axis]
            for coordinate in np.arange(np.floor(center)-3,np.ceil(center)+3,.25):
                g=renderer.scene.geoms[renderer.scene.ngeom];renderer.scene.ngeom+=1
                mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_LINE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([.42,.44,.47,1.]))
                other=camera.lookat[1-axis]
                p1=np.array([coordinate,other-3,.001]) if axis==0 else np.array([other-3,coordinate,.001])
                p2=np.array([coordinate,other+3,.001]) if axis==0 else np.array([other+3,coordinate,.001])
                mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_LINE,1.,p1,p2)
        im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
        draw.rectangle((0,0,960,25),fill='black');draw.text((10,7),f'ORCS UOLM | CompactFeedbackActor | t={(step+1)*.005:.3f} s | actual MJWarp rollout',fill='white')
        if writer is not None:writer.append_data(np.asarray(im))
        if len(pictures)<36 and step>=milestones[len(pictures)]:pictures.append(im.resize((320,240)))
    if writer is not None:writer.close()
    renderer.close()
    if not preview_only:
        reader=imageio.get_reader(str(path/'rollout.mp4'));count=reader.count_frames();metadata=reader.get_meta_data();reader.close()
        (path/'video_metadata.json').write_text(json.dumps(dict(frames=count,fps=metadata['fps'],duration_s=metadata['duration'],size=metadata['size'],
            rendering='Recorded actual ORCS states; display contrast and ground grid only, no state interpolation or time scaling'),indent=2))
    sheet=Image.new('RGB',(1920,1440))
    for i,im in enumerate(pictures):sheet.paste(im,((i%6)*320,(i//6)*240))
    sheet.save(path/'storyboard.jpg')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('artifact',type=Path);p.add_argument('--preview-only',action='store_true');a=p.parse_args();render(a.artifact,a.preview_only)
