"""Construct specs using ONLY the fork's accepted self-contained MJB assets.

The local schema supplies the tree and defaults. No file-backed mesh/texture or
canonical-checkout source is opened. Physical arrays are restored by name after
compilation so the original object and native robot retain their exact geometry.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=ROOT/'logs/sustained_carry_v1/complete2_model.mjb'
_old=None

def reference():
    global _old
    if _old is None:_old=mujoco.MjModel.from_binary_path(str(BASE))
    return _old

def spec(kind='full'):
    old=reference();tree=ET.fromstring((HERE/'schema_source.xml').read_text())
    # Remove all file references before parsing any MuJoCo spec.
    compiler=tree.find('compiler');compiler.attrib.pop('meshdir',None)
    for asset in tree.find('asset'):
        asset.attrib.pop('file',None)
        if asset.tag=='mesh':
            for attr in ['scale','refpos','refquat']:asset.attrib.pop(attr,None)
    # The accepted compiled mesh coordinates are used as input coordinates.
    def body_geoms(element):
        for b in element.findall('body'):
            name=b.get('name')
            if name.startswith('hand_'):continue
            try:ob=old.body(name).id
            except KeyError:ob=None
            if ob is not None:
                candidates=range(old.body_geomadr[ob],old.body_geomadr[ob]+old.body_geomnum[ob])
                for g in b.findall('geom'):
                    matches=[og for og in candidates if (g.get('name') and old.geom(og).name==g.get('name')) or (g.get('mesh') and old.geom_type[og]==7 and old.mesh(int(old.geom_dataid[og])).name==g.get('mesh'))]
                    if len(matches)==1:
                        og=matches[0];g.set('pos',' '.join(map(repr,old.geom_pos[og].tolist())));g.set('quat',' '.join(map(repr,old.geom_quat[og].tolist())))
            body_geoms(b)
    body_geoms(tree.find('worldbody'))
    # Discard the prototype hands/actuators; the active builder adds its version.
    for parent in tree.iter():
        for child in list(parent):
            if child.tag=='body' and child.get('name','').startswith('hand_'):parent.remove(child)
    actuators=tree.find('actuator')
    for a in list(actuators):
        if 'hand_' in a.get('joint',''):actuators.remove(a)
    if kind!='full':
        tree.remove(actuators)
        for tag in (['contact','equality','keyframe','sensor','tendon'] if kind=='object' else ['equality','keyframe','sensor','tendon']):
            for section in tree.findall(tag):tree.remove(section)
        world=tree.find('worldbody')
        for child in list(world):
            keep=child.tag=='body' and ((child.get('name')=='object/object')==(kind=='object'))
            if not keep:world.remove(child)
        if kind=='object':
            body=world.find('body');body.set('name','object')
            for j in body.findall('freejoint'):j.set('name','floating_base_joint')
            for j in body.findall('joint'):j.set('name','floating_base_joint')
    s=mujoco.MjSpec.from_string(ET.tostring(tree,encoding='unicode'))
    for mesh in s.meshes:
        mi=old.mesh(mesh.name).id;va=old.mesh_vertadr[mi];vn=old.mesh_vertnum[mi];fa=old.mesh_faceadr[mi];fn=old.mesh_facenum[mi]
        mesh.file='';mesh.uservert=old.mesh_vert[va:va+vn].ravel();mesh.userface=old.mesh_face[fa:fa+fn].ravel();mesh.scale[:]=1
        if old.mesh_texcoordnum[mi]:
            ta=old.mesh_texcoordadr[mi];tn=old.mesh_texcoordnum[mi]
            mesh.usertexcoord=old.mesh_texcoord[ta:ta+tn].ravel();mesh.userfacetexcoord=old.mesh_facetexcoord[fa:fa+fn].ravel()
    for t in s.textures:
        ti=old.texture(t.name).id;t.file='';t.width=int(old.tex_width[ti]);t.height=int(old.tex_height[ti]);t.nchannel=int(old.tex_nchannel[ti]);n=t.width*t.height*t.nchannel;t.data=old.tex_data[old.tex_adr[ti]:old.tex_adr[ti]+n].tobytes()
    return s

def native_spec():return spec('native')
def object_spec():return spec('object')

def restore(m):
    old=reference()
    assert m.nmesh==old.nmesh
    for name in ['mesh_vertnum','mesh_facenum','mesh_vertadr','mesh_faceadr','mesh_normalnum','mesh_normaladr','mesh_texcoordnum','mesh_texcoordadr']:
        assert np.array_equal(getattr(m,name),getattr(old,name)),name
    for name in ['mesh_vert','mesh_face','mesh_normal','mesh_facenormal','mesh_texcoord','mesh_facetexcoord','mesh_pos','mesh_quat','mesh_scale','mesh_graph','mesh_polynormal','mesh_polyvert','mesh_polyvertadr','mesh_polyvertnum','mesh_polymap','mesh_polyadr','mesh_polynum','mesh_polymapadr','mesh_polymapnum']:
        a=getattr(m,name);b=getattr(old,name)
        assert a.shape[1:]==b.shape[1:] and len(a)>=len(b),(name,a.shape,b.shape)
        a[:len(b)]=b
    for i in range(m.nmesh):
        start=int(m.mesh_bvhadr[i]);other=int(old.mesh_bvhadr[i]);n=int(m.mesh_bvhnum[i]);assert n==old.mesh_bvhnum[i]
        for name in ['bvh_aabb','bvh_child','bvh_nodeid','bvh_depth']:
            getattr(m,name)[start:start+n]=getattr(old,name)[other:other+n]
    for b in range(1,m.nbody):
        n=m.body(b).name
        if n.startswith('hand_'):continue
        ob=old.body(n).id
        for field in ['body_pos','body_quat','body_ipos','body_iquat','body_mass','body_inertia']:
            getattr(m,field)[b]=getattr(old,field)[ob]
        gs=range(m.body_geomadr[b],m.body_geomadr[b]+m.body_geomnum[b]);ogs=range(old.body_geomadr[ob],old.body_geomadr[ob]+old.body_geomnum[ob])
        for g in gs:
            if m.geom(g).name:matches=[h for h in ogs if old.geom(h).name==m.geom(g).name]
            elif m.geom_type[g]==7:matches=[h for h in ogs if old.geom_type[h]==7 and old.geom_dataid[h]==m.geom_dataid[g]]
            else:matches=[]
            assert len(matches)==1,(n,m.geom(g).name,matches)
            h=matches[0]
            for field in ['geom_pos','geom_quat','geom_size','geom_rbound','geom_aabb','geom_contype','geom_conaffinity','geom_condim','geom_friction','geom_solref','geom_solimp','geom_margin','geom_gap']:
                getattr(m,field)[g]=getattr(old,field)[h]
    for j in range(m.njnt):
        if m.joint(j).name.startswith('hand_'):continue
        oj=old.joint(m.joint(j).name).id
        for field in ['jnt_pos','jnt_axis','jnt_range','jnt_margin','jnt_stiffness','jnt_solref','jnt_solimp']:
            getattr(m,field)[j]=getattr(old,field)[oj]
    mujoco.mj_setConst(m,mujoco.MjData(m))
    return m
