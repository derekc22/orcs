"""Conservative planning-only separation from the exact object's outer hull.

No hull geometry is added to the physical model. A positive gap proves a
primitive lies wholly outside one of the enclosing hull's supporting planes.
"""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

class ObjectHull:
    def __init__(self,m):
        self.m=m;self.ob=m.body('object/object').id;points=[]
        for g in range(m.ngeom):
            if m.geom_bodyid[g]!=self.ob or not m.geom_contype[g]:continue
            mi=m.geom_dataid[g];v=m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi]+m.mesh_vertnum[mi]]
            points.append(Rotation.from_quat(np.roll(m.geom_quat[g],-1)).apply(v)+m.geom_pos[g])
        self.points=np.concatenate(points);h=ConvexHull(self.points);self.normals=h.equations[:,:3];self.offsets=h.equations[:,3]
        self.areas=np.array([np.linalg.norm(np.cross(self.points[t[1]]-self.points[t[0]],self.points[t[2]]-self.points[t[0]]))/2 for t in h.simplices])
    def world_planes(self,d):
        return self.normals@d.xmat[self.ob].reshape(3,3).T,self.offsets
    def gap(self,d,g):
        n,b=self.world_planes(d);m=self.m;p=d.geom_xpos[g]-d.xpos[self.ob];R=d.geom_xmat[g].reshape(3,3);s=m.geom_size[g];typ=int(m.geom_type[g]);dot=n@R
        if typ==2:extent=np.full(len(n),s[0])
        elif typ==3:extent=s[0]+s[1]*np.abs(dot[:,2])
        elif typ==6:extent=np.abs(dot)@s
        elif typ==5:extent=s[0]*np.sqrt(np.maximum(0,1-dot[:,2]**2))+s[1]*np.abs(dot[:,2])
        else:extent=np.full(len(n),m.geom_rbound[g])
        return float(np.max(n@p+b-extent))
    def contact_translation(self,d,centers,rotation,sizes,outward,gap=.003):
        n,b=self.world_planes(d);selection=(n@outward>.95)&(self.areas>.005)
        assert selection.any()
        n=n[selection];b=b[selection];shifts=[]
        for p,size in zip(centers,sizes):
            extent=np.abs(n@rotation)@size
            shifts.extend(((extent+gap-(n@(p-d.xpos[self.ob])+b))/(n@outward)).tolist())
        return max(shifts)*outward
