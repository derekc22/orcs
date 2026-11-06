"""Accepted compact-hand model expressed as ordinary MJLab entities.

Read only the fork's self-contained compiled model. No historical asset paths
or model-building scripts are executed. XML actuators preserve native servos.
"""
from functools import lru_cache, partial
from pathlib import Path

import mujoco
import numpy as np
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityCfg
from mjlab.entity.entity import EntityArticulationInfoCfg
from mjlab.sim import MujocoCfg

ROOT = Path(__file__).resolve().parents[5]
BASE = ROOT / "logs/compact_policy_learning_v1/complete_execution"
CHECKPOINT = ROOT / "logs/compact_policy_learning_v1/selected_policy.npz"


@lru_cache(maxsize=1)
def source_model():
    return mujoco.MjModel.from_binary_path(str(BASE / "rollout_model.mjb"))


def canonical(name):
    name = name.removeprefix("robot/")
    return name.removeprefix("object/") if name.startswith("object/object/") else name


def entity_spec(kind):
    m = source_model()
    s = mujoco.MjSpec()
    object_id = m.body("object/object").id
    keep = {i for i in range(1, m.nbody) if (i == object_id) == (kind == "object")}
    mesh_ids = [i for i in range(m.nmesh)
                if m.mesh(i).name.startswith("object/") == (kind == "object")]
    for i in mesh_ids:
        va, vn = m.mesh_vertadr[i], m.mesh_vertnum[i]
        fa, fn = m.mesh_faceadr[i], m.mesh_facenum[i]
        s.add_mesh(name=m.mesh(i).name, uservert=m.mesh_vert[va:va+vn].ravel(),
                   userface=m.mesh_face[fa:fa+fn].ravel())
    bodies = {0: s.worldbody}
    for i in sorted(keep):
        name = "object" if kind == "object" else m.body(i).name
        parent = bodies.get(int(m.body_parentid[i]), s.worldbody)
        b = parent.add_body(name=name, pos=m.body_pos[i], quat=m.body_quat[i],
                            mass=m.body_mass[i], inertia=m.body_inertia[i],
                            ipos=m.body_ipos[i], iquat=m.body_iquat[i], explicitinertial=True)
        bodies[i] = b
        for j in range(m.njnt):
            if m.jnt_bodyid[j] != i:
                continue
            name = "floating_base_joint" if kind == "object" else m.joint(j).name
            if m.jnt_type[j] == 0:
                b.add_freejoint(name=name)
            else:
                v = m.jnt_dofadr[j]
                b.add_joint(name=name, type=int(m.jnt_type[j]), pos=m.jnt_pos[j],
                            axis=m.jnt_axis[j], limited=bool(m.jnt_limited[j]),
                            range=m.jnt_range[j], stiffness=m.jnt_stiffness[j],
                            damping=m.dof_damping[v], armature=m.dof_armature[v],
                            frictionloss=m.dof_frictionloss[v], margin=m.jnt_margin[j],
                            solref_limit=m.jnt_solref[j], solimp_limit=m.jnt_solimp[j])
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != i:
                continue
            kw = dict(name=m.geom(g).name or f"compact_geom_{g}", type=int(m.geom_type[g]),
                      pos=m.geom_pos[g], quat=m.geom_quat[g], size=m.geom_size[g],
                      contype=int(m.geom_contype[g]), conaffinity=int(m.geom_conaffinity[g]),
                      condim=int(m.geom_condim[g]), friction=m.geom_friction[g],
                      solref=m.geom_solref[g], solimp=m.geom_solimp[g],
                      margin=m.geom_margin[g], gap=m.geom_gap[g],
                      group=int(m.geom_group[g]), rgba=m.geom_rgba[g], mass=0.)
            if m.geom_type[g] == 7:
                kw["meshname"] = m.mesh(int(m.geom_dataid[g])).name
            b.add_geom(**kw)
    if kind == "robot":
        for a in range(m.nu):
            s.add_actuator(name=m.actuator(a).name, target=m.joint(int(m.actuator_trnid[a, 0])).name,
                           trntype=int(m.actuator_trntype[a]), gear=m.actuator_gear[a],
                           gaintype=int(m.actuator_gaintype[a]), biastype=int(m.actuator_biastype[a]),
                           gainprm=m.actuator_gainprm[a], biasprm=m.actuator_biasprm[a],
                           ctrllimited=bool(m.actuator_ctrllimited[a]), ctrlrange=m.actuator_ctrlrange[a],
                           forcelimited=bool(m.actuator_forcelimited[a]), forcerange=m.actuator_forcerange[a])
        for signature in m.exclude_signature:
            a, b = divmod(int(signature), 65536)
            if a in keep and b in keep:
                s.add_exclude(bodyname1=m.body(a).name, bodyname2=m.body(b).name)
    return s


def robot_cfg():
    return EntityCfg(spec_fn=partial(entity_spec, "robot"),
                     articulation=EntityArticulationInfoCfg(
                         actuators=(XmlActuatorCfg(target_names_expr=(".*",), command_field="position"),),
                         soft_joint_pos_limit_factor=1.0))


def object_cfg():
    return EntityCfg(spec_fn=partial(entity_spec, "object"))


def add_terrain(s):
    m = source_model()
    g = m.geom("terrain").id
    s.worldbody.add_geom(name="terrain", type=int(m.geom_type[g]), pos=m.geom_pos[g],
                         quat=m.geom_quat[g], size=m.geom_size[g],
                         contype=int(m.geom_contype[g]), conaffinity=int(m.geom_conaffinity[g]),
                         condim=int(m.geom_condim[g]), friction=m.geom_friction[g],
                         solref=m.geom_solref[g], solimp=m.geom_solimp[g],
                         margin=m.geom_margin[g], gap=m.geom_gap[g], rgba=m.geom_rgba[g])


class AcceptedPhysicsCfg(MujocoCfg):
    """Restore compiled physical fields by name before MJWarp model upload."""
    def apply(self, model):
        old = source_model()
        for name in dir(old.opt):
            if name.startswith('_'):
                continue
            try:
                value = getattr(old.opt, name)
                if isinstance(value, np.ndarray):
                    getattr(model.opt, name)[:] = value
                elif isinstance(value, (float, int)):
                    setattr(model.opt, name, value)
            except (AttributeError, TypeError):
                pass
        # Mesh compilation recentres already-compiled vertices. Restore the
        # immutable source's geometry, including its acceleration structures.
        for i in range(model.nmesh):
            name = canonical(model.mesh(i).name)
            if name.startswith("object/object/"):
                name = name.removeprefix("object/")
            j = old.mesh(name).id
            for field, adr, count in [("mesh_vert", "mesh_vertadr", "mesh_vertnum"),
                                      ("mesh_face", "mesh_faceadr", "mesh_facenum")]:
                a, b, n = getattr(model, adr)[i], getattr(old, adr)[j], getattr(old, count)[j]
                getattr(model, field)[a:a+n] = getattr(old, field)[b:b+n]
        for kind, prefix, fields in [
            (mujoco.mjtObj.mjOBJ_BODY, "body", ("pos", "quat", "ipos", "iquat", "mass", "inertia")),
            (mujoco.mjtObj.mjOBJ_GEOM, "geom", ("pos", "quat", "size", "contype", "conaffinity", "condim", "friction", "solref", "solimp", "margin", "gap", "rbound", "aabb")),
            (mujoco.mjtObj.mjOBJ_JOINT, "jnt", ("pos", "axis", "range", "margin", "stiffness", "solref", "solimp")),
            (mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator", ("gainprm", "biasprm", "gear", "ctrlrange", "forcerange", "ctrllimited", "forcelimited")),
        ]:
            count = getattr(model, {"body":"nbody", "geom":"ngeom", "jnt":"njnt", "actuator":"nu"}[prefix])
            for i in range(count):
                name = canonical(mujoco.mj_id2name(model, kind, i) or "")
                if name.startswith("object/compact_geom_"):
                    name = name.removeprefix("object/")
                if not name:
                    continue
                j = int(name.split("compact_geom_")[-1]) if name.startswith("compact_geom_") else mujoco.mj_name2id(old, kind, name)
                if j < 0:
                    continue
                for field in fields:
                    getattr(model, f"{prefix}_{field}")[i] = getattr(old, f"{prefix}_{field}")[j]
        mujoco.mj_setConst(model, mujoco.MjData(model))
        # The accepted frozen model retains these compiled solver constants.
        # Recompiling its post-build inertias would otherwise change contact
        # regularization despite identical exposed mass/inertia parameters.
        for field in ("qpos0", "qpos_spring", "body_invweight0", "dof_M0",
                      "dof_invweight0", "actuator_acc0", "dof_length"):
            getattr(model, field)[:] = getattr(old, field)
        for field in dir(old):
            if not field.startswith(("mesh_", "bvh_", "body_bvh")):
                continue
            a, b = getattr(model, field), getattr(old, field)
            if isinstance(a, np.ndarray) and a.shape == b.shape:
                a[:] = b
