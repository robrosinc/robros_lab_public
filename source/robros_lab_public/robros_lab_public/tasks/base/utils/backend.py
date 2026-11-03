"""Physics-backend presets shared by all robros_lab_public tasks."""

from __future__ import annotations

from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sensors import ContactSensorCfg as PhysXContactSensorCfg
from isaaclab_tasks.utils import PresetCfg, preset


@configclass
class RobrosPhysicsCfg(PresetCfg):
    """PhysX and Newton/MJWarp backends used throughout robros_lab_public.

    Newton's scene-wide shape margin is intentionally zero. Rough-terrain IGRIS-C
    tasks opt the two feet into a 1 cm margin with
    :func:`spawn_igris_c_with_ankle_roll_margin`.
    """

    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=200,
            nconmax=100,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        num_substeps=1,
        debug_mode=False,
        default_shape_cfg=NewtonShapeCfg(margin=0.0),
    )
    # Newton/MJWarp is the robros_lab_public runtime default.  Keep PhysX available
    # as an explicit ``physics=physx`` override for compatibility/debugging.
    default = newton_mjwarp
    physx = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)


@configclass
class RobrosContactSensorCfg(PresetCfg):
    """Backend-specific full-articulation contact sensor."""

    newton_mjwarp = NewtonContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    default = newton_mjwarp
    physx = PhysXContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )


def contact_sensor_preset(**kwargs: Any) -> PresetCfg:
    """Create a contact-sensor preset with identical backend-independent options."""

    newton_mjwarp = NewtonContactSensorCfg(**kwargs)
    return preset(
        default=newton_mjwarp,
        physx=PhysXContactSensorCfg(**kwargs),
        newton_mjwarp=newton_mjwarp,
    )


IGRISC_FOOT_BODY_NAMES = ("Link_Ankle_Roll_Left", "Link_Ankle_Roll_Right")


@sim_utils.clone
def spawn_igris_c_with_ankle_roll_margin(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn IGRIS-C with a 1 cm Newton contact margin only on its feet."""

    return _spawn_with_foot_contact_margin(prim_path, cfg, IGRISC_FOOT_BODY_NAMES, translation, orientation, **kwargs)


def _spawn_with_foot_contact_margin(prim_path, cfg, foot_body_names, translation, orientation, **kwargs):
    """Apply the shared contact margin to the selected robot's foot bodies."""

    # Imported lazily so importing/configuring PhysX tasks does not require the
    # Newton USD schema module to initialize.
    from isaaclab_newton.sim.schemas import NewtonCollisionPropertiesCfg
    from pxr import Usd

    prim = sim_utils.spawn_from_usd(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )

    foot_collision_cfg = NewtonCollisionPropertiesCfg(contact_margin=0.01)
    stage = sim_utils.get_current_stage()
    for foot_body_name in foot_body_names:
        direct_path = f"{prim_path}/{foot_body_name}"
        direct_prim = stage.GetPrimAtPath(direct_path)
        if direct_prim.IsValid():
            foot_prim_path = direct_path
        else:
            # Isaac Sim 6's layered URDF export nests articulation links below
            # /Geometry/base_link, while the older flat export placed them
            # directly below the articulation root. Resolve the link by its
            # exact name and reject ambiguous assets rather than guessing.
            matches = [
                prim
                for prim in stage.Traverse()
                if prim.GetName() == foot_body_name and prim.GetPath().pathString.startswith(prim_path)
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected exactly one '{foot_body_name}' below '{prim_path}', found "
                    f"{len(matches)}: {[p.GetPath() for p in matches]}"
                )
            foot_prim_path = str(matches[0].GetPath())
        # The robot USD keeps each link's visual/collision geometry in
        # instanceable references. USD does not permit per-instance schema
        # overrides below those references, so de-instance only the two feet
        # before authoring their Newton margin.
        sim_utils.make_uninstanceable(foot_prim_path)
        sim_utils.modify_collision_properties(foot_prim_path, foot_collision_cfg)

        foot_prim = stage.GetPrimAtPath(foot_prim_path)
        margin_was_authored = any(
            prim.HasAttribute("newton:contactMargin")
            and abs(prim.GetAttribute("newton:contactMargin").Get() - foot_collision_cfg.contact_margin) < 1.0e-6
            for prim in Usd.PrimRange(foot_prim, Usd.TraverseInstanceProxies())
        )
        if not margin_was_authored:
            raise RuntimeError(f"Failed to apply Newton foot contact margin below '{foot_prim_path}'.")

    return prim
