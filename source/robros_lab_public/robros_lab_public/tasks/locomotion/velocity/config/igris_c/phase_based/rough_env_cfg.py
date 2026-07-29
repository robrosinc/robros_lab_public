# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robros_lab_public.tasks.locomotion.velocity.mdp as robros_mdp

from ..base.rough_env_cfg import IGRISCActions as IGRISCBaseActions
from ..base.rough_env_cfg import IGRISCObservations as IGRISCBaseObservations
from ..base.rough_env_cfg import IGRISCRewards as IGRISCBaseRewards
from ..base.rough_env_cfg import IGRISCRoughEnvCfg as IGRISCBaseRoughEnvCfg


@configclass
class IGRISCRewards(IGRISCBaseRewards):
    """Phase-based reward deltas layered on top of the base locomotion rewards."""

    # -- task
    contact_swing_track = RewTerm(
        func=robros_mdp.contact_swing_track,
        weight=1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*Ankle_Roll.*")},
    )
    # -- style
    feet_air_time = None
    penalize_double_stance = None
    arm_swing = RewTerm(
        func=robros_mdp.arm_swing,
        weight=0.2,
        params={
            "arms_asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Shoulder_.*", ".*_Elbow_.*"]),
            "swing_asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Shoulder_Pitch_.*"]),
        },
    )
    # -- regularization
    gait_action_limit = RewTerm(func=robros_mdp.gait_action_limit, weight=-0.25, params={"action_term": "gait_action"})


@configclass
class IGRISCActions(IGRISCBaseActions):
    """Native joint-position actions plus the phase-frequency action."""

    gait_action = robros_mdp.GaitFreqActionCfg(asset_name="robot", lpf_alpha=0.3)


@configclass
class IGRISCObservations(IGRISCBaseObservations):
    """Phase-based IGRIS-C observations built on top of the base IGRIS-C observation layout."""

    @configclass
    class PolicyCfg(IGRISCBaseObservations.PolicyCfg):
        phase = ObsTerm(
            func=robros_mdp.gait_phase_obs,
            params={"asset_cfg": SceneEntityCfg("robot"), "default_gait_freq": 1.0},
        )
        lpf_f = ObsTerm(func=robros_mdp.gait_action)

    @configclass
    class CriticCfg(IGRISCBaseObservations.CriticCfg):
        phase = ObsTerm(
            func=robros_mdp.gait_phase_obs,
            params={"asset_cfg": SceneEntityCfg("robot"), "default_gait_freq": 1.0},
        )
        lpf_f = ObsTerm(func=robros_mdp.gait_action)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class IGRISCRoughEnvCfg(IGRISCBaseRoughEnvCfg):
    """Phase-based rough-terrain environment built on top of the base locomotion config."""

    actions: IGRISCActions = IGRISCActions()
    observations: IGRISCObservations = IGRISCObservations()
    rewards: IGRISCRewards = IGRISCRewards()


@configclass
class IGRISCRoughEnvCfg_PLAY(IGRISCRoughEnvCfg):
    pass
