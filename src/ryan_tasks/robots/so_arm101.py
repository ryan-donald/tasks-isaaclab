# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SO-ARM101 (SO-101) arm: articulation config, servo model constants and helpers.

The URDF is TheRobotStudio's ``so101_new_calib.urdf`` with the joint limits of
the real arm. The actuator values are fitted to the real servos.
"""

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg
from isaaclab_tasks.utils import preset

URDF_ASSETS_DATA_DIR = Path(__file__).resolve().parent / "data" / "so101"

# encoder resolution, rad (4096 ticks per turn), and the servo deadband in
# normalized units (off).
POS_QUANTUM = 2 * math.pi / 4096
ACTION_DEADBAND = 0.0

# servo speed limit, rad/s, at Goal_Velocity 1000, fitted to the real arm. 0.0
# disables.
MAX_SLEW_RAD_S = 1.57

# servo setpoint acceleration limit, rad/s^2, fitted to the real arm. 0.0
# disables.
MAX_ACCEL_RAD_S2 = 600.0

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def _arm(value: float) -> dict[str, float]:
    # the same value on every arm joint, keyed by name.
    return dict.fromkeys(ARM_JOINTS, value)


SO_ARM101_URDF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{URDF_ASSETS_DATA_DIR}/so101_new_calib.urdf",
        fix_base=False,
        fix_root_link=True,
        merge_fixed_joints=False,
        activate_contact_sensors=False,
        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=False, max_depenetration_velocity=5.0
        ),
        articulation_props=[
            PhysxArticulationCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
            NewtonArticulationCfg(self_collision_enabled=False),
        ],
        # the Newton backend reads the drive gains from the imported joints.
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=400, damping=40
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={**_arm(0.0), "wrist_flex": -0.0385, "gripper": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        # one STS3215 per joint, fitted to the real arm per physics backend (PhysX
        # is the default).
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*", "elbow_flex", "wrist_.*"],
            effort_limit_sim=preset(default=2.48, newton_mjwarp=2.82),
            velocity_limit_sim=4.99,
            armature=0.015,
            stiffness=preset(default=_arm(21.6), newton_mjwarp=_arm(18.8)),
            damping=preset(default=_arm(1.65), newton_mjwarp=_arm(1.20)),
            # load-dependent gearbox friction: a unitless coefficient on PhysX, N*m
            # on Newton.
            friction=preset(
                default={
                    "shoulder_pan": 0.25,
                    "shoulder_lift": 0.014,
                    "elbow_flex": 0.33,
                    "wrist_.*": 0.23,
                },
                newton_mjwarp={
                    "shoulder_pan": 0.18,
                    "shoulder_lift": 0.41,
                    "elbow_flex": 0.38,
                    "wrist_.*": 0.23,
                },
            ),
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=2.5,
            velocity_limit_sim=3.18,
            armature=0.03,
            stiffness=17.8,
            damping=1.5,
        ),
    },
    # no padding on the joint limits, so normalized +/-100 is the full joint range.
    soft_joint_pos_limit_factor=1.0,
)


# prim paths of the gripper bodies, as laid out by the URDF importer.
GRIPPER_PRIM = (
    "{ENV_REGEX_NS}/Robot/Geometry/base_link/shoulder_link/upper_arm_link"
    "/lower_arm_link/wrist_link/gripper_link"
)
JAW_PRIM = GRIPPER_PRIM + "/moving_jaw_so101_v1_link"


def enable_nested_contact_reporting(spawn_cfg):
    """Wrap ``spawn_cfg.func`` so every nested rigid body gets the contact-report API.

    ``activate_contact_sensors`` stops at the first rigid body, but the URDF importer
    nests the links (base_link/.../gripper_link/moving_jaw), so a contact sensor on the
    gripper would find no reporting bodies. This tags the whole subtree.
    """
    if getattr(spawn_cfg, "_contact_reporting_wrapped", False):
        return
    original = spawn_cfg.func

    def _spawn(prim_path, cfg, *args, **kwargs):
        from isaaclab.sim.utils import get_current_stage
        from pxr import Usd, UsdPhysics

        prim = original(prim_path, cfg, *args, **kwargs)
        stage = get_current_stage()
        for root in stage.Traverse():
            path = str(root.GetPath())
            if path.endswith("/Robot") and "/envs/env_" in path:
                for sub in Usd.PrimRange(root):
                    if sub.HasAPI(UsdPhysics.RigidBodyAPI):
                        applied = sub.GetAppliedSchemas()
                        if "PhysxRigidBodyAPI" not in applied:
                            sub.AddAppliedSchema("PhysxRigidBodyAPI")
                        if "PhysxContactReportAPI" not in applied:
                            sub.AddAppliedSchema("PhysxContactReportAPI")
        return prim

    spawn_cfg.func = _spawn
    spawn_cfg._contact_reporting_wrapped = True
