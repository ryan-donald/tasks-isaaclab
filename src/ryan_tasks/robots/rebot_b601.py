# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Seeed reBot Arm B601-DM, with the SO-101's joint names and the vendor's MIT gains.

Gravity is off on the arm links, standing in for the vendor's gravity feed-forward.
``left_clamp`` mimics ``right_clamp``, so set them as a mirrored pair.
"""

from pathlib import Path

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

from ryan_tasks.robots.so_arm101 import SO_ARM101_URDF_CFG

DATA_DIR = Path(__file__).resolve().parent / "data" / "rebot_b601"

# feedback position step, rad: 16 bits over +/-12.5 rad.
POS_QUANTUM = 25.0 / 65536

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
JAW_JOINT = "right_clamp"
MIMIC_JAW_JOINT = "left_clamp"

# right_clamp travel, m: 0 is closed.
JAW_OPEN = 0.05

# fingers along +x with their tips at the origin, closing along y.
GRIPPER_BODY = "gripper_link"

# hover with the fingers straight down, jaws closing along world y.
DEFAULT_POSE = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -1.9551,
    "elbow_flex": -1.7231,
    "wrist_flex": 1.3388,
    "wrist_yaw": 0.0,
    "wrist_roll": 0.0,
}

REBOT_B601_CFG = SO_ARM101_URDF_CFG.replace(
    spawn=SO_ARM101_URDF_CFG.spawn.replace(
        asset_path=str(DATA_DIR / "urdf" / "rebot_b601.urdf"),
        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=True, max_depenetration_velocity=5.0
        ),
    ),
    init_state=SO_ARM101_URDF_CFG.init_state.replace(
        joint_pos={**DEFAULT_POSE, JAW_JOINT: JAW_OPEN, MIMIC_JAW_JOINT: -JAW_OPEN}
    ),
    actuators={
        # DM-J4340P: 40:1, 2e-5 kg*m^2 rotor.
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*", "elbow_flex"],
            effort_limit_sim=27.0,
            velocity_limit_sim=5.5,
            stiffness=120.0,
            damping=8.0,
            armature=0.032,
        ),
        # DM-J4310: 10:1, 1.8e-5 kg*m^2 rotor.
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_.*"],
            effort_limit_sim=7.0,
            velocity_limit_sim=20.9,
            stiffness=18.0,
            damping=2.0,
            armature=0.0018,
        ),
        # placeholder drive, not fitted to the real gripper.
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[JAW_JOINT],
            effort_limit_sim=8.0,
            velocity_limit_sim=0.08,
            stiffness=800.0,
            damping=40.0,
            armature=0.001,
        ),
        # driving the mimic joint would fight the mimic constraint.
        "gripper_mimic": ImplicitActuatorCfg(
            joint_names_expr=[MIMIC_JAW_JOINT], stiffness=0.0, damping=0.0
        ),
    },
)
