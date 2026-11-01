# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""B601-DM reach: the SO-101 reach task with the robot swapped and the goals scaled."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from ryan_tasks.robots.rebot_b601 import (
    JAW_JOINT,
    JAW_OPEN,
    MIMIC_JAW_JOINT,
    POS_QUANTUM,
    REBOT_B601_CFG,
)
from ryan_tasks.tasks.reach import mdp
from ryan_tasks.tasks.reach.config.so_arm101.joint_pos_env_cfg import (
    SoArm101ReachNormalizedEnvCfg,
    SoArm101ReachNormalizedEnvCfg_PLAY,
)

# tip at (0.60, 0, 0.40) m, fingers level.
REACH_POSE = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -1.9002,
    "elbow_flex": -1.7489,
    "wrist_flex": -0.1339,
    "wrist_yaw": 0.0,
    "wrist_roll": 0.0,
}


def use_b601(cfg, ee_pose):
    """Swap the robot and scale the goals in a manager-based or direct reach cfg."""
    cfg.scene.robot = REBOT_B601_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.scene.robot.init_state = cfg.scene.robot.init_state.replace(
        joint_pos={**REACH_POSE, JAW_JOINT: JAW_OPEN, MIMIC_JAW_JOINT: -JAW_OPEN}
    )
    # the SO-101 goal box scaled to this arm's reach.
    ee_pose.ranges.pos_x = (0.18, 0.45)
    ee_pose.ranges.pos_y = (-0.36, 0.36)
    ee_pose.ranges.pos_z = (0.036, 0.72)
    # no action drives these, so only a reset recovers them from a floor strike.
    cfg.events.reset_passive_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["wrist_yaw", "wrist_roll", MIMIC_JAW_JOINT, JAW_JOINT],
            ),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


def _use_quantum(cfg):
    cfg.observations.policy.joint_pos.params["pos_quantum"] = POS_QUANTUM
    cfg.observations.policy.joint_vel.params["pos_quantum"] = POS_QUANTUM


@configclass
class B601ReachNormalizedEnvCfg(SoArm101ReachNormalizedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        use_b601(self, self.commands.ee_pose)
        _use_quantum(self)


@configclass
class B601ReachNormalizedEnvCfg_PLAY(SoArm101ReachNormalizedEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        use_b601(self, self.commands.ee_pose)
        _use_quantum(self)
