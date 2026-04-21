# Copyright (c) 2025-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Normalized joint position environment configuration for SO-ARM101.

This environment uses normalized observations and actions in [-100, +100] range
to match the real SO-101 robot hardware format.
"""

from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm, ObservationGroupCfg as ObsGroup
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import math
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg
from isaaclab_tasks.manager_based.manipulation.robots.so_arm101_urdf_cfg import SO_ARM101_URDF_CFG


@configclass
class SoArm101ReachNormalizedEnvCfg(ReachEnvCfg):
    """SO-ARM101 reach environment with normalized observations and actions.

    - Observations: Joint positions in normalized [-100, +100] space (not radians)  
                    Velocities are still in rad/s
    - Actions: Joint position commands in normalized [-100, +100] space (not radians)
    """
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Use calibrated URDF directly to match LeRobot FK coordinate frame exactly
        self.scene.robot = SO_ARM101_URDF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set the body name for the end effector in rewards
        # Gripper Tip
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["gripper_frame_link"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["gripper_frame_link"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["gripper_frame_link"]

        # Set workspace bounds for SO-ARM101
        self.commands.ee_pose.ranges.pos_x = (0.12, 0.29)
        self.commands.ee_pose.ranges.pos_y = (-0.24, 0.24)
        self.commands.ee_pose.ranges.pos_z = (0.15, 0.40)

        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)

        # Set command generator body name
        self.commands.ee_pose.body_name = "gripper_frame_link"

        # 400Hz timesteps in simulation
        # env step every timestep
        # total steps per episode = 400Hz * 12s
        self.sim_dt = 1/400.0 
        self.decimation = 1 
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0

        # Using normalized joint positions. [-100, 100].
        # Arm is controlled with joint positions. Using normalized is just a sim2real consideration,
        # as using the SO-ARM101 with LeRobot requires this convention for joints.
        self.actions.arm_action = mdp.NormalizedJointPositionActionCfg(
            class_type=mdp.NormalizedJointPositionAction,
            asset_name="robot",
            joint_names=["shoulder_.*", "elbow_flex", "wrist_.*"],
            scale=15.0,
            use_default_offset=True,
        )

        # Ignore gripper orientation, just need to get tip to correct location.
        self.rewards.end_effector_orientation_tracking.weight = 0.0

        # Normalized joint observations for same reason as above.
        @configclass
        class NormalizedPolicyCfg(ObsGroup):
            """Observations for policy group - all in normalized [-100, +100] space."""

            # Joint positions relative to default (0 is center of range of motion for all 
            # except gripper, which is the closed position), in normalized space.
            joint_pos = ObsTerm(
                func=mdp.joint_pos_normalized_100_rel, 
                noise=Unoise(n_min=-1.0, n_max=1.0)
            )
            
            # Joint velocities (using rad/s, in sim2real this is calculated as we know roughly
            # how many radians one normalized step is)
            joint_vel = ObsTerm(
                func=mdp.joint_vel_normalized,
                params={"velocity_scale": 1.0},
                noise=Unoise(n_min=-0.01, n_max=0.01)
            )
            
            # Target pose command, same as standard environments
            pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
            
            # Last action
            actions = ObsTerm(func=mdp.last_action)

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True

        # Replace observation config
        self.observations.policy = NormalizedPolicyCfg()


@configclass
class SoArm101ReachNormalizedEnvCfg_PLAY(SoArm101ReachNormalizedEnvCfg):
    """Play configuration for SO-ARM101 normalized reach environment."""
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play 
        self.observations.policy.enable_corruption = False
