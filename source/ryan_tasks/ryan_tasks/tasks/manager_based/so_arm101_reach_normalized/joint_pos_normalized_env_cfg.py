# Copyright (c) 2025-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Normalized joint position environment configuration for SO-ARM101.

This environment uses normalized observations and actions in [-100, +100] range
to match the real SO-101 robot hardware format.
"""

from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm, ObservationGroupCfg as ObsGroup, EventTermCfg as EventTerm, SceneEntityCfg, CurriculumTermCfg as CurrTerm, RewardTermCfg as RewTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import math
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg
from ryan_tasks.tasks.robots.so_arm101_urdf_cfg import SO_ARM101_URDF_CFG


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

        # use calibrated URDF directly to match LeRobot FK coordinate frame exactly.
        # updated specifically for my robot.
        self.scene.robot = SO_ARM101_URDF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # set the body name for the end effector in rewards
        # gripper Tip
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["gripper_frame_link"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["gripper_frame_link"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["gripper_frame_link"]

        # set workspace bounds for SO-ARM101 end-effector targets
        self.commands.ee_pose.ranges.pos_x = (0.2, 0.25)
        self.commands.ee_pose.ranges.pos_y = (-0.2, 0.2)
        self.commands.ee_pose.ranges.pos_z = (0.15, 0.3)

        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)

        # set command generator body name
        self.commands.ee_pose.body_name = "gripper_frame_link"

        # simulation update rates. 60hz default.
        hz = 60.0
        self.sim_dt = 1.0/hz
        self.decimation = 1 
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0

        # curriculum terms that allow the robot to first learn how to reach the goal,
        # then progressively learn smoother and slower motions to reach it.
        self.curriculum.action_rate.params["num_steps"] = 5_000 * 24
        self.curriculum.action_rate.params["weight"] = -0.001
        self.curriculum.action_rate_s2 = CurrTerm(
            func=mdp.modify_reward_weight,
            params={"term_name": "action_rate", "weight": -0.004, "num_steps": 7_500 * 24},
        )
        self.curriculum.action_rate_s3 = CurrTerm(
            func=mdp.modify_reward_weight,
            params={"term_name": "action_rate", "weight": -0.008, "num_steps": 10_000 * 24},
        )

        self.curriculum.joint_vel.params["num_steps"] = 5_000 * 24
        self.curriculum.joint_vel.params["weight"] = -0.001
        self.curriculum.joint_vel_s2 = CurrTerm(
            func=mdp.modify_reward_weight,
            params={"term_name": "joint_vel", "weight": -0.003, "num_steps": 7_500 * 24},
        )
        self.curriculum.joint_vel_s3 = CurrTerm(
            func=mdp.modify_reward_weight,
            params={"term_name": "joint_vel", "weight": -0.006, "num_steps": 10_000 * 24},
        )

        # arm is controlled using position control, in normalized ranges [-100, 100].
        # matches lerobot.
        self.actions.arm_action = mdp.NormalizedJointPositionActionCfg(
            class_type=mdp.NormalizedJointPositionAction,
            asset_name="robot",
            joint_names=["shoulder_.*", "elbow_flex", "wrist_flex"],
            scale=100.0,
            use_default_offset=True,
        )

        # maximum number of steps for action delay.
        self.actions.arm_action.delay_steps = 6

        # Ignore gripper orientation, just need to get tip to correct location.
        self.rewards.end_effector_orientation_tracking.weight = 0.0

        # the four joints we are concerned with and controlling. for reach, gripper
        # doesn't have an impact on the end-effector location, and wrist_roll
        # has minimal, added a lot of noise during training.
        obs_joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]

        @configclass
        class NormalizedPolicyCfg(ObsGroup):
            """Observations for policy group - all in normalized [-100, +100] space."""

            # joint positions relative to default (0 is center of range of motion for all 
            # except gripper, which is the closed position), in normalized space.
            joint_pos = ObsTerm(
                func=mdp.joint_pos_normalized_100_rel,
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=obs_joints)},
                noise=Unoise(n_min=-1.0, n_max=1.0)
            )
            
            # joint velocities (rad/s), computed as a finite difference in the same
            # manner as the lerobot deployment script. 
            joint_vel = ObsTerm(
                func=mdp.joint_vel_finite_diff,
                params={"velocity_scale": 1.0, "asset_cfg": SceneEntityCfg("robot", joint_names=obs_joints)},
                noise=Unoise(n_min=-0.01, n_max=0.01)
            )
            
            # target pose command, same as standard environments
            pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
            
            # action buffer including the last 6 action taken by the actor. allows
            # the network to learn through the real-world robot motor delays.
            actions = ObsTerm(func=mdp.last_action, history_length=6)

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True

        # replaces observation with normalized observation matching lerobot.
        self.observations.policy = NormalizedPolicyCfg()

        # PD gain domain randomization, in the range [50%, 150%] of the nominal value.
        self.events.randomize_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.5, 1.5),
                "damping_distribution_params": (0.5, 1.5),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        # action delay domain randomization, on the real robot I measured around 50-70ms
        # of delay.
        self.events.randomize_action_delay = EventTerm(
            func=mdp.randomize_action_delay,
            mode="reset",
            params={"action_name": "arm_action", "min_delay": 4, "max_delay": 6},
        )

@configclass
class SoArm101ReachNormalizedEnvCfg_PLAY(SoArm101ReachNormalizedEnvCfg):
    # configuration of task used for play. disables domain randomization.
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # disable sim2real domain randomization so play runs under nominal
        self.events.randomize_gains = None
        self.events.randomize_action_delay = EventTerm(
            func=mdp.randomize_action_delay,
            mode="reset",
            params={"action_name": "arm_action", "min_delay": 5, "max_delay": 5},
        )


@configclass
class SoArm101ReachNormalizedEnvCfg_FIXEDDELAY(SoArm101ReachNormalizedEnvCfg):
    # version identical to the default version, but with action delay fixed at 5

    def __post_init__(self):
        super().__post_init__()
        # self.events.randomize_action_delay = EventTerm(
        #     func=mdp.randomize_action_delay,
        #     mode="reset",
        #     params={"action_name": "arm_action", "min_delay": 5, "max_delay": 5},
        # )

        # tight tracking term with a small std, gives a strong gradient within the
        # last couple cm of the goal so the agent locks on instead of hovering.
        self.rewards.end_effector_position_tracking_tight = RewTerm(
            func=mdp.position_command_error_tanh,
            weight=0.05,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["gripper_frame_link"]),
                "std": 0.02,
                "command_name": "ee_pose",
            },
        )
        self.rewards.end_effector_position_tracking_fine_grained.weight=0.05
