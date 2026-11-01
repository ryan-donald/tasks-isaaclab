# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SO-ARM101 cube pick-and-place with normalized joint observations and actions.

The cube is picked up, carried to a random target on the table, released there, and
the arm returns to its default pose. The rewards form a ladder whose top is that
resting state, so a single run with fixed weights trains the whole task.
"""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from ryan_tasks.tasks.lift import mdp
from ryan_tasks.tasks.lift.config.so_arm101.joint_pos_env_cfg import (
    SoArm101LiftNormalizedEnvCfg,
)

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
# cube half edge [m].
HALF_EXTENT = 0.015
# shared by the placed, released and home terms.
PLACE_PARAMS = {
    "std": 0.02,
    "command_name": "object_pose",
    "rest_clearance": 0.005,
    "half_extent": HALF_EXTENT,
}
RELEASE_PARAMS = {
    **PLACE_PARAMS,
    # open is 1.6 rad; a jaw stalled on the cube sits near 0.4.
    "open_position": 1.0,
    "gripper_cfg": SceneEntityCfg("robot", joint_names=["gripper"]),
}


@configclass
class SoArm101PlaceNormalizedEnvCfg(SoArm101LiftNormalizedEnvCfg):
    """SO-ARM101 pick-and-place to a table target, then back to the default pose."""

    def __post_init__(self):
        super().__post_init__()
        # time to pick, carry, place and return at the half-speed servo limit.
        self.episode_length_s = 12.0

        # one action unit spans about a joint's whole range.
        self.actions.arm_action.scale = 100.0
        self.rewards.action_l2 = None
        self.rewards.action_past_limits = None
        self.rewards.arm_command_rate = None
        self.rewards.action_rate.weight = -1.0
        # the carry posture is free: the cube ends on the table.
        self.rewards.gripper_level = None
        # fixed weights for the whole run.
        self.curriculum = None
        self.rewards.ee_table_contact.weight = -5.0

        # one target per episode on the table, at least 8 cm from the cube's spawn.
        self.commands.object_pose = mdp.PlaceTargetCommandCfg(
            asset_name="robot",
            object_name="object",
            resampling_time_range=(self.episode_length_s + 1.0,) * 2,
            debug_vis=False,
            position_only=True,
            min_distance=0.08,
            carry_clearance=0.02,
            half_extent=HALF_EXTENT,
            ranges=mdp.PlaceTargetCommandCfg.Ranges(
                pos_x=(0.16, 0.30),
                pos_y=(-0.15, 0.15),
                pos_z=(HALF_EXTENT, HALF_EXTENT),
                roll=(0.0, 0.0),
                pitch=(0.0, 0.0),
                yaw=(0.0, 0.0),
            ),
        )
        # the carried latch the place rewards are gated on; on hardware it follows
        # from the observed cube height.
        self.observations.policy.carried = ObsTerm(
            func=mdp.carried_observation, params={"command_name": "object_pose"}
        )

        # rewards, as a ladder: pick (lift's terms, ~14 at the carry height) < carried
        # (+20, with the pick terms off so lowering and letting go cost nothing) <
        # carry to the target (+21) < set down (+25) < release (+20) < arm home (+10
        # coarse, +10 fine).
        self.rewards.reaching_object.func = mdp.object_ee_distance_until_carried
        self.rewards.jaw_closing_on_cube.func = mdp.object_grasped_until_carried
        self.rewards.object_pinched.func = mdp.object_pinched_until_carried
        self.rewards.lifting_object.func = mdp.object_lift_until_carried
        self.rewards.lifting_success.func = mdp.object_lifted_until_carried
        for name in (
            "reaching_object",
            "jaw_closing_on_cube",
            "object_pinched",
            "lifting_object",
            "lifting_success",
        ):
            getattr(self.rewards, name).params["command_name"] = "object_pose"
        self.rewards.object_carried = RewTerm(
            func=mdp.object_carried,
            params={"command_name": "object_pose"},
            weight=20.0,
        )
        self.rewards.object_goal_tracking = RewTerm(
            func=mdp.object_target_distance,
            params={"std": 0.1, "command_name": "object_pose"},
            weight=16.0,
        )
        self.rewards.object_goal_tracking_fine_grained = RewTerm(
            func=mdp.object_target_distance,
            params={"std": 0.02, "command_name": "object_pose"},
            weight=5.0,
        )
        self.rewards.object_placed = RewTerm(
            func=mdp.object_placed, params=PLACE_PARAMS, weight=25.0
        )
        # a gradient toward opening the gripper on the target.
        self.rewards.gripper_open_command = RewTerm(
            func=mdp.gripper_open_command, params=PLACE_PARAMS, weight=5.0
        )
        self.rewards.object_released = RewTerm(
            func=mdp.object_released, params=RELEASE_PARAMS, weight=20.0
        )
        # coarse kernel for the way back from the placing pose (~1.2 rad away), fine
        # kernel for the last stretch.
        home_params = {
            **RELEASE_PARAMS,
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS),
        }
        self.rewards.arm_home = RewTerm(
            func=mdp.arm_home,
            params={**home_params, "home_std": 1.0},
            weight=10.0,
        )
        self.rewards.arm_home_fine_grained = RewTerm(
            func=mdp.arm_home,
            params={**home_params, "home_std": 0.2},
            weight=10.0,
        )


@configclass
class SoArm101PlaceNormalizedEnvCfg_PLAY(SoArm101PlaceNormalizedEnvCfg):
    """Evaluation config: no observation noise, delays fixed at the measured value."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
        self.events.randomize_joint_friction = None
        # 2 steps, the measured latency.
        self.events.randomize_action_delay.params["min_delay"] = 2
        self.events.randomize_action_delay.params["max_delay"] = 2
        self.events.randomize_gripper_action_delay.params["min_delay"] = 2
        self.events.randomize_gripper_action_delay.params["max_delay"] = 2
        self.commands.object_pose.debug_vis = True
