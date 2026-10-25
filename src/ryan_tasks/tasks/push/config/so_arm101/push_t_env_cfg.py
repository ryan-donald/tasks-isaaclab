# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SO-ARM101 Push-T: push a T block to a planar goal pose (position and yaw).

Built on the cube push: the same arm, normalized action and observation spaces,
delay randomization and smoothness curriculum. wrist_roll is added to the action
and all six joints are observed. There are no gripper contact sensors or downforce
penalty.
"""

import copy
from pathlib import Path

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import (
    ObservationTermCfg as ObsTerm,
)
from isaaclab.managers import (
    RewardTermCfg as RewTerm,
)
from isaaclab.managers import (
    SceneEntityCfg,
)
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sim.schemas import MassCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

from ryan_tasks.robots.so_arm101 import MAX_SLEW_RAD_S, SO_ARM101_URDF_CFG
from ryan_tasks.tasks.push import mdp
from ryan_tasks.tasks.push.config.so_arm101.joint_pos_env_cfg import (
    _EE_OFFSET,
    SoArm101PushNormalizedEnvCfg,
    SoArm101PushNormalizedEnvCfg_PLAY,
)

# the T block, and its bare geometry for the goal marker (a marker spawned from the
# physics file would add a second, real block).
_ASSETS = Path(__file__).resolve().parent / "assets"
T_BLOCK_USD = str(_ASSETS / "t_block.usda")
T_BLOCK_VISUAL_USD = str(_ASSETS / "t_block_geom.usda")

# block half-thickness: 30 mm total, the same as the cube.
_T_HALF_Z = 0.015

# block mass, kg: solid PLA (1271 kg/m^3 over 55.1 cm^3), so it matches a printed T.
_T_MASS = 0.07

# spawn and goal yaw ranges, rad. Together they give relative errors up to 135 deg;
# the full Push-T benchmark samples both over the whole circle.
_SPAWN_YAW = 0.785398  # +/-45 deg
_GOAL_YAW = 1.570796  # +/-90 deg


@configclass
class SoArm101PushTNormalizedEnvCfg(SoArm101PushNormalizedEnvCfg):
    """SO-ARM101 Push-T: push a T block to a commanded planar pose."""

    def __post_init__(self):
        super().__post_init__()

        # the T replaces the cube. It is two box colliders, which keep the concave
        # notch under the bar that a convex hull would fill in.
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.25, 0.0, _T_HALF_Z], rot=[0, 0, 0, 1]
            ),
            spawn=UsdFileCfg(
                usd_path=T_BLOCK_USD,
                mass_props=MassCfg(mass=_T_MASS),
                rigid_props=PhysxRigidBodyCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # goal pose: the cube task's x band moved 4 cm out for the larger block, flat
        # (no roll or pitch), with a commanded yaw.
        self.commands.object_pose.ranges.pos_x = (0.22, 0.32)
        self.commands.object_pose.ranges.pos_z = (_T_HALF_Z, _T_HALF_Z)
        self.commands.object_pose.ranges.roll = (0.0, 0.0)
        self.commands.object_pose.ranges.pitch = (0.0, 0.0)
        self.commands.object_pose.ranges.yaw = (-_GOAL_YAW, _GOAL_YAW)
        # report orientation_error alongside position_error in the command metrics.
        self.commands.object_pose.position_only = False
        # keep the goal quaternion's sign consistent with what the reward reads.
        self.commands.object_pose.make_quat_unique = True

        # spawn offsets relative to the init state, including yaw.
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.03, 0.03),
            "y": (-0.05, 0.05),
            "z": (0.0, 0.0),
            "yaw": (-_SPAWN_YAW, _SPAWN_YAW),
        }

        # yaw progress, mirroring the inherited position progress term. Weight is
        # reward per radian (3.5 per degree, against 3.0 per mm for position).
        # Raising it trades away position accuracy, which is the harder half.
        self.rewards.object_yaw_tracking = RewTerm(
            func=mdp.object_yaw_progress,
            params={"command_name": "object_pose", "max_closing_rate": 8.0},
            weight=200.0,
        )

        # success: within 2 cm and 15 deg. The inherited position-only object_at_goal
        # stays alongside it as an anchor on position near the goal.
        self.rewards.object_pose_at_goal = RewTerm(
            func=mdp.object_pose_at_goal,
            params={
                "dist_threshold": 0.02,
                "yaw_threshold": 0.261799,
                "command_name": "object_pose",
            },
            weight=1.0,
        )

        # fingertip speed within 3 cm of the T, to soften contact.
        self.rewards.tip_speed_near_object = RewTerm(
            func=mdp.tip_speed_near_object_penalty,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=["gripper_link"]),
                "ee_offset": _EE_OFFSET,
                "radius": 0.03,
                "ref_speed": 0.1,
            },
            weight=-3.0,
        )

        # no contact sensing: drop the gripper sensors, the downforce penalty they
        # feed, its curriculum and the contact-report wrapper.
        self.scene.contact_gripper = None
        self.scene.contact_jaw = None
        self.rewards.gripper_downforce = None
        self.curriculum.downforce_partial = None
        self.curriculum.downforce_full = None
        self.scene.robot.spawn = copy.deepcopy(SO_ARM101_URDF_CFG.spawn)

        # final action-rate weight, 5x the cube push's, against tapping the block.
        self.curriculum.action_rate_full.params["weight"] = -3.0

        # wrist_roll is an action here, to turn the pusher's face, so its hold event
        # is dropped. The gripper stays held closed; as an action, the policy opens it
        # and pushes with the wrist motor housings.
        self.actions.arm_action.joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ]
        self.events.hold_wrist_roll = None
        # half the cube push's servo speed limit: deploy with Goal_Velocity 500.
        self.actions.arm_action.max_slew_rad_s = 0.5 * MAX_SLEW_RAD_S

        # start pose (and zero action): gripper pitched ~60 deg down, tip ~8 cm up
        # and 16 cm out, just behind the T.
        self.scene.robot.init_state.joint_pos = {
            **self.scene.robot.init_state.joint_pos,
            "shoulder_lift": -1.45,
            "elbow_flex": 1.0,
            "wrist_flex": 1.5,
        }

        # all six joints observed, the pinned gripper included.
        all_joints = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
        self.observations.policy.joint_pos.params["joint_names"] = all_joints
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=all_joints
        )

        # block yaw, and the goal pose (replacing the position-only goal term).
        self.observations.policy.object_yaw = ObsTerm(
            func=mdp.object_yaw_sin_cos,
            # +/-0.02 rad (~1.1 deg) of noise.
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        self.observations.policy.goal_position = ObsTerm(
            func=mdp.goal_pose_in_robot_root_frame,
            params={"command_name": "object_pose"},
        )


@configclass
class SoArm101PushTNormalizedEnvCfg_PLAY(
    SoArm101PushNormalizedEnvCfg_PLAY, SoArm101PushTNormalizedEnvCfg
):
    """Evaluation config: no observation noise or sim-to-real randomization."""

    def __post_init__(self):
        super().__post_init__()

        # draw the goal as a translucent T instead of a coordinate frame.
        self.commands.object_pose.debug_vis = True
        self.commands.object_pose.goal_pose_visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Command/goal_pose",
            markers={
                "goal": UsdFileCfg(
                    usd_path=T_BLOCK_VISUAL_USD,
                    visual_material=PreviewSurfaceCfg(
                        diffuse_color=(0.15, 0.75, 0.35), opacity=0.4
                    ),
                ),
            },
        )
        # hide the current-pose marker.
        self.commands.object_pose.curr_pose_visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Command/body_pose",
            markers={
                "curr": UsdFileCfg(usd_path=T_BLOCK_VISUAL_USD, visible=False),
            },
        )
