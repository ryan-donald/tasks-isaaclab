# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pick and place, manager-based.

The cube is picked up, carried to a random target on the table and released there, and
the arm returns to its default pose. The rewards are a ladder whose top is that resting
state, so one run with fixed weights trains the whole task.
"""

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.schemas import MassCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

from ryan_tasks.robots.rebot_b601 import ARM_JOINTS, JAW_JOINT, REBOT_B601_CFG

from . import mdp
from . import params as p


def simulation() -> SimulationCfg:
    return SimulationCfg(
        dt=1.0 / p.HZ,
        render_interval=1,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0
        ),
        physics=PhysxCfg(
            bounce_threshold_velocity=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_patch_count=2**18,
        ),
    )


@configclass
class PlaceSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )
    robot = REBOT_B601_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(2 * p.CUBE_HALF,) * 3,
            mass_props=MassCfg(mass=p.CUBE_MASS),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            rigid_props=PhysxRigidBodyCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.3, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(p.CUBE_LOW[0], 0.0, p.CUBE_HALF), rot=(0.0, 0.0, 0.0, 1.0)
        ),
    )


@configclass
class PlaceEventsCfg:
    reset_scene = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    reset_joints = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS),
            "position_range": (-0.2, 0.2),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_cube = EventTerm(func=mdp.reset_cube, mode="reset")
    randomize_gains = EventTerm(
        func=base_mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[*ARM_JOINTS, JAW_JOINT]),
            "stiffness_distribution_params": (0.75, 1.25),
            "damping_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    jaw_friction = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["clamp_1", "clamp_2"]),
            "static_friction_range": (2.5, 2.5),
            "dynamic_friction_range": (2.5, 2.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )
    randomize_delay = EventTerm(
        func=mdp.randomize_action_delay,
        mode="reset",
        params={"low": 1, "high": p.MAX_DELAY},
    )


@configclass
class ActionsCfg:
    arm = mdp.PlaceActionCfg(asset_name="robot")


@configclass
class CommandsCfg:
    place_target = mdp.PlaceTargetCfg()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos_normalized,
            noise=NoiseModelWithAdditiveBiasCfg(
                noise_cfg=Unoise(n_min=-p.POS_NOISE, n_max=p.POS_NOISE),
                bias_noise_cfg=Unoise(
                    n_min=-p.POS_BIAS, n_max=p.POS_BIAS, operation="abs"
                ),
            ),
        )
        jaw_pos = ObsTerm(
            func=mdp.jaw_pos_normalized,
            noise=Unoise(n_min=-p.JAW_NOISE, n_max=p.JAW_NOISE),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel, noise=Unoise(n_min=-p.VEL_NOISE, n_max=p.VEL_NOISE)
        )
        cube = ObsTerm(
            func=mdp.cube_position,
            noise=Unoise(n_min=-p.CUBE_NOISE, n_max=p.CUBE_NOISE),
        )
        target = ObsTerm(
            func=base_mdp.generated_commands, params={"command_name": "place_target"}
        )
        carried = ObsTerm(func=mdp.carried)
        actions = ObsTerm(func=mdp.last_action, history_length=p.HISTORY)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


def _reward(name: str) -> RewTerm:
    return RewTerm(func=mdp.reward_term, weight=p.WEIGHTS[name], params={"name": name})


@configclass
class RewardsCfg:
    reaching_object = _reward("reaching_object")
    jaw_closing_on_cube = _reward("jaw_closing_on_cube")
    object_pinched = _reward("object_pinched")
    lifting_object = _reward("lifting_object")
    lifting_success = _reward("lifting_success")
    object_carried = _reward("object_carried")
    object_goal_tracking = _reward("object_goal_tracking")
    object_goal_tracking_fine_grained = _reward("object_goal_tracking_fine_grained")
    object_placed = _reward("object_placed")
    gripper_open_command = _reward("gripper_open_command")
    object_released = _reward("object_released")
    arm_home = _reward("arm_home")
    arm_home_fine_grained = _reward("arm_home_fine_grained")
    action_rate = _reward("action_rate")
    joint_vel = _reward("joint_vel")
    joint_near_limits = _reward("joint_near_limits")
    gripper_on_table = _reward("gripper_on_table")


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    diverged = DoneTerm(func=mdp.cube_flew_off)


@configclass
class PlaceEnvCfg(ManagerBasedRLEnvCfg):
    sim: SimulationCfg = simulation()
    scene: PlaceSceneCfg = PlaceSceneCfg(num_envs=12288, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: PlaceEventsCfg = PlaceEventsCfg()
    decimation = 1
    episode_length_s = p.EPISODE_S


@configclass
class PlaceEnvCfg_PLAY(PlaceEnvCfg):
    """No observation noise or randomization, the action delay fixed at 2 steps."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
        self.events.randomize_gains = None
        self.events.randomize_delay.params = {"low": 2, "high": 2}
