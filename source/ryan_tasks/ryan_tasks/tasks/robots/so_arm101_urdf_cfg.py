import copy
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

URDF_ASSETS_DATA_DIR = Path(__file__).resolve().parent

##
# Configuration - URDF Version
##

SO_ARM101_URDF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=True,
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{URDF_ASSETS_DATA_DIR}/so101_new_calib.urdf",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        # joint_drive must be defined here for the newton physics backend.
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=400, damping=40
            )
        ),
    ),
    # default starting state
    init_state=ArticulationCfg.InitialStateCfg(
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": -0.0,
            "wrist_flex": -0.0385,
            "wrist_roll": -0.0,
            "gripper": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # shoulder pan, shoulder lift, elbow, wrist pitch, wrist roll, gripper
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*", "elbow_flex", "wrist_.*"],
            effort_limit_sim=1.9,
            velocity_limit_sim=5.5,
            armature=0.03,
            stiffness={
                "shoulder_pan": 400.0,
                "shoulder_lift": 400.0,
                "elbow_flex": 400.0,
                "wrist_flex": 400.0,
                "wrist_roll": 600.0,
            },
            damping={
                "shoulder_pan": 40.0,
                "shoulder_lift": 40.0,
                "elbow_flex": 40.0,
                "wrist_flex": 40.0,
                "wrist_roll": 40.0,
            },
            friction={
                "shoulder_pan": 0.005,
                "shoulder_lift": 0.005,
                "elbow_flex": 0.005,
                "wrist_flex": 0.005,
                "wrist_roll": 0.005,
            },
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=2.5,
            velocity_limit_sim=3.18,
            armature=0.03,
            stiffness=400.0,
            damping=40.0,
        ),
    },
    # there is no safety padding on joint limits with 1.0. i.e. with 0.9, sim would
    # tell the agent [-100, 100], but the actual joint range would scale to [-90, 90].
    soft_joint_pos_limit_factor=1.0,
)


##
# Configuration - Parallel-gripper URDF
##

# Same calibrated arm as SO_ARM101_URDF_CFG, but with a new parallel gripper.
#   right_clamp = 0.000 m -> jaws closed (~13 mm gap)
#   right_clamp = 0.037 m -> jaws open   (~87 mm gap)
SO_ARM101_PARALLEL_URDF_CFG = copy.deepcopy(SO_ARM101_URDF_CFG)
SO_ARM101_PARALLEL_URDF_CFG.spawn.asset_path = (
    f"{URDF_ASSETS_DATA_DIR}/so101_parallel.urdf"
)
# Collision on gripper is only with fingers, not the strip with teeth for the geal to
# actuate, as that caused a bad collision mesh preventing grasping..
SO_ARM101_PARALLEL_URDF_CFG.spawn.collision_type = "Convex Decomposition"
SO_ARM101_PARALLEL_URDF_CFG.spawn.articulation_props.enabled_self_collisions = False
SO_ARM101_PARALLEL_URDF_CFG.init_state.joint_pos = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": -0.0,
    "wrist_flex": 0.75,
    "wrist_roll": -0.0,
    "right_clamp": 0.0,  # closed at rest; lift task overrides to 0.037 (open)
}
SO_ARM101_PARALLEL_URDF_CFG.actuators["gripper"] = ImplicitActuatorCfg(
    joint_names_expr=["right_clamp"],
    effort_limit_sim=20.0,
    velocity_limit_sim=0.5,
    armature=0.001,
    stiffness=800.0,
    damping=40.0,
)
