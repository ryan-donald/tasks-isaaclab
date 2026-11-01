# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The numbers of the place task, for the manager-based and direct tasks."""

# control and physics rate, Hz.
HZ = 60
EPISODE_S = 12.0

# half edge of the cube, m.
CUBE_HALF = 0.025
CUBE_MASS = 0.07

# cube spawn and target boxes in the root frame, m.
CUBE_LOW = (0.33, -0.075)
CUBE_HIGH = (0.405, 0.075)
TARGET_LOW = (0.24, -0.225)
TARGET_HIGH = (0.45, 0.225)
MIN_TARGET_DISTANCE = 0.12

# action scale: the SO-101's 100 over the 1.8x longer arm, so that action noise moves
# the tip as far as it does on the SO-101.
ACTION_SCALE = 55.0
MAX_DELAY = 3
HISTORY = 6

# observation noise: normalized units, rad/s, jaw units, m.
POS_NOISE = 0.3
POS_BIAS = 0.7
VEL_NOISE = 0.01
JAW_NOISE = 1.5
CUBE_NOISE = 0.005

# the grasp point in the gripper body's frame, 1 cm above the finger tips past the
# cube's half edge, so the fingers clear the table with the cube on it.
GRASP_OFFSET = (-CUBE_HALF - 0.01, 0.0, 0.0)

# jaw states, m: on the cube it reads CUBE_HALF, past OPEN_POSITION it is off.
PINCH_DISTANCE = 0.02
PINCH_MARGIN = 0.003
OPEN_POSITION = 0.0375

# heights of the cube's lowest vertex, m.
CARRY_CLEARANCE = 0.02
LIFTED_CLEARANCE = 0.01
REST_CLEARANCE = 0.005

# past this from the origin, m, an env has diverged.
MAX_CUBE_RANGE = 1.5

# reward kernel scales, m and rad.
REACH_STD = 0.1
GRASP_STD = 0.02
LIFT_STD = 0.05
CARRY_STD = 0.1
CARRY_FINE_STD = 0.02
PLACE_STD = 0.02
HOME_STD = 1.0
HOME_FINE_STD = 0.2
LIMIT_BAND = 0.1745
TABLE_MARGIN = 0.005

# a ladder: pick < carried < carry to the target < set down < release < arm home.
WEIGHTS = {
    "reaching_object": 1.0,
    "jaw_closing_on_cube": 1.0,
    "object_pinched": 2.0,
    "lifting_object": 15.0,
    "lifting_success": 5.0,
    "object_carried": 20.0,
    "object_goal_tracking": 16.0,
    "object_goal_tracking_fine_grained": 5.0,
    "object_placed": 25.0,
    "gripper_open_command": 5.0,
    "object_released": 20.0,
    "arm_home": 10.0,
    "arm_home_fine_grained": 10.0,
    "action_rate": -1.0,
    "joint_vel": -0.001,
    "joint_near_limits": -1.0,
    "gripper_on_table": -5.0,
}
