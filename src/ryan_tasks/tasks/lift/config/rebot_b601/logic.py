# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The place task's state and rewards, for the manager-based and direct tasks.

The task reads no contact forces: a grasp, a release and a gripper on the table all
follow from the jaw position and the poses, as they would on the real arm.
"""

import torch
from isaaclab.utils.math import quat_apply

from ryan_tasks.robots.rebot_b601 import JAW_OPEN

from . import params as p

# in the order the manager-based task lists them.
TERM_NAMES = (
    "reaching_object",
    "jaw_closing_on_cube",
    "object_pinched",
    "lifting_object",
    "lifting_success",
    "object_carried",
    "object_goal_tracking",
    "object_goal_tracking_fine_grained",
    "object_placed",
    "gripper_open_command",
    "object_released",
    "arm_home",
    "arm_home_fine_grained",
    "action_rate",
    "joint_vel",
    "joint_near_limits",
    "gripper_on_table",
)
WEIGHTS = tuple(p.WEIGHTS[name] for name in TERM_NAMES)


class PlaceLogic:
    """Targets, the carried latch and the reward terms."""

    def __init__(self, num_envs: int, origins: torch.Tensor):
        device = origins.device
        self.origins = origins
        self.target_b = torch.zeros(num_envs, 3, device=device)
        self.target_b[:, 2] = p.CUBE_HALF
        self.carried = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.low = torch.tensor(p.TARGET_LOW, device=device)
        self.span = torch.tensor(p.TARGET_HIGH, device=device) - self.low
        self.grasp_offset = torch.tensor(p.GRASP_OFFSET, device=device).repeat(
            num_envs, 1
        )
        self.terms = torch.zeros(num_envs, len(TERM_NAMES), device=device)
        self.pinched = torch.zeros(num_envs, device=device)

    def reset(self, env_ids, cube_pos_w: torch.Tensor):
        """Draw targets at least MIN_TARGET_DISTANCE (xy) from the cube's spawn."""
        cube = (cube_pos_w - self.origins)[env_ids, :2]
        target = self.low + self.span * torch.rand_like(cube)
        for _ in range(8):
            close = torch.norm(target - cube, dim=1) < p.MIN_TARGET_DISTANCE
            redraw = self.low + self.span * torch.rand_like(cube)
            target = torch.where(close.unsqueeze(1), redraw, target)
        self.target_b[env_ids, :2] = target
        self.carried[env_ids] = False

    def write_terms(
        self,
        gripper: torch.Tensor,
        cube_pose: torch.Tensor,
        jaw: torch.Tensor,
        joint_pos: torch.Tensor,
        home: torch.Tensor,
        joint_vel: torch.Tensor,
        margin_to_limit: torch.Tensor,
        action: torch.Tensor,
        prev_action: torch.Tensor,
    ):
        """Fill ``terms`` and update ``carried``. ``joint_vel`` has the jaw's too."""
        gripper_pos, gripper_quat = gripper[:, :3], gripper[:, 3:7]
        cube_pos = cube_pose[:, :3]
        grasp = gripper_pos + quat_apply(gripper_quat, self.grasp_offset)
        to_grasp = torch.norm(cube_pos - grasp, dim=1)
        reach = 1.0 - torch.tanh(to_grasp / p.REACH_STD)

        # the cube's lowest vertex above the table, from the rotation's third row.
        x, y, z, w = (cube_pose[:, i] for i in range(3, 7))
        span = (
            (2.0 * (x * z - w * y)).abs()
            + (2.0 * (y * z + w * x)).abs()
            + (1.0 - 2.0 * (x * x + y * y)).abs()
        ) * p.CUBE_HALF
        clearance = cube_pos[:, 2] - span

        # the jaws cannot close past the cube, so a jaw this near the cube's half width
        # with the cube between them has it pinched.
        pinched = (to_grasp < p.PINCH_DISTANCE) & (jaw < p.CUBE_HALF + p.PINCH_MARGIN)
        lifted = clearance > p.CARRY_CLEARANCE
        carried = self.carried | (lifted & pinched)
        self.carried.copy_(carried)
        carried_f = carried.float()
        until_carried = 1.0 - carried_f

        offset = self.target_b + self.origins - cube_pos
        distance = torch.norm(offset, dim=1)
        xy = torch.norm(offset[:, :2], dim=1)
        on_table = (clearance < p.REST_CLEARANCE).float()
        placed = carried_f * on_table * (1.0 - torch.tanh(xy / p.PLACE_STD))
        released = placed * (jaw > p.OPEN_POSITION).float()
        home_distance = torch.norm(joint_pos - home, dim=1)

        terms = (
            until_carried * reach,
            until_carried
            * (1.0 - torch.tanh(to_grasp / p.GRASP_STD))
            * ((JAW_OPEN - jaw) / JAW_OPEN).clamp(0.0, 1.0),
            until_carried * pinched.float(),
            until_carried * torch.tanh(clearance.clamp(min=0.0) / p.LIFT_STD) * reach,
            until_carried * (clearance > p.LIFTED_CLEARANCE).float(),
            carried_f,
            carried_f * (1.0 - torch.tanh(distance / p.CARRY_STD)),
            carried_f * (1.0 - torch.tanh(distance / p.CARRY_FINE_STD)),
            placed,
            placed * (action[:, -1].clamp(-1.0, 1.0) + 1.0),
            released,
            released * (1.0 - torch.tanh(home_distance / p.HOME_STD)),
            released * (1.0 - torch.tanh(home_distance / p.HOME_FINE_STD)),
            torch.sum(torch.square(action - prev_action), dim=1),
            torch.sum(torch.square(joint_vel), dim=1),
            (1.0 - margin_to_limit / p.LIMIT_BAND).clamp(0.0, 1.0).sum(dim=1),
            (p.TABLE_MARGIN - gripper_pos[:, 2]).clamp(min=0.0) / p.TABLE_MARGIN,
        )
        for index, value in enumerate(terms):
            self.terms[:, index] = value
