# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pose command without IsaacLab's per-step error metrics."""

from __future__ import annotations

from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
from isaaclab.utils.math import combine_frame_transforms


class UniformPoseCommandNoMetrics(UniformPoseCommand):
    # skips the per-step pose error metrics, which only feed extras logging.

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # success tracking is updated by _update_metrics, so it is turned off too.
        self.metrics.clear()
        self._track_success = False

    def _update_metrics(self):
        pass

    def _debug_vis_callback(self, event):
        # the goal marker reads pose_command_w, which _update_metrics updated.
        pos_w, quat_w = combine_frame_transforms(
            self.robot.data.root_pos_w.torch,
            self.robot.data.root_quat_w.torch,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        self.pose_command_w[:, :3] = pos_w
        self.pose_command_w[:, 3:] = quat_w
        super()._debug_vis_callback(event)
