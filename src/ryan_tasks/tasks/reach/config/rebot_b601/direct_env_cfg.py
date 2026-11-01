# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct-workflow B601-DM reach: the SO-101 direct task with the robot swapped."""

from isaaclab.utils.configclass import configclass

from ryan_tasks.robots.rebot_b601 import POS_QUANTUM
from ryan_tasks.tasks.reach.config.so_arm101.direct_env_cfg import (
    SoArm101ReachDirectEnvCfg,
)

from .joint_pos_env_cfg import use_b601


@configclass
class B601ReachDirectEnvCfg(SoArm101ReachDirectEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        use_b601(self, self.ee_pose)
        self.pos_quantum = POS_QUANTUM
