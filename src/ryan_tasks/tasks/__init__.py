# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task registrations, loaded through the ``isaaclab.tasks`` entry point."""

from isaaclab_tasks.utils import import_packages

import_packages(__name__, ["utils", ".mdp"])
