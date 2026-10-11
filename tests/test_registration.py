# Copyright (c) 2024-2026, Ryan Donald
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Every task registers, and every entry point it names resolves."""

import importlib
from pathlib import Path

import gymnasium as gym
import pytest

import ryan_tasks.tasks  # noqa: F401

TASKS = sorted(i for i in gym.registry if i.startswith("Ryan-"))


def _resolve(entry_point: str):
    module, _, attr = entry_point.partition(":")
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as e:
        # modules that import Kit at import time need the running app.
        if e.name.split(".")[0] in ("carb", "omni", "pxr"):
            pytest.skip(f"{module} needs the Kit app: {e.name}")
        raise
    if attr.endswith((".yaml", ".yml")):
        path = Path(mod.__file__).parent / attr
        assert path.is_file(), path
        return path
    return getattr(mod, attr)


def test_tasks_registered():
    assert len(TASKS) > 0


@pytest.mark.parametrize("task_id", TASKS)
def test_entry_points_resolve(task_id):
    spec = gym.spec(task_id)
    if isinstance(spec.entry_point, str):
        _resolve(spec.entry_point)
    for key, value in spec.kwargs.items():
        if key.endswith("entry_point") and isinstance(value, str):
            _resolve(value)
