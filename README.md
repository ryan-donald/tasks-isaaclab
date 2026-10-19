[![Isaac Lab](https://img.shields.io/badge/Isaac_Lab-v3.0.0--EA-76B900?logo=nvidia&logoColor=white)](https://github.com/isaac-sim/IsaacLab/tree/v3.0.0-EA)
[![Robot](https://img.shields.io/badge/Robot-SO--ARM101-1f6feb?logo=github&logoColor=white)](https://github.com/TheRobotStudio/SO-ARM100)

## SO-ARM101 tasks for Isaac Lab

Isaac Lab tasks for the SO-ARM101, an open-source robot arm. I train policies on these tasks and deploy them onto the real robot. The tasks observe and command the arm in the same normalized [-100, +100] joint space the real servos use, and model the servos' latency, speed limit and encoder resolution.

# Tasks

Every task runs on Isaac Sim PhysX by default. Add `physics=newton_mjwarp` to the command line to run it on Newton (MJWarp) instead. A `-Play` task is the evaluation variant of its task: fewer envs, no observation noise, less domain randomization, and the action delay pinned to its measured value.

| Family | Task IDs |
|---|---|
| Reach | `Ryan-Reach-SO-ARM101-Normalized-v0` (and `-Play`), `-Matched` (library-neutral agent configs for benchmarking), `-Direct` (direct-workflow port, same MDP) |
| Lift | `Ryan-Lift-Cube-SO-ARM101-Normalized-Settle-v0` (and `-Settle-Play`): lift to a goal with a bonus for holding still there; `Ryan-Lift-Cube-SO-ARM101-Normalized-v0` (and `-Play`): the base lift |
| Push | `Ryan-Push-Cube-SO-ARM101-Normalized-v0`, `Ryan-Push-T-SO-ARM101-Normalized-v0` (each with `-Play`) |

# Layout

The package follows the Isaac Lab 3.0 external-project template:

```
src/ryan_tasks/
  robots/            the SO-ARM101 config and its asset files in data/
  mdp/               the servo and encoder model shared by reach, lift and push
  tasks/<family>/    one folder per task family: mdp/, a base cfg where there is one,
                     and config/so_arm101/ with the task cfgs, registrations and
                     agent configs
tests/               registration test
```

# Installation

- Install Isaac Lab, found [here](https://github.com/isaac-sim/IsaacLab).
- Install this package into Isaac Lab's environment from the repository root: `uv pip install --python <IsaacLab>/.venv/bin/python -e .`

# Usage

`import ryan_tasks.tasks` registers the tasks. The package also declares an `isaaclab.tasks` entry point. Launch Isaac Sim (`AppLauncher`) before importing it, since the task configs import USD modules.

Train with [ryan_ppo](https://github.com/ryan-donald/ppo):

```
python -m ryan_ppo.isaaclab.train --task Ryan-Reach-SO-ARM101-Normalized-v0 --num_envs 4096
python -m ryan_ppo.isaaclab.train --task Ryan-Reach-SO-ARM101-Normalized-v0 physics=newton_mjwarp
```

The tasks also register agent configs for rsl_rl (all tasks) and for rl_games, sb3 and skrl (reach). Isaac Lab 3.0-EA's own `train.py` does not load external task packages, so to train with those libraries, launch `AppLauncher`, `import ryan_tasks.tasks`, then call `isaaclab_rl.entrypoints.run_train_cli()`.

# License

BSD 3-Clause, see [LICENSE](LICENSE). The robot model and meshes keep their own license; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
