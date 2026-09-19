[![Isaac Lab](https://img.shields.io/badge/Isaac_Lab-v3.0.0--EA-76B900?logo=nvidia&logoColor=white)](https://github.com/isaac-sim/IsaacLab/tree/v3.0.0-EA)
[![Robot](https://img.shields.io/badge/Robot-SO--ARM101-1f6feb?logo=github&logoColor=white)](https://github.com/TheRobotStudio/SO-ARM100)

## IsaacLab Custom Tasks

This is my repository containing my custom tasks for Isaac Lab. Currently, mainly reach tasks with the SO-ARM101, an opensource robot arm. I use these tasks to train policies within the Isaac Lab framework, and then deploy them onto the real robot for my own personal research  (and fun).

# Installation

- To install this package, simply run 'python -m pip install -e source/ryan_tasks' in the base directory.
- Afterwards, this task can be used in the same method as any other IsaacLab task.
- To use this in your code, simply add the import "import ryan_tasks" in your training scripts.
- The 'Ryan-Reach-SO-ARM101-Normalized-v0' task will be now be available to train your agents with like any other IsaacLab task.