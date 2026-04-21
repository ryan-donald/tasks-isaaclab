## IsaacLab Custom Tasks

Currently, the only task within this package is my 'Ryan-Reach-SO-ARM101-Normalized-v0'. This is a task with the SO-ARM101 with the goal of controlling the arm end-effector tip to a desired position in a three dimensional workspace. I have used this task to perform sim2real transfer from IsaacLab to a real-world SO-ARM101 robot. 

# Installation

- To install this package, simply run 'python -m pip install -e source/ryan_tasks' in the base directory.
- Afterwards, this task can be used in the same method as any other IsaacLab task.
- To use this in your code, simply add the import "import ryan_tasks" in your training scripts.
- The 'Ryan-Reach-SO-ARM101-Normalized-v0' task will be now be available to train your agents with like any other IsaacLab task.