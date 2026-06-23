# MoveIt Plan Preview

The public command remains `tools/robot/moveit_plan_preview.py`.

- `arguments.py`: CLI options
- `plan_io.py`: plan and calibration loading
- `orientation.py`: yaw, quaternion, pose, and tool-offset math
- `steps.py`: plan-step validation and expansion
- `trajectory.py`: trajectory and gripper helpers
- `execution.py`: MoveIt planning/execution primitives
- `tf_node.py`: ROS 2 node and TF resolution
- `app.py`: top-level orchestration
