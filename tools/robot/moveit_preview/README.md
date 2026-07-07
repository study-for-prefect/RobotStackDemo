# MoveIt Plan Preview

The public command remains `tools/robot/moveit_plan_preview.py`.

- `arguments.py`: CLI options
- `plan_io.py`: plan and calibration loading
- `orientation.py`: yaw, quaternion, pose, and tool-offset math
- `steps.py`: plan-step validation and expansion
- `trajectory.py`: trajectory and gripper helpers
- `execution.py`: MoveIt planning/execution primitives
- `push.py`: four-stage push-clearing preflight and execution
- `tf_node.py`: ROS 2 node and TF resolution
- `app.py`: top-level orchestration

`--max-joint-delta` checks the largest adjacent trajectory-point jump after
continuous joints are unwrapped. It is a discontinuity guard, not a limit on the
total smooth joint travel of a MoveIt plan.

For `--close-gripper-for-push`, a DH gripper close command may return status
`0` when the fingers contact an object before the exact close target. The push
path accepts that as a rigid-paddle close only when the reported position is
near the configured close position; open commands still require a successful
status.

Push clearing executes the open-gripper `pre_push` move before closing the
gripper as a rigid paddle. The closed gripper is then used only for `contact`,
`push_end`, and `retreat`. Before closing and descending to contact, the runtime
checks the actual tool orientation against the requested push orientation and
refuses the push if the tool is tilted beyond `--max-grasp-orientation-error-deg`.
