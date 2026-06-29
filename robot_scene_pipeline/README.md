# Robot Scene Pipeline

This directory contains the split single-frame robot scene pipeline:

- `snapshot_pipeline.py`: main orchestration.
- `ros_topic_capture.py`: ROS 2 RGB-D topic snapshot capture.
- `realsense_capture.py`: legacy direct RealSense RGB-D snapshot capture.
- `detector_runtime.py`: custom detector runtime.
- `depth_geometry.py`: depth-to-3D and scene state helpers.
- `tf_transform.py`: TF lookup and camera-to-base point transform.
- `grasp_yaw_search.py`: 连续抓取 yaw 搜索，输出可行/阻塞 yaw 区间。
- `grasp_obstruction_decision.py`: 根据 yaw 搜索结果生成 pick、push 或 replan 决策字段。
- `llm_scene_reasoner.py`: LLM prompt and Ollama call.

For hover/TCP offset and XY-bias calibration, see
`../docs/HOVER_XY_CALIBRATION.md`.

## Static TF

Publish the hand-eye calibration result in a terminal:

```bash
ros2 run tf2_ros static_transform_publisher \
  0.08599923 -0.03025062 0.00174066 \
  -0.706447 -0.01772374 -0.70735528 0.01634021 \
  wrist_3_link camera_link
```

Verify the TF chain after the robot driver or `robot_state_publisher` is running:

```bash
python -m robot_scene_pipeline.tf_transform \
  --base-frame base_link \
  --camera-frame camera_color_optical_frame
```

The default topic path uses `aligned_depth_to_color` plus `color/camera_info`,
so measured points are optical-frame XYZ from `camera_color_optical_frame`. Use
`base_link <- camera_color_optical_frame` directly and keep `--tf-point-mode
direct`. Use `camera_depth_optical_frame` only when you switch to an unaligned
depth image and depth camera_info. Do not manually convert optical XYZ to
`camera_link` unless you are intentionally running the legacy
`base_link <- camera_link` path.

## Offline Detector Test

```bash
conda run -n scene_graph_benchmark python -m robot_scene_pipeline.snapshot_pipeline \
  --image-in input_dir/1.jpg \
  --output-dir /tmp/robot_scene_pipeline_test \
  --device cpu \
  --skip-llm
```

## RealSense + LLM

Start the RealSense ROS driver outside this project before running online
perception. The pipeline subscribes to aligned color/depth topics by default:

```bash
ros2 launch realsense2_camera rs_launch.py \
  align_depth.enable:=true \
  enable_color:=true \
  enable_depth:=true
```

Default subscribed topics:

```text
/camera/camera/color/image_raw
/camera/camera/aligned_depth_to_color/image_raw
/camera/camera/color/camera_info
```

Use `--color-topic`, `--depth-topic`, and `--camera-info-topic` if your driver
uses a different namespace. The old in-process `pyrealsense2` capture path is
available only when explicitly requested with `--camera-source realsense`.
Online topic capture requires a Python environment that can import `rclpy` and
`sensor_msgs`; if YOLO runs in conda, expose the ROS 2 Python packages there.

Capture immediately:

```bash
conda run -n scene_graph_benchmark python -m robot_scene_pipeline.snapshot_pipeline \
  --instruction "把绿色方块放到红色方块左边" \
  --output-dir /tmp/robot_scene_pipeline
```

Wait for manual confirmation before capture:

```bash
conda run -n scene_graph_benchmark python -m robot_scene_pipeline.snapshot_pipeline \
  --capture-trigger enter \
  --instruction "把绿色方块放到红色方块左边" \
  --output-dir /tmp/robot_scene_pipeline
```

Use TF to add base-frame coordinates:

```bash
python3 tools/robot/tf_lookup_json.py \
  --base-frame base_link \
  --camera-frame camera_color_optical_frame \
  --tool-frame tool0 \
  --require-tool \
  --timeout 8 \
  --output /tmp/scene_tf_base_color_optical.json
```

```bash
conda run -n scene_graph_benchmark python -m robot_scene_pipeline.snapshot_pipeline \
  --use-tf \
  --base-frame base_link \
  --camera-frame camera_color_optical_frame \
  --instruction "把绿色方块放到红色方块左边" \
  --output-dir /tmp/robot_scene_pipeline
```

Estimate the tabletop plane and per-object point-cloud dimensions/yaw:

```bash
conda run -n scene_graph_benchmark python -m robot_scene_pipeline.snapshot_pipeline \
  --use-tf \
  --tf-json /tmp/scene_tf_base_color_optical.json \
  --camera-frame camera_color_optical_frame \
  --tf-point-mode direct \
  --estimate-tabletop \
  --skip-llm \
  --output-dir /tmp/robot_scene_geometry
```

## Integrated LLM + Two-Stage Pick

The main robot entry now combines LLM task reasoning with the tabletop-yaw and
second-snapshot pick workflow:

```bash
python3 tools/workflows/stack_demo_pipeline.py \
  --instruction "把绿色方块放到红色方块左边"
```

The integrated flow is:

1. Move to the configured ready pose and open the gripper.
2. Refresh `base_link <- camera_color_optical_frame` TF.
3. Capture the first RGB-D snapshot, estimate tabletop geometry/yaw, run the
   LLM, and compile `robot_execution_plan.json`.
4. Read the first planned `pick` target from the LLM plan and build a
   deterministic geometry pick plan for that object.
5. Pre-rotate the wrist from the detected object yaw, then move above it.
6. Capture a second RGB-D snapshot and correct only base-link XY. Preserve the
   first plan's yaw and Z.
7. Execute the corrected pick while keeping the object held.
8. Execute the remaining LLM plan, such as `place_relative`.

The default integrated TCP offset is `--tcp-offset-tool -0.015 0 0.15`,
meaning the gripper center is 1.5 cm along tool0 -X and 15 cm along tool0 +Z.
Other defaults include `--pick-target-lift-m 0.010`, `--release-gap-m 0.010`,
and negative wrist pre-rotation. The controller
accepts one LLM-selected `pick` per run; a later second `pick` is rejected
before motion because it would require another two-stage correction cycle.

The gripper is initialized and opened only during the ready stage. The pick and
remaining-plan MoveIt processes reconnect with `--skip-gripper-init`, preserving
the closed grip until the `place_relative` target is reached and its explicit
open command runs.

Important output files:

```text
/tmp/robot_scene_pipeline/private_scene_state.json
/tmp/robot_scene_pipeline/robot_execution_plan.json
/tmp/robot_scene_pipeline_second/private_scene_state.json
/tmp/robot_scene_pipeline/second_snapshot_xy_correction.json
/tmp/robot_scene_pipeline/robot_execution_plan_after_two_stage_pick.json
```

The deterministic rotation-only test remains available:

```bash
python3 tools/workflows/two_stage_visual_pick.py \
  --object-label "square green" \
  --execute \
  --yes
```

Use the two-stage workflow directly when you only need one deterministic pick.
