# Robot Scene Pipeline

This directory contains the split single-frame robot scene pipeline:

- `snapshot_pipeline.py`: main orchestration.
- `realsense_capture.py`: RealSense RGB-D snapshot capture.
- `detector_runtime.py`: custom detector runtime.
- `depth_geometry.py`: depth-to-3D and scene state helpers.
- `tf_transform.py`: TF lookup and camera-to-base point transform.
- `llm_scene_reasoner.py`: LLM prompt and Ollama call.

For real TCP calibration, tabletop plane validation, and object size/yaw
acceptance criteria, see `docs/TCP_TABLETOP_YAW_CALIBRATION.md`.

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
  --camera-frame camera_link
```

The `--camera-frame` must match the coordinate convention of the 3D points from
RealSense deprojection. If the calibration uses ROS optical-frame axes, publish
and use an optical frame such as `camera_color_optical_frame`.

## Offline Detector Test

```bash
conda run -n scene_graph_benchmark python -m robot_scene_pipeline.snapshot_pipeline \
  --image-in input_dir/1.jpg \
  --output-dir /tmp/robot_scene_pipeline_test \
  --device cpu \
  --skip-llm
```

## RealSense + LLM

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
conda run -n scene_graph_benchmark python -m robot_scene_pipeline.snapshot_pipeline \
  --use-tf \
  --base-frame base_link \
  --camera-frame camera_link \
  --instruction "把绿色方块放到红色方块左边" \
  --output-dir /tmp/robot_scene_pipeline
```

Estimate the tabletop plane and per-object point-cloud dimensions/yaw:

```bash
conda run -n scene_graph_benchmark python -m robot_scene_pipeline.snapshot_pipeline \
  --use-tf \
  --tf-json /tmp/scene_tf_base_camera.json \
  --tf-point-mode optical-to-camera-link \
  --estimate-tabletop \
  --skip-llm \
  --output-dir /tmp/robot_scene_geometry
```

## Integrated LLM + Two-Stage Pick

The main robot entry now combines LLM task reasoning with the tabletop-yaw and
second-snapshot pick workflow:

```bash
bash tools/run_robot_scene_final.sh "把绿色方块放到红色方块左边"
```

The integrated flow is:

1. Move to the configured ready pose and open the gripper.
2. Refresh `base_link <- camera_link` TF.
3. Capture the first RGB-D snapshot, estimate tabletop geometry/yaw, run the
   LLM, and compile `robot_execution_plan.json`.
4. Read the first planned `pick` target from the LLM plan and build a
   deterministic geometry pick plan for that object.
5. Pre-rotate the wrist from the detected object yaw, then move above it.
6. Capture a second RGB-D snapshot and correct only base-link XY. Preserve the
   first plan's yaw and Z.
7. Execute the corrected pick while keeping the object held.
8. Execute the remaining LLM plan, such as `place_relative`.

The default integrated calibration values are `--tool-offset-base -0.015 0 0`,
`--pick-target-lift-m 0.005`, and negative wrist pre-rotation. The controller
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
python3 tools/two_stage_visual_pick.py \
  --object-label "square green" \
  --execute \
  --yes
```

Set `INTEGRATED_TWO_STAGE_PICK=0` when invoking `tools/run_robot_scene_final.sh`
to use the legacy single-snapshot execution path.
