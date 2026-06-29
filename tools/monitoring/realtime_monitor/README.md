# Realtime Monitor

The public command remains `tools/monitoring/realtime_yolo_monitor.py`.

- `camera.py`: RealSense configuration
- `stream.py`: ROS-topic and legacy RealSense stream adapters
- `transforms.py`: TF JSON and coordinate conversion
- `display.py`: filtering and drawing helpers
- `app.py`: realtime loop

Default camera input is ROS 2 topics from an externally started RealSense
driver:

Start the TF JSON bridge in a ROS 2 Python terminal first. Keep it running.
Use one explicit path on the same Ubuntu machine that runs realtime detection:

```bash
mkdir -p ~/code/RobotStackDemo/tmp

python3 tools/robot/tf_lookup_json.py \
  --base-frame base_link \
  --camera-frame camera_color_optical_frame \
  --tool-frame tool0 \
  --require-tool \
  --timeout 8 \
  --output ~/code/RobotStackDemo/tmp/scene_tf_base_color_optical.json
```

The monitor reads `/tmp/scene_tf_base_color_optical.json` by default. This is
Linux `/tmp`, not the project `tmp/` directory. Passing the same explicit
`--tf-json` path to both commands avoids Mac/Ubuntu/project temp-directory
mixups:

```bash
python3 tools/monitoring/realtime_yolo_monitor.py \
  --color-topic /camera/camera/color/image_raw \
  --depth-topic /camera/camera/aligned_depth_to_color/image_raw \
  --camera-info-topic /camera/camera/color/camera_info \
  --tf-json ~/code/RobotStackDemo/tmp/scene_tf_base_color_optical.json
```

By default the monitor requires a fresh `base_link<-camera_color_optical_frame`
JSON because the default depth topic is aligned to the color image and uses
`color/camera_info`. Use `camera_depth_optical_frame` only if you switch to an
unaligned depth image with depth camera_info. The monitor exits with a clear
error if the JSON is missing, stale, or written for the wrong child frame. Use
`--allow-missing-tf` only when intentionally checking detection/depth without
base coordinates.

Use `--camera-source realsense` only for the legacy direct `pyrealsense2`
path.
