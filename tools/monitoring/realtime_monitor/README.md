# Realtime Monitor

The public command remains `tools/monitoring/realtime_yolo_monitor.py`.

- `camera.py`: RealSense configuration
- `stream.py`: ROS-topic and legacy RealSense stream adapters
- `transforms.py`: TF JSON and coordinate conversion
- `display.py`: filtering and drawing helpers
- `app.py`: realtime loop

Default camera input is ROS 2 topics from an externally started RealSense
driver:

```bash
python3 tools/monitoring/realtime_yolo_monitor.py \
  --color-topic /camera/camera/color/image_raw \
  --depth-topic /camera/camera/aligned_depth_to_color/image_raw \
  --camera-info-topic /camera/camera/color/camera_info
```

Use `--camera-source realsense` only for the legacy direct `pyrealsense2`
path.
