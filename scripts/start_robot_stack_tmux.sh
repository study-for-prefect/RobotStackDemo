#!/usr/bin/env bash
set -euo pipefail

SESSION="robot_stack"
PROJECT="$HOME/code/RobotStackDemo"
TF_JSON="/tmp/scene_tf_base_color_optical.json"

UR_IP="192.168.3.101"
KINEMATICS="$HOME/my_robot_calibration.yaml"

# 二选一：
# static = 使用下面手写的 static_transform_publisher
# easy   = 使用 easy_handeye2 publish.launch.py
HAND_EYE_MODE="static"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not installed. Run: sudo apt install -y tmux"
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach: tmux attach -t $SESSION"
  exit 0
fi

rm -f "$TF_JSON"

tmux new-session -d -s "$SESSION" -n ur_driver "
source /opt/ros/humble/setup.bash
if [ -f /home/wxm/realsense_ws/install/setup.bash ]; then
  source /home/wxm/realsense_ws/install/setup.bash
fi

if [ -f /home/wxm/ros2_ws/install/setup.bash ]; then
  source /home/wxm/ros2_ws/install/setup.bash
fi

ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur5 \
  robot_ip:=$UR_IP \
  kinematics_params_file:=$KINEMATICS
exec bash
"

tmux new-window -t "$SESSION" -n handeye "
source /opt/ros/humble/setup.bash
if [ -f /home/wxm/realsense_ws/install/setup.bash ]; then
  source /home/wxm/realsense_ws/install/setup.bash
fi

if [ -f /home/wxm/ros2_ws/install/setup.bash ]; then
  source /home/wxm/ros2_ws/install/setup.bash
fi

if [ \"$HAND_EYE_MODE\" = \"easy\" ]; then
  ros2 launch easy_handeye2 publish.launch.py \
    name:=my_eih_calib_camera_link
else
  ros2 run tf2_ros static_transform_publisher \
    0.06961816052731129 -0.014142581173124943 0.001529263786740309 \
    -0.7056477316624251 -0.014859437136558906 -0.7080993367909076 0.020875947018878998 \
    wrist_3_link camera_link
fi
exec bash
"

tmux new-window -t "$SESSION" -n moveit "
sleep 5
source /opt/ros/humble/setup.bash
if [ -f /home/wxm/realsense_ws/install/setup.bash ]; then
  source /home/wxm/realsense_ws/install/setup.bash
fi

if [ -f /home/wxm/ros2_ws/install/setup.bash ]; then
  source /home/wxm/ros2_ws/install/setup.bash
fi

ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5
exec bash
"

tmux new-window -t "$SESSION" -n camera "
sleep 3
source /opt/ros/humble/setup.bash
if [ -f /home/wxm/realsense_ws/install/setup.bash ]; then
  source /home/wxm/realsense_ws/install/setup.bash
fi

if [ -f /home/wxm/ros2_ws/install/setup.bash ]; then
  source /home/wxm/ros2_ws/install/setup.bash
fi

ros2 launch realsense2_camera rs_launch.py \
  align_depth.enable:=true \
  enable_color:=true \
  enable_depth:=true
exec bash
"

tmux new-window -t "$SESSION" -n tf_bridge "
sleep 8
source /opt/ros/humble/setup.bash
if [ -f /home/wxm/realsense_ws/install/setup.bash ]; then
  source /home/wxm/realsense_ws/install/setup.bash
fi

if [ -f /home/wxm/ros2_ws/install/setup.bash ]; then
  source /home/wxm/ros2_ws/install/setup.bash
fi

cd $PROJECT

while true; do
  echo \"[tf_bridge] checking base_link<-camera_color_optical_frame and base_link<-tool0 ...\"
  python3 tools/robot/tf_lookup_json.py \
    --base-frame base_link \
    --camera-frame camera_color_optical_frame \
    --tool-frame tool0 \
    --require-tool \
    --output $TF_JSON \
    --timeout 3.0 \
    --once
  sleep 1.0
done
"

tmux new-window -t "$SESSION" -n perception "
sleep 12
source /opt/ros/humble/setup.bash
if [ -f /home/wxm/realsense_ws/install/setup.bash ]; then
  source /home/wxm/realsense_ws/install/setup.bash
fi

if [ -f /home/wxm/ros2_ws/install/setup.bash ]; then
  source /home/wxm/ros2_ws/install/setup.bash
fi

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate yolo
cd $PROJECT

while [ ! -f $TF_JSON ]; do
  echo \"[perception] waiting for fresh TF JSON: $TF_JSON\"
  sleep 1.0
done

python3 -m robot_scene_pipeline.perception_server \
  --detector-weight models/yolo/weights/best.pt \
  --base-frame base_link \
  --camera-frame camera_color_optical_frame \
  --tf-json $TF_JSON
exec bash
"

tmux select-window -t "$SESSION:ur_driver"
tmux attach -t "$SESSION"
