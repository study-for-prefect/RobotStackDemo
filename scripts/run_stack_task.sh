#!/usr/bin/env bash
set -uo pipefail

LOG_DIR="/home/wxm/code/RobotStackDemo/runtime/desktop_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_stack_task_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "===== Run Stack Task ====="
echo "Log: $LOG_FILE"
echo

pause_exit() {
  code=$?
  echo
  echo "===== 任务结束，退出码: $code ====="
  echo "日志文件: $LOG_FILE"
  echo
  read -p "按 Enter 关闭窗口..."
  exit "$code"
}
trap pause_exit EXIT

set +u
source /opt/ros/humble/setup.bash
if [ -f /home/wxm/realsense_ws/install/setup.bash ]; then
  source /home/wxm/realsense_ws/install/setup.bash
fi

if [ -f /home/wxm/ros2_ws/install/setup.bash ]; then
  source /home/wxm/ros2_ws/install/setup.bash
fi

set -u

if [ -f /home/wxm/miniconda3/etc/profile.d/conda.sh ]; then
  source /home/wxm/miniconda3/etc/profile.d/conda.sh
else
  echo "ERROR: 找不到 conda.sh"
  exit 1
fi

conda activate yolo || {
  echo "ERROR: conda activate yolo 失败"
  exit 1
}

cd /home/wxm/code/RobotStackDemo || {
  echo "ERROR: 找不到 RobotStackDemo 目录"
  exit 1
}

if [ ! -f /tmp/scene_tf_base_color_optical.json ]; then
  echo "ERROR: 找不到 /tmp/scene_tf_base_color_optical.json"
  echo "先双击 Start Robot Stack，等 TF bridge 和 perception 正常启动后再运行任务。"
  exit 1
fi

echo "检查并刷新 TF：base_link <- camera_color_optical_frame, tool0"
/usr/bin/python3 tools/robot/tf_lookup_json.py \
  --base-frame base_link \
  --camera-frame camera_color_optical_frame \
  --tool-frame tool0 \
  --require-tool \
  --output /tmp/scene_tf_base_color_optical.json \
  --timeout 3.0 \
  --once || {
    echo
    echo "ERROR: TF 未就绪。通常是 UR 动态关节 TF 没有发布，或 ROS_DOMAIN_ID/RMW 环境不一致。"
    echo "请先确认：ros2 topic echo /joint_states --once"
    echo "再确认：ros2 run tf2_ros tf2_echo base_link wrist_3_link"
    exit 1
  }

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="runtime/stack_push_execute_${RUN_ID}"

python3 tools/workflows/stack_demo_pipeline.py \
  --instruction "以红色积木为底，把绿色积木放到红色上面，再把蓝色积木放到绿色上面，再把黄色积木放到蓝色上面" \
  --output-dir "$OUTPUT_DIR" \
  --memory-json runtime/stack_push_execute/scene_memory.json \
  --detector-weight models/yolo/weights/best.pt \
  --tf-json /tmp/scene_tf_base_color_optical.json \
  --base-frame base_link \
  --camera-frame camera_color_optical_frame \
  --tool-frame tool0 \
  --execute \
  --execute-push-clearing \
  --yes

echo
echo "输出目录：$OUTPUT_DIR"
