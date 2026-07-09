# RobotStackDemo

UR5 + RealSense D435i 的积木识别、抓取与堆叠演示项目。

## 目录结构

```text
RobotStackDemo/
├── config/                  # 检测器、机械臂预备位等配置
├── docs/                    # 标定与工作流说明
├── models/                  # 模型权重路径和训练结果
├── robot_scene_pipeline/    # 可复用的感知、几何、TF 与规划逻辑
├── tests/                   # 不依赖真实机器人运行的测试
├── tools/
│   ├── calibration/         # hover/TCP/XY 偏差标定
│   ├── data/                # RealSense 数据采集
│   ├── diagnostics/         # 感知稳定性诊断
│   ├── monitoring/          # 实时检测监控
│   ├── planning/            # 决策到执行计划的转换
│   ├── robot/               # ROS 2、TF、MoveIt、夹爪接口
│   └── workflows/           # 完整抓取、堆叠工作流
└── training/                # 数据转换和 YOLO 训练脚本
```

大型命令采用“薄入口 + 内部功能包”结构，原命令路径保持兼容。完整模块地图见
[项目结构说明](docs/PROJECT_STRUCTURE.md)。

## 常用入口

```bash
# 单帧感知
python3 -m robot_scene_pipeline.snapshot_pipeline --skip-llm

# 实时检测监控，默认订阅 ROS 相机话题
python3 tools/monitoring/realtime_yolo_monitor.py

# yaw-only 姿态生成和代码驱动记录检查
python3 tools/calibration/yaw_rotation_probe.py \
  --target-label-contains green \
  --output-dir runtime/yaw_rotation_probe/green \
  --execute --yes

# hover 标定
python3 tools/calibration/hover_tool_offset_calibration.py --label green

# XY 偏差采集和分析
python3 tools/calibration/xy_bias_diagnosis.py --help

# 两阶段视觉抓取
python3 tools/workflows/two_stage_visual_pick.py --help

# 闭环堆叠
python3 tools/workflows/stack_demo_pipeline.py --help

# 测试
python3 -m unittest discover -s tests
```

闭环堆叠的自动清障抓取策略见
[`tools/workflows/stack_demo/README.md`](tools/workflows/stack_demo/README.md)：
真实推障仍必须显式 `--execute --execute-push-clearing`，短距离 nudge
候选会先进入 MoveIt 预检，预检通过后才会执行。

清障决策可以显式切到 VLM/LLM policy：

```bash
python3 tools/workflows/stack_demo_pipeline.py \
  ... \
  --use-vlm-clearance-policy
```

当前架构分工：

- LLM/VLM = task rule policy，只从物理候选里选择 `candidate_id`。
- Code = YOLO + D435i 感知、object state、depth geometry、base frame 坐标、
  物理候选生成、碰撞/工作空间/夹爪/MoveIt 安全门、执行。
- MoveIt = IK、碰撞检测、轨迹规划、关节跳变检查。

开启 VLM 清障后，代码不再用几何分数、任务收益或 `target_yaw_gain`
选择最终清障动作；VLM JSON 解析失败或安全门拒绝时，会停在 fail-safe
状态等待重新观察或人工处理，不会自动回退到几何最高分。

涉及真实机械臂运动的命令默认只规划或采集；确认 UR5、MoveIt、TF、相机和夹爪状态后，再显式添加 `--execute`。yaw 角误差检测不要手动转动末端，使用 `yaw_rotation_probe.py --record-mode auto --execute` 让代码只改变目标 yaw 后自动记录。检验位姿的 `tool0` XYZ 写在 `config/yaw_rotation_probe_pose.json`，姿态由代码固定为 tool0 `+Z` 对准 base_link `-Z`，只允许 yaw 变化。

## 运行前 TF 检查

闭环堆叠依赖 `base_link <- camera_color_optical_frame` 和
`base_link <- tool0` 同时连通。`scripts/start_robot_stack_tmux.sh` 会先删除旧
`/tmp/scene_tf_base_color_optical.json`，再由 TF bridge 周期性刷新；perception
server 会等 fresh TF JSON 出现后再启动。`scripts/run_stack_task.sh` 在真正执行前
还会用 `/usr/bin/python3 tools/robot/tf_lookup_json.py --once --require-tool`
强制刷新一次，避免使用旧 TF。

如果只看到 `Known TF frames`，并且 frame 列表里只有相机静态链，或只有
`tool0_controller -> base` 这类 controller frame，但没有可连通的
`base_link <- tool0` 与 `base_link <- camera_color_optical_frame`，说明不是
LLM 没决策，而是 base 到相机/末端的 TF 树断开。先检查：

```bash
ros2 topic echo /joint_states --once
ros2 run tf2_ros tf2_echo base_link wrist_3_link
```

若这两项失败，先恢复 UR driver、robot_state_publisher / MoveIt、hand-eye
static TF，并确认所有终端的 `ROS_DOMAIN_ID` / RMW 配置一致。

堆叠执行中的抓取/放置 yaw 预旋转默认使用 `--pre-rotate-wrist-yaw-sign auto`。
MoveIt 会先尝试 wrist_3 的正负 yaw 映射；如果执行后 yaw 仍超过阈值，会在
当前位置再做一次原地 pose 姿态修正。修正仍无法满足
`--max-grasp-yaw-error-deg` 时才拒绝继续平移。

## 相机启动方式

项目内默认不直接打开 D435i 设备，而是订阅外部 ROS 2 RealSense 驱动发布的已对齐 RGB-D 话题。先在项目外终端启动相机驱动，例如：

```bash
ros2 launch realsense2_camera rs_launch.py \
  align_depth.enable:=true \
  enable_color:=true \
  enable_depth:=true
```

项目默认订阅：

```text
/camera/camera/color/image_raw
/camera/camera/aligned_depth_to_color/image_raw
/camera/camera/color/camera_info
```

如果你的 RealSense ROS 包发布的是旧命名空间，运行项目命令时改 `--color-topic`、`--depth-topic`、`--camera-info-topic`。旧的进程内 `pyrealsense2` 直连路径仍保留为显式兼容模式：`--camera-source realsense`。

在线话题订阅命令需要运行在能导入 `rclpy`、`sensor_msgs` 的 Python 环境中。Ubuntu + ROS 2 Humble + conda YOLO 环境建议在同一个终端里先加载 ROS，再进 conda：

```bash
source /opt/ros/humble/setup.bash
conda activate yolo
python tools/monitoring/realtime_yolo_monitor.py
```

如果仍提示 `No module named 'rclpy'`，说明当前 Python 还看不到 `/opt/ros/humble` 的 Python 包路径，先检查 `echo $PYTHONPATH` 是否包含 ROS Humble 的 `dist-packages`。

详细说明见：

- [Hover 与 XY 标定](docs/HOVER_XY_CALIBRATION.md)
- [LLM 堆叠入口](docs/LLM_STACK_BLOCKS.md)
- [感知流水线](robot_scene_pipeline/README.md)
- [项目结构说明](docs/PROJECT_STRUCTURE.md)
