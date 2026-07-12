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

# 语义任务闭环：搭房子或整理积木
python3 tools/workflows/stack_demo_pipeline.py --help

# 测试
python3 -m unittest discover -s tests
```

默认工作流支持 `build_house` 与 `organize_blocks`。它先让 VLM 生成一次不含检测
`object_id` 的 `task_contract`，再在每个 `scene_revision` 由 VLM 生成临时
`grounded_task_plan` 和单步动作；完成状态由重新感知后的几何谓词计算。

`build_house` 只有一种合法结构：四个不同正方形组成左右两列、每列两层；一个
`concave_rectangle`（缺失时才降级为普通 `rectangle`）横跨上层；一个三角形位于
屋顶中央。合法角色固定为 `left_support_lower`、`right_support_lower`、
`left_support_upper`、`right_support_upper`、`roof` 和 `triangle_top`，不得退化为旧版三块房子。

所有传给 VLM 的当前帧物体统一使用 `object_ref=scene_<revision>:obj_<detector_id>`，
跨帧身份使用 `track_id`。语义任务动作输出 `selected_object_ref` 或
`selected_track_id`；解析为当前检测对象后，兼容适配层内部才使用 `selected_object_id`。
房子 `pick_place` 或
`pick_reorient_place` 还包含
`role_id`；整理 `pick_place` 包含 `group_id` 和 `target_region_id`。语义校验通过后，
唯一的兼容适配层再把已解析的 `selected_object_id` 复制为旧 MoveIt/清障模块使用的
`object_id`；两个字段冲突会直接拒绝，不会改选对象或修改目标位姿。

```bash
# 搭房子：四个正方形 + 一个屋顶 + 一个三角形
python3 tools/workflows/stack_demo_pipeline.py --instruction "搭一个房子"

# 按颜色整理成行
python3 tools/workflows/stack_demo_pipeline.py --instruction "按颜色整理积木"
```

旧线性堆叠仅作为兼容路径保留，必须显式启用 `--legacy-linear-stack`。
详细自主 VLM 策略见
[`tools/workflows/stack_demo/README.md`](tools/workflows/stack_demo/README.md)：
真实推障仍必须显式 `--execute --execute-push-clearing`，短距离 nudge
动作会先进入代码碰撞检查与 MoveIt 预检，预检通过后才会执行。

每次重新观察都会搜索全部合法房子角色组合，并按当前谓词重建动态保护对象、
保护区域和保护关系。整理任务使用带 yaw 的完整二维足迹检查区域、重叠和边界间距，
并分别计算 rows、columns、grid。`--execute` 与 `--offline-scene-state`（以及 mock、
recorded 等离线感知源）不能组合，检查发生在任何机器人初始化之前。

凹槽屋顶错误面朝上时，代码根据当前/目标四元数和抓取变换生成安全高度上的
roll/pitch 翻面轨迹；仅绕 yaw 不能改变正反面。旋转量由观测计算，45°只可能是某次
样例结果，不是固定参数。轨迹使用四元数 SLERP 分段，每个中间姿态必须通过 MoveIt
plan-only 后才允许执行。三角形方向由三顶点内角、深度/点云几何和 VLM 语义融合；
接近 90° 的顶点不会被默认当作目标尖端。真实 UR5 测试必须先 dry-run 和 plan-only。

以下内容描述 `--legacy-linear-stack` 兼容路径；它不参与默认的房子/整理任务状态机。

## 旧线性堆叠兼容路径

旧线性堆叠同样是 VLM-first 决策；已无作用的 `--use-vlm-action-policy` 兼容开关已删除：

- VLM = 初始结构/堆叠顺序，以及每轮场景问题、动作类型、操作物体、方向、距离和预期收益。
- 初始结构只输出一个权威字段 `full_stack_order`；代码自动派生底座和后续放置顺序，
  不再比较三个等价字段。
- 每轮动作输出 `strategy_id`、`scene_problem`、`action_type`、`object_ref`、
  `object_track_id`、`object_label`、`object_center_base_m`、`target_object_ref`、
  `target_object_track_id`、`target_object_label`、
  `target_object_center_base_m`、
  `contact_side`、`direction_base`、`distance_m`、`gripper_yaw_rad`、
  `safe_place_center_base_m`、
  `predicted_scene_benefit`、`risk_assessment`、`reason`、`confidence`。
- Code = YOLO + D435i 感知、TF 坐标转换、object state、base_link 坐标、
  碰撞/抓取/放置可行性、安全验证和执行，不生成动作候选或任务语义结论。
- MoveIt = IK、碰撞检测、轨迹规划、关节跳变检查。

VLM 输入不会包含相机内参；相机内参只在感知模块里用于像素和深度到三维坐标转换。
大模型输入使用快照图、带编号图、已经计算好的 `base_link` 坐标、物体尺寸、
bbox、任务目标、堆叠进度和记忆。同色/同 label 物体会以实例组形式列出，VLM 必须用
`object_ref`、`track_id`、bbox 和 `base_link` 中心区分，不能只按颜色猜。

每轮输入不包含代码计算的 `grasp_feasible`、`blocking_objects`、候选动作、候选评分、
推荐推向或推荐距离。`current_plan_focus` 只是先前 VLM 堆叠计划的上下文，不会在验证器中
强制 `pick` 操作对象与它相等。VLM 选定 `pick`/`pick_away` 物体后，代码才搜索抓取 yaw；
VLM 选定 `nudge` 后，代码才检查终点和扫掠路径。

VLM 必须把所选引用对应的检测 label 和 `base_link` 中心原样回填。代码根据当前场景重新
计算实际语义，用来阻止“reason 说绿色、track 实际指向黄色”的 grounding 错误。
初始堆叠输出同样通过 `object_bindings` 绑定 id、label 和中心，但代码不替 VLM解释任务顺序。

VLM 输出后，代码会解析当前 `object_ref`/`track_id`，校验 base/placed/locked/protected 状态、推动距离
范围、`base_link` 单位方向、`pick_away` 临时放置点、保护结构终点区域、
保护结构扫掠碰撞和 MoveIt 预检。普通 `pick` 也必须在真实执行前通过 plan-only 预检。
`nudge` 的接触侧、方向、距离和夹爪 yaw 均由 VLM 决定。代码使用考虑
`tool0→TCP`、指长和末端 yaw 的分段 OBB 做保守粗筛：尖端 `0–0.025 m / 0.025 m`，
上指 `0.025–0.070 m / 0.062 m`，壳体 `0.070–0.150 m / 0.112 m`。这些高度是安装后的
默认值，真实硬件必须重新实测校准。抓取模型使用两根实体手指，中间 `0.049 m` 开口不是
碰撞实体。桌面、工作区边缘、支撑物和 protected 结构始终硬拒绝；未保护散乱积木仅在
侵入、被动位移、出界和倾覆风险均通过配置阈值时允许 `controlled_contact`，随后必须重新感知。
非法 JSON、未知 object id、不安全方向/距离、保护结构碰撞或 MoveIt 不可行会生成结构化反馈，
连同原场景再次发送给 VLM；代码不会生成替代动作。每轮动作决策前，代码会先保证当前 snapshot
内 object id 唯一；若检测结果出现重复 id，会写 `scene_state_unique_object_ids.json`
记录重分配。
失败动作使用基于 `track_id` 的 `ActionFingerprint`：方向离散，距离按 `0.005 m`、yaw 按
`5°` 量化，reason/confidence 不参与。失败指纹进入硬黑名单，在几何、碰撞和 MoveIt 前拒绝。
第 3 次可禁用连续失败的动作类型，第 4 次强制更换高层策略；只有至少两个唯一指纹和两个
唯一策略均失败后才允许 safe-stop，否则要求重新观察。动作后递增 `scene_revision` 并执行
一对一 track 重绑定，旧 `object_ref` 立即失效。

VLM 决策日志：

- `vlm_stack_decision_input.json`
- `vlm_stack_decision_raw.json`
- `vlm_stack_decision_validated.json`
- `vlm_action_decision_input.json`
- `vlm_action_decision_raw.json`
- `vlm_action_decision_validated.json`
- `vlm_action_safety_report.json`
- `vlm_action_attempt_XX_input.json` / `output.json` / `validation.json`
- `autonomous_action_history.json`
- `action_fingerprint.json` / `failure_ledger.json` / `replanning_context.json`
- `track_assignment.json` / `track_history.json` / `role_binding_history.json`
- `gripper_collision_profile.json` / `controlled_contact_evaluation.json`

旧的代码侧动作发现/评分路径不再进入堆叠 workflow；初始堆叠 JSON 也不再由颜色规则
解析器覆盖或修复。代码只检查 schema、引用 id 和执行所需几何，任务理解由 VLM 负责。

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

堆叠执行中的抓取/放置 yaw 预旋转默认使用 `--pre-rotate-wrist-yaw-sign negative`。
如需排查 wrist_3 yaw 映射，可显式改为 `auto` 让 MoveIt 尝试正负两个映射。
如果执行后 yaw 仍超过阈值，会在当前位置再做一次原地 pose 姿态修正。修正仍无法满足
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
