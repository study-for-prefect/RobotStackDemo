# RobotStackDemo

UR5 + RealSense D435i 的积木识别、抓取与堆叠演示项目。

## 当前目标与开发状态（2026-07-14）

本项目当前的最终目标不是完成一个固定场景的单步演示，而是从桌面上一堆任意散落、
可能互相遮挡或挤在一起的积木开始，闭环完成两个主要任务：

1. `organize_blocks`：把全部积木按颜色分类，每种颜色占一个目标区域并排成一行；
2. `build_house`：从散落积木中取材，搭成四个方块支撑、一个屋顶和一个三角顶的六角色房子。

两类任务共享同一条恢复原则：**能安全抓取并推进任务就先抓；当前目标不能抓时必须清障，
不能因为一次选择不可抓、碰撞预检失败或 VLM 重复动作就直接结束。** 清障优先抓走可抓的
障碍物；没有可靠抓取角度时，才从障碍物侧面空处下降并水平推动约 3–5 cm。整理任务中，
抓走障碍物时直接把它放进自己的同色行；房子任务中则把障碍物移到安全临时区。推到其他
未保护散乱积木属于可恢复接触，推桌面、支撑物、已完成结构或 protected 对象仍是硬拒绝。

当前整理任务已经在实机上连续完成过红块和蓝块的抓取、搬运、放置与动作后重新观测，
但**尚未完整跑完一次全场颜色分类，也尚未开始最终房子闭环实机验收**。最近一次整理实机
暴露了两个安全漏检：蓝块只有约 4° 的狭窄抓取角仍被接受；蓝块放置时张开的夹爪撞到
第一步已放好的红块。代码已增加对应修复和回归测试，但按用户要求尚未做修复后的下一次
实机验证。因此不能宣称当前代码已经能够无人干预完整完成任务。

整理任务当前新增的关键行为：

- YOLO 负责检测框和三维几何；低置信、漏检或类别不可靠时由 VLM 复核类别，几何坐标仍以
  RGB-D/TF 为准。
- 每轮对所有目标行外对象做物理抓取扫描，不严格按颜色顺序；有可抓对象时，VLM 即使选到
  不可抓对象、`stop` 或 `reobserve`，代码也改选当前可靠可抓对象。
- 真实抓取必须有至少 10° 连续无碰撞 yaw 区间，并从区间内部取角度；窄缝边界角不执行。
- 颜色目标区域首次生成后在同一任务内保持不变，不随每次观测漂移。代码在同色区域内生成
  有效空槽；小于 15 mm 的搬动不算整理进展。
- 放置不仅检查积木本体是否重叠，还用张开 GF225 的两根实体手指检查下降、释放位置是否会
  碰到所有当前对象，包括前几步已经放好的颜色行；被挡住时在同色区域内改找安全位置。
- 动作后目标行判断允许最多 3 mm 的观测足迹抖动，避免刚放好的方块因检测 yaw 波动又被抓走。
- 当前全部不可抓时，不等待 8 次相同 VLM 输出：进入四个基坐标方向、两种腕角的有界推障
  搜索，每个候选仍必须通过语义、工作区、GF225 扫掠体和 MoveIt plan-only。

实机约束和当前参数：

- 工作区使用 `base_link` 米制坐标：`x=0.235..0.65 m`、`y=-0.10..0.40 m`；最终颜色区
  使用 `config/workspace_bounds.json` 中独立的 `organize_layout_bounds`。
- 每次观测前回到 `config/rectangle_ready_pose.json` 标准关节位，并现场刷新
  `base_link <- camera_color_optical_frame` 和 `base_link <- tool0`。
- 主平移、接近物体和标准位复位默认速度/加速度比例为 `0.08/0.08`；安全高位的末端 Z 轴
  预旋转默认是其 3 倍，即 `0.24/0.24`，可用 `--pre-rotate-velocity` 和
  `--pre-rotate-acceleration` 显式覆盖。
- 整理放置默认增加 10 mm 释放间隙，避免夹爪触桌。
- RTX 3090 当前调试功耗上限保持 250 W。此前 300 W 仍发生过无 OOM/Xid/panic 日志的硬重启，
  完成稳定验收前不要恢复 370 W。
- 默认模型使用 `qwen3-vl:8b-instruct`；30B 对照应使用
  `qwen3-vl:30b-a3b-instruct`，不要用会耗尽 thinking 预算的 `qwen3-vl:30b` 标签。

机器人、MoveIt、相机和感知服务由桌面 `Start_Robot_Stack.desktop` 启动。任务工作流只运行
下面的 pipeline；不要为任务调试修改或重复启动已经稳定工作的机械臂/相机驱动文件。

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

### 三个任务的运行指令

桌面启动器已经启动机械臂、MoveIt、相机和感知服务后，在新终端先加载 ROS 环境：

```bash
cd /home/wxm/code/RobotStackDemo
source /opt/ros/humble/setup.bash
source /home/wxm/ros2_ws/install/setup.bash
```

推荐使用以下三条自然语言指令。叠积木必须明确给出底层和逐层颜色关系；整理与房子由
任务关键词路由到各自独立的合同、规划和完成条件。

```text
叠积木：以红色积木为底，把绿色积木放到红色上面，再把蓝色积木放到绿色上面，再把黄色积木放到蓝色上面
整理积木：按颜色整理积木
搭房子：搭一个房子
```

实机 MoveIt 仅规划，不运动机械臂或夹爪：

```bash
# 红→绿→蓝→黄四色堆叠
python3 tools/workflows/stack_demo_pipeline.py \
  --instruction "以红色积木为底，把绿色积木放到红色上面，再把蓝色积木放到绿色上面，再把黄色积木放到蓝色上面" \
  --model qwen3-vl:8b-instruct --moveit-plan-only \
  --output-dir runtime/stack_blocks_plan_only

# 按颜色整理
python3 tools/workflows/stack_demo_pipeline.py \
  --instruction "按颜色整理积木" \
  --model qwen3-vl:8b-instruct --moveit-plan-only \
  --output-dir runtime/organize_blocks_plan_only

# 六角色房子：四个正方形 + 一个屋顶 + 一个三角形
python3 tools/workflows/stack_demo_pipeline.py \
  --instruction "搭一个房子" \
  --model qwen3-vl:8b-instruct --moveit-plan-only \
  --output-dir runtime/build_house_plan_only
```

完整实机执行使用对应的同一条命令，把 `--moveit-plan-only` 改为 `--execute --yes`，
并使用新的输出目录。例如完整四色堆叠：

```bash
python3 tools/workflows/stack_demo_pipeline.py \
  --instruction "以红色积木为底，把绿色积木放到红色上面，再把蓝色积木放到绿色上面，再把黄色积木放到蓝色上面" \
  --model qwen3-vl:8b-instruct --execute --yes \
  --output-dir runtime/stack_blocks_execute
```

`--execute` 默认不授权真实清障。只有需要且已确认允许机械臂推开/移走障碍物时，才额外
添加 `--execute-push-clearing`。30B 对照测试只需把模型改为
`qwen3-vl:30b-a3b-instruct`；不要使用 thinking 变体 `qwen3-vl:30b`。

从散落积木执行完整整理或搭房子时，本项目当前目标要求清障始终开启：

```bash
# 按颜色整理：能抓先分类，不能抓则抓走/侧推障碍后重新观察
python3 tools/workflows/stack_demo_pipeline.py \
  --instruction "按颜色整理积木" \
  --model qwen3-vl:8b-instruct \
  --execute --yes --execute-push-clearing \
  --output-dir runtime/organize_blocks_execute

# 六角色房子：先完成整理实机验收后再运行
python3 tools/workflows/stack_demo_pipeline.py \
  --instruction "搭一个房子" \
  --model qwen3-vl:8b-instruct \
  --execute --yes --execute-push-clearing \
  --output-dir runtime/build_house_execute
```

当前阶段不要直接运行第二条房子实机命令：严格先完成颜色整理，再单独调房子，不能同时修改
两个任务。每次实机都使用新的输出目录并保留完整 runtime，遇到问题先读最新现场文件再改。

动作安全检查使用 `config/workspace_bounds.json`。该文件必须明确写出 `base_link`、
米制单位以及 `xmin/xmax/ymin/ymax/zmin/zmax` 六个方向；也可用
`--workspace-bounds-json` 指向重新标定后的文件。当前默认值结合 D435i/桌面平面标定与
实机确认边界（`xmax=0.65 m`、`y=-0.10..0.40 m`），不从积木检测框推断。每次在线观测和重新观测都会把这份边界注入
scene state，文件缺失、frame/unit 错误或边界倒置都会在动作规划前停止。

`--moveit-plan-only` 会调用机器人端真实 MoveIt 检查首动作，但会移除轨迹执行和夹爪参数；
它与仅生成计划 JSON 的默认 dry-run 不同，也不会像 `--execute` 那样驱动 UR5。
`--moveit-plan-only` 与 `--execute` 互斥，同时出现会在任何动作前立即拒绝。

Ollama 各策略上下文有固定硬上限，不会根据 prompt 体积自动扩展。`--vlm-num-gpu 0`
可在显卡供电诊断期间强制 CPU 推理；默认 `-1` 仍由 Ollama 自动选择 GPU。
语义任务默认使用 `qwen3-vl:8b-instruct`；质量对照使用
`--model qwen3-vl:30b-a3b-instruct`。不要用同名的 thinking 标签执行严格 JSON
动作规划，它可能把整个输出预算消耗在 thinking 而不生成 content。

本机 RTX 3090 曾在模型推理期间发生没有 OOM/Xid/panic 日志的硬复位。实机调试期间
先执行 `sudo nvidia-smi -pl 250` 并确认 `nvidia-smi` 显示 250W；该限制在重启或
驱动重载后可能需要重新设置。三个任务稳定前不要直接恢复默认 370W。

明确包含底层和逐层“上面”关系的指令会自动路由到 `stack_blocks` 兼容实现；只有在
指令本身无法明确路由但需要强制进入该实现时，才手工添加 `--legacy-linear-stack`。
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

### Ollama Thinking 与调用重试

所有 stack order、任务合同、grounded plan、动作和场景分析调用统一经过
`robot_scene_pipeline/ollama_policy_client.py`。Qwen3 在默认
`--vlm-think-mode auto` 下发送 `think=true`；Qwen2.5-VL 不发送该字段，但两者响应都兼容
`message.thinking` 存在或缺失。

如果 Qwen3 已生成 thinking、但 content 为空或不是合法 Schema JSON，系统会保留原消息历史并执行
`think=false` 的 Finalization Call。`done_reason=length` 只扩大生成预算并重新进行 reasoning，
不会进入任务或动作失败账本。HTTP、超时、非法信封和空消息属于 Backend Retry，也不会被伪造成 stop。

四种计数器相互独立：

- `Backend Attempt`：连接、HTTP、超时、空模型消息；
- `Budget Retry`：生成长度不足或 JSON 截断；
- `Order Attempt`：四颜色实例绑定语义错误或重复绑定；
- `Action Attempt`：取得合法动作 JSON 后的引用、语义、几何、碰撞和 MoveIt 重规划。

相关参数：`--vlm-think-mode`、`--vlm-num-ctx`、`--vlm-num-predict`、
`--vlm-finalizer-num-predict`、`--vlm-read-timeout-sec`、`--vlm-keep-alive`、
`--vlm-max-backend-retries`、`--vlm-max-budget-retries` 和 `--unload-model-after-task`。
每个调用在 `ollama_calls/` 下保存完整 request、response、thinking、content 和 diagnostics；
任务根目录保存 `model_runtime_diagnostics.json`。

默认生成预算按调用复杂度分级：task contract 为 2K，action 为 3K，stack/grounded 为 4K，
orientation analysis 为 8K；上下文分别为 8K、12K、16K、24K。遇到明确截断时按
4K→6K→8K→12K 渐进扩容，避免正常调用直接占用 12K–20K 输出预算。

### 任务路由与当前协议

代码只根据指令确定任务家族：整理/分类指令选择独立的
`ORGANIZE_BLOCKS_CONTRACT_SCHEMA`，搭房子选择 `BUILD_HOUSE_CONTRACT_SCHEMA`，明确堆叠指令进入
stack workflow。整理任务提示词不携带任何房屋本体或六角色定义。

房屋 grounded plan 的 VLM 输出只包含当前 `role_bindings`、`orientation_observations`、reason 和
confidence。label、中心和尺寸由代码根据 `object_ref/track_id` 回填；固定六步
`canonical_house_assembly_steps()` 由代码注入，模型不能声明 placed/completed。
`concave`、`concave rectangle` 和 `concave_rectangle` 均归一为 `concave_rectangle`。

明确的红绿蓝黄堆叠使用 `stack_binding_v2`：颜色槽位由指令确定，同色实例仍由 VLM 选择。
OrderFingerprint 是四个颜色对应的 track 组合。每色只有一个合法实例时使用
`stack_binding_deterministic_unique_fallback`；存在多个组合而模型未选择时返回
`stack_binding_selection_failed`，不会随机绑定。

任何动作规划前都必须存在代码提供的 `table_bounds` 或 `workspace_bounds`。缺失时直接返回
`WORKSPACE_CONFIGURATION_MISSING`，不会归类为 VLM 失败，也不会让模型猜测边界。

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
