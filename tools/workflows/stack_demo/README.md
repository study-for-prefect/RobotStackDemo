# Stack Demo Workflow

`tools/workflows/stack_demo_pipeline.py` 是唯一主入口，最终任务只支持
`organize_blocks` 和 `build_house`。旧 `stack_blocks`、通用 task contract、grounded task
plan 和开放式 VLM 动作入口已删除，不存在新旧 planner 并行运行。

## 每轮数据流

```text
最新 D435i RGB-D + 实时 base_link TF
  -> ClutterSceneState
  -> 任务层给出当前合法 track/role
  -> ClutterExtractionPlanner 生成物理可行首步边
  -> QwenTargetSelector 从 TargetOption ID 中选目标
  -> QwenEdgeSelector 从 PhysicalActionEdge ID 中选边
  -> FinalSafetyGate 对原边重新验证
  -> dry-run / MoveIt plan-only / 执行至多一条边
  -> fresh observation
  -> 验证真实结果并重新构建状态
```

代码不会预测动作后的多步场景。抓取、放置或推动后都用真实新观测生成下一轮候选。

## 0715retry4 根因与修复

原始运行目录是 `runtime/organize_blocks_plan_only_20260715`，不是不存在的
`runtime/0715retry4`。原目录已有 `snapshot.jpg`、detector 结果、有效
`base_link <- camera_color_optical_frame` TF 和 `tabletop_geometry.json`，但缺少
`private_scene_state.json`。安全复现的服务端正文为：

```text
'types.SimpleNamespace' object has no attribute 'instruction'
```

`depth_geometry.build_private_state` 把感知状态错误地绑定到任务参数 `instruction`，而
`perception_server` 的请求 namespace 故意不含任务字段；旧 urllib 客户端又丢掉了 HTTP 500
正文。现在感知状态完全 task-independent，错误正文会原样进入结构化日志。完整调查记录在
`runtime/0715retry4_analysis.json`，原失败目录没有被修改。

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `app.py` | 配置、任务选择、生命周期、异常和输出目录 |
| `arguments.py` | 当前 CLI；不暴露动作几何覆盖参数 |
| `commands.py` | TF、MoveIt、GF225 和场景采集命令边界 |
| `perception_client.py` | 感知 server 健康检查、严格请求契约和结构化 HTTP 错误 |
| `live_integration.py` | ROS/topic 前检、无运动参数守卫、关节差值和集成报告 |
| `common/config.py` | 加载统一 planner、工作区配置 |
| `common/scene_state.py` | revision-scoped 对象和持久 track 状态 |
| `common/action_edges.py` | 固定动作枚举、不可变物理边和 fingerprint |
| `common/action_validation.py` | 最终安全门 |
| `common/action_execution.py` | 分阶段执行、fresh observation 和门控 |
| `common/result_verification.py` | 抓取、放置、推动结果验证 |
| `clutter/grasp_edges.py` | 0--180 度连续抓取 yaw 搜索 |
| `clutter/edge_generation.py` | 直接抓取、局部清障、中转和任务放置边 |
| `clutter/path_safety.py` | 张开夹爪、持物运输和推动分段扫掠 |
| `clutter/target_options.py` | 仅从存在可行首步的目标构建 TargetOption |
| `clutter/extraction_planner.py` | 两级选择和最终安全门共享编排 |
| `policy/` | 无状态 Qwen client、严格 schema 和两个 selector |
| `organize/` | 颜色区域、槽位、状态和完成谓词 |
| `house/` | 六角色、姿态、结构、放置和完成谓词 |

共享层负责如何安全取出物体，任务层负责哪些物体/角色合法、最终放哪里以及何时完成。

## 感知参数契约

pipeline 每轮先用 `tf_lookup_json.py` 写最新 TF，再向 server 发送同一个绝对 `tf_json`。主要
衔接如下：

| 参数 | pipeline CLI | TF lookup | `/snapshot` | server / MoveIt |
| --- | --- | --- | --- | --- |
| 输出目录 | `--output-dir` | 不使用 | 当前 observation 的绝对路径 | server 只写该绝对路径 |
| TF 文件 | `--tf-json` | 写入该路径 | 绝对路径、必填 | server 验证并读取最新文件 |
| base frame | `--base-frame` | `--base-frame` | `base_frame` | server 固定配置比对；MoveIt `--base-link` |
| camera frame | `--camera-frame` | `--camera-frame` | `camera_frame` | 与 camera_info/source frame 和 server 固定配置比对 |
| tool frame | `--tool-frame` | `--tool-frame --require-tool` | 不作为点变换 frame | MoveIt `--end-effector` |
| TF 点模式 | `--tf-point-mode` | 不使用 | `tf_point_mode` | server 固定配置比对 |
| detector | weight/imgsz/iou/device CLI | 不使用 | 固定配置不在请求中重复修改 | `/health` 与 pipeline 严格比对 |
| threshold | `--score-thresh` | 不使用 | 每请求 `score_thresh` | snapshot 动态值 |

请求还包含 `request_id`、`capture_reason`、`scene_revision`、client timeout 和最大 TF 年龄。
server 验证 TF 文件存在、JSON 和矩阵有限、parent/child 正确、timestamp 新鲜；验证 RGB/depth/
camera_info frame 一致且时间偏差不超过配置。

`/health` 返回实际 topic、frame、TF mode、detector 固定参数、RGB-D readiness/seq/timestamp、PID
和 config fingerprint。固定配置不一致时客户端在 snapshot 前以 409
`perception_server_config_mismatch` 停止。结构化错误区分：

- 400：缺失/非法请求、TF 文件缺失/无效/过期、frame mismatch；
- 503：`camera_not_ready`、RGB-D 不同步；
- 500：`detector_failed`、`scene_processing_failed` 或不可预期内部错误。

fallback 默认禁止。显式 `--allow-snapshot-subprocess-fallback` 后才可回退，并记录原 server
错误和完整 fallback 命令；回退的 TF/frame/topic/detector 参数与 server 请求保持一致。

## 状态与身份规则

`ClutterSceneState.coordinate_frame` 固定为 `base_link`。对象至少带 detector ID、持久
`track_id`、revision-scoped `object_ref`、语义、三维中心/尺寸/yaw、邻居、边缘余量、阻挡侧、
安全侧、完成/protected/可见状态。

初始稳定观测建立 `expected_tracks`。后续单帧漏检只把 track 放入
`missing_expected_tracks`，不会减少任务总数。detector ID 不跨帧使用；跨帧只能通过 track 和
显式一对一重绑定。重绑定歧义时不生成可执行边。

## 物理边生成

动作类型只能是：

```text
pick_place              extract_then_place       extract_to_staging
regrasp_for_orientation pick_away_blocker         nudge_blocker
place_house_role        repair_structure          reobserve
```

每条边已经包含 grasp/approach/lift/place/release pose，或 push direction/distance/start/end、
任务角色、预期效果、净空收益、风险、protected track、预检和失败 fingerprint。Qwen 看不到可
修改参数的接口。

抓取 yaw 默认以 5 度扫描 `[0, 180)`，不要求沿检测 yaw 或物体边。代码验证开口、有效双指
接触、中心偏差、指尖、掌部、下降、抬升及 MoveIt。连续安全区间至少 10 度，并从区间内部
取角，不使用贴边角。单点角接触或窄安全区间不会成为边。

目标不可抓时，只分析真正占用其抓取区间或夹爪扫掠的直接阻挡物。清障枚举配置中的
`±x/±y`、3/4/5 cm 和安全腕角，可生成抓走、推动或安全中转。预推从接触侧空闲位置下降；
松散接触只允许水平推动阶段。无净空收益、进入目标颜色区、损坏 protected/已完成结构或
MoveIt plan-only 失败的候选不会交给 Qwen。

`place_pose` 是期望物体最终位姿。代码保存抓取时物体相对夹爪 yaw，并由它计算不可由 Qwen
修改的释放夹爪 yaw；`release_pose` 使用配置的额外 10 mm 释放间隙。放置边在选择前验证
张开 GF225 下降、掌部、持物运输、释放和退回路径。

## Qwen 选择协议

默认 `qwen3-vl:8b-instruct`、`temperature=0`、`think=false`、`stream=false`，上下文和输出
预算有界。每次请求是一个新的短 system + user 请求，只带最新 RGB、只标短 ID 的 overlay、
最新 revision/state、当前候选和有限失败。不会携带历史 assistant 回答、旧图片或依靠聊天
记忆保存机器人状态。

目标响应：

```json
{
  "selected_target_option_id": "target_...",
  "reason_codes": ["directly_graspable", "low_clearance_cost"]
}
```

边响应：

```json
{
  "selected_candidate_id": "edge_...",
  "backup_candidate_ids": ["edge_..."],
  "reason_codes": ["direct_task_progress", "lowest_structure_risk"]
}
```

解析拒绝未知/重复/空/过期 ID、候选外 backup、`task_complete`、object ID 和任何动作参数。
非法 JSON 只允许一次全新短请求进行格式修复。再次失败、Ollama 连接失败、超时或预算失败
记录为 `policy_invalid_output`，当前周期 reobserve/结束，绝不由代码静默换目标或动作。

单一目标或单一边会跳过相应模型调用，真实记录 `single_feasible_target_option` 或
`single_feasible_edge`，不会伪装成 Qwen 选择。

## 抓取后重新观测门控

执行器将抓取拆成接近、下降、闭合、抬升、fresh observation、验证。只有新观测支持同一
track 明显抬升/位移，或在视图连续证据及可选夹爪辅助证据支持下离开原位置，才继续运输。
夹爪状态不能单独证明成功；目标仅因末端相机视角变化而消失也不算成功。

失败会写 `grasp_failed`、yaw/pose/fingerprint，并跳过运输和放置。成功后才运输；放置后再次
观测并运行任务谓词。推动结果也以实际方向位移、净空收益和 protected 稳定性验证。

末端 D435i 可能看不到抬升后夹爪内部，因此当前实现采用保守多证据判断；证据不足时安全
失败。这不等同于已经具备夹爪内物体检测。

## OrganizePlanner

整理状态独立维护 expected/visible/missing/completed/unresolved track、颜色区域、安全槽、占用、
当前结果和最近失败。所有未完成合法对象参与目标比较，Qwen 能看到邻居数、最近邻、拥挤侧、
安全 yaw、清障成本、释放其他物体的收益、任务收益、风险和失败次数。

颜色区域、槽位、占用和完成均由代码计算。最终 yaw 不要求精确。只有最新观测确认所有预期
track 在正确颜色区域、无缺失、无非法重叠且杂乱区无未完成对象时才能完成；无边不等于完成。

固定区域都使用 `x=0.250..0.420 m`，Y 范围分别为红 `0.230..0.265`、绿
`0.275..0.310`、蓝 `0.320..0.355`、黄 `0.365..0.400 m`。每区宽 `0.035 m`，三个
通道均宽 `0.010 m`，并与默认初始杂乱区分离。完成判定同时检查颜色、中心和 yaw-aware 二维
占用足迹。颜色矩形不参与推动碰撞拒绝；空区域不保护，实际已完成物体才按尺寸加安全余量成为
保护体。推入自身颜色区域允许并增加任务收益，其他空颜色区域允许但提高风险分数。

实测 `tool0 -> GF225 grasp TCP` 为 `[0.0, 0.0, 0.16] m`。该值由共享工具几何常量约束，
planner config、stack CLI 与 MoveIt 参数必须一致；转换方向是从目标 TCP 位置减去旋转到
`base_link` 的 tool-frame offset，且每个 MoveIt 目标只应用一次。

候选生成对每个未完成 track 独立执行：先扫描 `[-90°, 90°)` 内全部抓取轴角，再扫描
`+x/-x/+y/-y` 四个推动方向和配置的全部距离/腕角。推动边同时记录 `push_direction` 与其
相反的 `contact_side`。GF225 无法从接触侧进入时只拒绝该方向；目标扫掠接触普通未完成
积木时会构建方向一致的 `chain_track_ids`，使用保守位移上界验证联合扫掠，只有碰到实际
已完成物体、固定障碍、工作空间边界或机器人排除区才硬拒绝。颜色矩形不参与碰撞拒绝。

每个 cycle 写出 `candidate_generation_summary.json` 和 `candidate_rejections.json`，因此原始
生成、几何拒绝、MoveIt 拒绝、物理边和目标选项数量都可追踪。空候选不会立即结束：同一任务
最多连续重新观测 3 次，每次重绑 track、重建 task state 并重新生成候选；只有连续空扫描且
场景几何签名未变化才停止。

连锁推动到达 `retreat` 阶段后，普通连锁物体按保守预测终点参与垂直退离碰撞检查，不继续用
推动前位置制造虚假碰撞。Qwen 只接收候选 ID、物理参数摘要、代码门结果和风险/进度指标；
完整姿态与碰撞明细仍保存在本地审计文件中，合法候选较多时也不会挤爆固定上下文。

两指夹爪轴角在几何、动作参数和 MoveIt 前统一规范为 `[-90°, 90°)`（例如 `145° -> -35°`）。
正常 pick-place 仍在抓取姿态垂直抬升后，于安全高度调整释放姿态。MoveIt 分别规划目标轴角
的 `base-180°/base/base+180°` 等价表示，按总关节变化、`wrist_3` 变化和 IK 分支代价选择
最小可行解；所有关键段继续执行 `wrist_3 start_goal <= 1.75 rad` 检查。

## HousePlanner

角色固定为：

```text
left_support_lower   right_support_lower
left_support_upper   right_support_upper
roof                 triangle_top
```

每轮先按代码前置条件产生合法的 `track × role` 组合，所以左右顺序不写死。上支撑依赖同侧
稳定下支撑；roof 依赖两个稳定上支撑、合法间距/高差；triangle 依赖已验证 roof。完成结构
自动成为 protected。

凹槽屋顶显式维护 groove face、face up、长轴、抓取姿态、覆盖、中心偏差、两侧余量和稳定性；
三角顶维护 apex/base direction、face、目标 yaw、底边接触、质心投影、支撑余量和 roof 相对
位姿。代码用最新观测验证结构。姿态不满足时走 staging -> fresh observation -> regrasp，绝不
假定 wrist yaw 能翻面。只有确定性感知给出可靠姿态证据时才生成最终放置边。

## 最终安全门

Qwen 选中后重新检查原 edge：revision/object ref、track/rebinding、任务前置条件、工作区、
GF225 指尖/掌部、下降/抬升/运输/放置/释放/退回、推动预接近/水平/终点、protected/房屋、
MoveIt plan-only 和 forbidden fingerprint。失败不会原地改 yaw、方向、距离或对象。

相反推动方向会得到不同 fingerprint。fingerprint 使用持久 track 和完整签名，不因普通
revision 变化而遗忘。

## 日志

每个 `cycle_NNN_revision_R/` 写：

```text
scene_state.json               task_state.json
target_options.json            physical_action_edges.json
qwen_target_request.json       qwen_target_response.json
selected_target.json           qwen_edge_request.json
qwen_edge_response.json        selected_edge.json
final_safety_gate.json         execution_result.json
post_grasp_verification.json   post_place_verification.json
action_history.json
```

不适用阶段写 `skipped_reason`；任务结果另写 `task_completion.json`。日志中的
`decision_source` 只使用有限真实枚举，不存在隐藏改选。

## 配置和运行

- `config/stack_demo_planner.json`：GF225、抓取、推动、速度、释放、Qwen 和任务参数。
- `config/workspace_bounds.json`：唯一工作区来源，`base_link` 下 x 0.235--0.65 m、
  y -0.10--0.40 m。

查看实际 CLI：

```bash
python3 tools/workflows/stack_demo_pipeline.py --help
```

完全离线 dry-run：

```bash
python3 tools/workflows/stack_demo_pipeline.py \
  --task-type organize_blocks \
  --offline-scene-state tests/fixtures/new_arch_single_red_scene.json \
  --output-dir /tmp/robot_stack_new_arch_dry_run \
  --no-image
```

在线感知但不运动：

```bash
python3 tools/workflows/stack_demo_pipeline.py \
  --task-type organize_blocks \
  --instruction "按颜色整理积木" \
  --output-dir runtime/organize_blocks_dry_run
```

使用已有 MoveIt 做 plan-only：

```bash
python3 tools/workflows/stack_demo_pipeline.py \
  --task-type organize_blocks \
  --instruction "按颜色整理积木" \
  --moveit-plan-only \
  --tcp-offset-tool 0 0 0.16 \
  --output-dir runtime/organize_blocks_region_tcp_fix_test
```

真实设备链路、严格无运动的集成模式：

```bash
python3 tools/workflows/stack_demo_pipeline.py \
  --task-type build_house \
  --instruction "用六个固定角色搭建房屋" \
  --live-integration-check \
  --max-task-steps 1 \
  --perception-server-url http://127.0.0.1:8765 \
  --tf-json /tmp/scene_tf_base_color_optical.json \
  --base-frame base_link \
  --camera-frame camera_color_optical_frame \
  --tool-frame tool0 \
  --tf-point-mode direct \
  --output-dir runtime/build_house_live_integration_20260715_resume3
```

该模式与 `--execute`、`--yes`、`--execute-push-clearing` 互斥，并强制 plan-only、单轮、禁用
ready pose/轨迹/夹爪/动作后模拟。`assert_non_actuating_command` 在每个 subprocess 前再次拒绝
已知执行和夹爪 flag。报告包含 `before_joint_state.json`、`after_joint_state.json`、
`joint_state_delta.json` 和 `integration_report.json`。

8765 是外部维护的日常服务，不应由 workflow 重启。需要隔离 server 时使用 8766 独立进程；
启动参数必须显式匹配 `/health` 所列 topic、base/camera/TF mode 和 detector 配置，测试后只停止
自己启动的 PID。

实机路径统一要求 `--execute --yes`，这两个显式参数也授权执行已通过全部门禁的 nudge；
`--execute-push-clearing` 仅作为旧命令兼容参数保留。离线/mock 输入和 `--execute` 的组合会在
机器人初始化之前被拒绝。

已删除旧 `--legacy-linear-stack`、stack order/base ID/decision JSON、task contract/grounded plan、
resume、开放式 VLM action 尝试次数，以及分散的抓取/推动几何覆盖参数。当前参数以 `--help`
为准，几何常量只从统一配置加载。

## 尚未实机验证

真实无运动检查已验证 D435i topic/快照、最新 TF、感知契约、无状态 Qwen target/edge 选择和
MoveIt plan-only。成功报告是
`runtime/build_house_live_integration_20260715_resume3/integration_report.json`，其中
`robot_motion_executed=false`、`gripper_command_executed=false`。organize 同场景报告在
`runtime/organize_blocks_live_integration_20260715_resume2/`：感知成功，但没有完整安全边，未把
无边当完成。

仍未验证真实末端相机抓取成功证据、GF225 斜角接触、实际清障、真实 pick/place、完整颜色
整理、六角色结构、凹槽面/三角尖端可靠感知、中转重抓或放置稳定性。不得把 plan-only 结果
解释为动作成功。
