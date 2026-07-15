# RobotStackDemo

RobotStackDemo 当前只提供两个最终任务：

- `organize_blocks`：把初始稳定观测中的全部积木放入各自颜色区域。
- `build_house`：用六个固定角色搭房子。

主入口是 `tools/workflows/stack_demo_pipeline.py`。旧的开放式 VLM 动作生成、通用
`task_contract`、`grounded_task_plan`、线性堆叠兼容入口和快照内 LLM 动作编译均已删除。
单帧快照工具现在只负责感知，不再生成机器人动作。

## 架构

```text
D435i RGB-D
  -> YOLO、对齐深度、内参、最新实时 TF
  -> ClutterSceneState(base_link, scene_revision, track_id)
  -> 代码扫描抓取 yaw 并生成 PhysicalActionEdge
  -> QwenTargetSelector 只选择 target_option_id
  -> QwenEdgeSelector 只选择 candidate_id
  -> 最终安全门再次检查不可变 edge
  -> UR5/GF225 最多执行一条 edge
  -> D435i 重新观测、显式 track 重绑定和结果验证
  -> 新状态重新规划
```

代码分为四层：

1. 共享感知与物理能力：RGB-D、YOLO、TF、三维几何、GF225、MoveIt、执行和结果验证。
2. 共享杂乱场景取物：`ClutterExtractionPlanner` 生成直接抓取、抓走阻挡物、推动和中转边。
3. 任务专用规划：`OrganizePlanner` 与 `HousePlanner` 分别维护状态、放置和完成谓词。
4. 无状态 Qwen 选择：Qwen3-VL 只能从代码给出的 ID 中选择，不能生成动作参数。

每轮只规划并执行一条真实边，不预测执行后的多步几何。动作后必须用最新观测重新生成边。

## 状态与身份

`ClutterSceneState` 明确记录 `scene_revision`、`base_link` 时间戳、当前对象、初始
`expected_tracks`、可见/缺失/完成/protected track、区域、最近结果和禁止 fingerprint。

- detector `object_id` 只属于当前 `scene_revision`。
- `object_ref` 形如 `scene_7:obj_3`，跨 revision 必须失效。
- 跨帧身份只使用 `track_id` 和一对一几何/语义重绑定；歧义绑定不会生成动作边。
- 单帧漏检不会删除 `expected_tracks`，存在 `missing_expected_tracks` 时不能完成任务。

## 代码生成的物理边

动作类型是固定枚举：`pick_place`、`extract_then_place`、`extract_to_staging`、
`regrasp_for_orientation`、`pick_away_blocker`、`nudge_blocker`、`place_house_role`、
`repair_structure` 和 `reobserve`。

抓取扫描覆盖 `[0°, 180°)`，默认步长 5°。抓取不要求沿物体长边、短边或检测 yaw，
安全斜角抓取可以成为候选。代码检查 GF225 开口、有效接触长度、中心偏差、指尖、掌部、
下降和抬升通道；连续安全区间至少 10°，最终 yaw 从区间内部选择。

清障只针对实际阻塞目标抓取扫掠的 track。代码枚举 `±x/±y`、离散距离和安全腕角，
检查接触侧预推下降、GF225 分段扫掠、终点、工作区、颜色目标区和 protected 结构。
松散邻接触只允许出现在水平推动阶段。没有预计净空收益的推动不会进入候选。

放置使用代码生成的槽位或房屋角色位姿。`place_pose` 表示预期最终物体位姿；代码根据抓取时
物体相对夹爪 yaw 计算释放夹爪 yaw。`release_pose` 在目标上方 10 mm，MoveIt 预检与开爪执行
使用该夹爪姿态。张开夹爪下降、
掌部、持物运输、释放和退回均在 Qwen 调用前预检，并在选中后由最终安全门复核。

## Qwen3-VL 协议

默认模型来自 `config/stack_demo_planner.json`：`qwen3-vl:8b-instruct`。配置固定
`temperature=0`、`think=false`、`stream=false` 和有界输出预算。

每次调用都是独立单轮请求：短 system 规则、一个当前 user JSON、最新干净 RGB 和只标短
detector ID 的图。不会追加旧 assistant 回答、旧图片或对话历史。唯一一次 JSON 格式修复
也是全新的短请求，不含 assistant 消息和旧图片。

目标选择只能返回：

```json
{
  "selected_target_option_id": "target_...",
  "reason_codes": ["directly_graspable", "releases_other_objects"]
}
```

边选择只能返回：

```json
{
  "selected_candidate_id": "edge_...",
  "backup_candidate_ids": ["edge_..."],
  "reason_codes": ["direct_task_progress", "lowest_structure_risk"]
}
```

未知、重复、空、过期或候选外 ID，以及任何 `task_complete`、对象或动作参数字段都会被拒绝。
第二次格式仍无效、连接失败、超时或预算失败均记录为 `policy_invalid_output`，不会静默改选，
也不会解释为完成。单一目标或单一边可跳过 Qwen，但日志会如实写
`single_feasible_target_option` 或 `single_feasible_edge`。

## organize_blocks

`OrganizeTaskState` 只包含整理语义：预期/可见/缺失/完成/未解决 track、颜色区域、安全槽、
占用、当前结果和最近失败。代码计算颜色区域、槽位、占用和完成状态；最终 yaw 不要求精确。

只有最新观测确认全部 expected track 位于正确颜色区域、没有缺失、没有非法重叠、杂乱区没有
未完成对象时，`task_complete` 才为 true。当前没有可行边不等于完成。

## build_house

房屋角色固定为：

- `left_support_lower`
- `right_support_lower`
- `left_support_upper`
- `right_support_upper`
- `roof`
- `triangle_top`

左右下支撑没有固定先后顺序。同一方块会同时形成所有合法的 `track × role` 目标选项，
不会按 detector ID 或数组顺序绑定左右角色。上层支撑依赖同侧下层；屋顶依赖两个稳定上层；
三角顶依赖已验证屋顶。完成结构自动进入 `protected_tracks`。

屋顶状态显式记录凹槽面、正反面、长轴、抓取姿态、左右覆盖、中心偏差、支撑余量和稳定性；
代码用最新观测重新计算支撑间距、高差、覆盖、中心和接触。三角顶显式记录尖端、底边、
face、目标 yaw、底边接触、质心投影、支撑余量和屋顶相对位姿。姿态证据不足时只生成安全
中转/重新观测路径，不假定 wrist yaw 能翻面，也不接受自然语言“翻转”。

当前确定性感知必须提供可靠的凹槽面和三角形朝向证据后，直接最终放置边才会生成；否则保持
在中转/重新抓取流程。这部分新架构尚未经过实机验证。

## 抓取后重新观测

抓取执行被拆为接近、下降、闭合、抬升到观察高度、fresh observation 和验证。目标同一 track
被观测到明显抬升/位移，或目标消失且其他 track 证明视图连续（可靠夹爪状态只能作辅助）时，
才允许运输。仅因目标在新视角消失不会判定抓取成功。

抓取失败立即跳过运输和放置并记录 fingerprint。放置后再次观测，只有目标区域或房屋结构
谓词通过才记录成功。推动同样用实际 track 位移、目标净空增益和 protected 稳定性验证。

末端 D435i 可能无法直接看到抬升后夹爪内的物体；因此当前实现依赖原位置变化、track 位移、
其他对象的视图连续性和可选夹爪状态。证据不足会安全失败，不会假装具有夹爪内视觉检测。

## 配置与实机约束

- 工作区唯一来源：`config/workspace_bounds.json`，`base_link` 下 `x=0.235..0.65 m`、
  `y=-0.10..0.40 m`。
- GF225、抓取、推动、速度、10 mm 释放间隙和房屋参数：
  `config/stack_demo_planner.json`。
- RTX 3090 自动功耗上限为 250 W；systemd unit 和脚本均不设置 300 W。
- D435i 安装在末端，每次场景获取前刷新实时 TF；动作后必须重新观测。
- 高位预旋转使用 0.24，近物体移动使用 0.03；MoveIt 执行器先抬到安全高度再预旋转。
- MoveIt 成功不是 GF225 安全证明；代码侧指尖、掌部和分阶段扫掠检查不可省略。
- 实机驱动由外部维护；本入口不会启动或重复启动 UR5、RealSense、MoveIt 或 GF225 驱动。

## 运行

查看真实参数：

```bash
python3 tools/workflows/stack_demo_pipeline.py --help
```

完全离线、无相机/MoveIt/Ollama、不会运动的 dry-run：

```bash
python3 tools/workflows/stack_demo_pipeline.py \
  --task-type organize_blocks \
  --offline-scene-state tests/fixtures/new_arch_single_red_scene.json \
  --output-dir /tmp/robot_stack_new_arch_dry_run \
  --no-image
```

使用已运行的感知服务和 Ollama，但不执行 MoveIt 或夹爪：

```bash
python3 tools/workflows/stack_demo_pipeline.py \
  --task-type organize_blocks \
  --instruction "按颜色整理积木" \
  --output-dir runtime/organize_blocks_dry_run
```

调用现有 MoveIt 做 plan-only，不发送轨迹或夹爪命令：

```bash
python3 tools/workflows/stack_demo_pipeline.py \
  --task-type build_house \
  --instruction "搭一个房子" \
  --moveit-plan-only \
  --output-dir runtime/build_house_plan_only
```

实机命令保留如下，但本次重构未执行：

```bash
python3 tools/workflows/stack_demo_pipeline.py \
  --task-type organize_blocks \
  --instruction "按颜色整理积木" \
  --execute --yes --execute-push-clearing \
  --output-dir runtime/organize_blocks_20260715_manual
```

`--execute` 与 `--moveit-plan-only` 互斥；实机必须同时给出 `--yes`。只有选中 nudge 时才还需
`--execute-push-clearing`。离线状态与 `--execute` 的组合会在动作前拒绝。

保留/新增的 CLI 分组包括：任务与输出、统一配置、离线 scene/mock policy、步数上限、
dry-run/plan-only/execute、Qwen/Ollama 有界参数、感知服务/YOLO、TF、ready pose 和 GF225 端口。
已删除旧 `--legacy-linear-stack`、stack order/base object/decision JSON、task contract/grounded plan、
开放式 action 重规划、旧 resume 和散落的抓取/推动几何覆盖参数。

## 日志

每个 `cycle_NNN_revision_R/` 至少包含：

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

不适用项写明确 `skipped_reason`。另有 `task_completion.json`，模型调用的底层诊断位于周期目录的
`ollama_calls/`。

## 验证状态

本次只运行离线 dry-run、编译和 mock/单元测试，没有执行真实 UR5 运动、夹爪闭合、推动或
机器人驱动启动。新架构尚待按“单目标抓取门控 -> 单次颜色放置 -> 受控清障 -> 完整整理 ->
左右支撑 -> 屋顶中转/重抓 -> 三角顶”的顺序实机验证。

更详细的模块和日志说明见
[`tools/workflows/stack_demo/README.md`](tools/workflows/stack_demo/README.md)，感知说明见
[`robot_scene_pipeline/README.md`](robot_scene_pipeline/README.md)，交接事实见 [`HANDOFF.md`](HANDOFF.md)。
