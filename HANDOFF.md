# RobotStackDemo 交接记录

更新时间：2026-07-15。本文只描述当前代码、已验证事实和未完成的实机验证，不保留旧决策方案。

## 1. 当前分支

- 分支：`llm-decision-explore`
- 工作区保留未提交改动，未创建/切换分支，未 commit，未 push。
- 本次没有运行任何 `--execute`、`--yes`、真实夹爪闭合、UR5 运动或驱动启动命令。

## 2. 新架构数据流

```text
D435i 最新 RGB-D
  -> YOLO、对齐深度、内参、最新实时 TF
  -> ClutterSceneState(base_link, scene_revision, track_id)
  -> 代码生成当前 PhysicalActionEdge
  -> Qwen 选择 TargetOption ID
  -> Qwen 选择 PhysicalActionEdge ID
  -> 最终安全门复核原 edge
  -> UR5/GF225 最多执行一条 edge
  -> D435i fresh observation
  -> 验证抓取/推动/放置真实结果
  -> 新状态重新规划
```

四层边界是共享感知与物理能力、共享杂乱取物、任务专用规划、无状态 Qwen 候选选择。模型
没有动作参数生成接口，任务完成只由最新观测上的代码谓词产生。

## 3. 共享底层能力

继续使用现有 RGB-D、YOLO、深度/内参、实时 TF、点云尺寸/yaw、track 重绑定、MoveIt、UR5、
GF225 和分段 swept-volume 能力。新公共层统一提供：

- revision-scoped `object_ref` 和跨帧 `track_id`；
- 邻居、边缘余量、blocked/free side、目标/占用/空闲区域；
- GF225 指尖、掌部、下降、抬升、持物运输、释放、退回和推动扫掠；
- MoveIt plan-only adapter、不可变动作 edge、最终安全门；
- fresh observation 后的抓取、放置和推动多证据验证；
- 包含持久 track、角色、抓取 yaw/pose、带符号推动方向/距离和放置 pose 的 fingerprint。

物理常量来自 `config/stack_demo_planner.json`，工作区只来自
`config/workspace_bounds.json`。

## 4. ClutterExtractionPlanner

`ClutterExtractionPlanner` 被两个任务共享。每轮接收任务层的合法 `track` 或 `track × role`，
扫描 `[0, 180)` 抓取 yaw，生成直接抓取/放置边；不可抓时只识别占用实际抓取区间或扫掠的
直接阻挡物，再生成抓走、`±x/±y` 离散推动或安全中转边。

只有至少一条预检通过的首步边才形成 `TargetOption`。目标选择后只展开该目标已有 edge，边
选择后由最终安全门再次验证同一个不可变 edge。门失败不会改对象、yaw、方向或距离。

## 5. OrganizePlanner

`OrganizeTaskState` 独立维护 expected/visible/missing/completed/unresolved track、持久颜色、颜色
目标区域、安全槽、区域占用、当前动作结果和最近失败。颜色优先使用 detector 的
`visual_color`；只有形状类名也能进入整理状态。

代码负责颜色区域、非重叠槽位、张开夹爪下降空间和完成判定。安全斜角抓取可以直接放到同色
区域，不要求最终精确 yaw。全部 expected track 在最新观测中处于正确区域、无缺失、无非法
重叠且无 unresolved track 时才完成；无可行边不是完成证据。

## 6. HousePlanner

`HouseTaskState` 独立维护六角色：

```text
left_support_lower   right_support_lower
left_support_upper   right_support_upper
roof                 triangle_top
```

每轮生成所有合法 `track × role`，不按 detector ID/数组顺序绑定左右。上支撑依赖同侧稳定下
支撑；roof 依赖两个稳定上支撑；triangle 依赖已验证 roof。最新几何谓词验证中心、高度、
接触、间距、覆盖和稳定性。完成角色自动进入 protected，后续取物、推动和运输不得破坏。

## 7. Qwen 无状态接口

默认模型是 `qwen3-vl:8b-instruct`，`temperature=0`、`think=false`、`stream=false`、输出预算
有界。每次请求只有新的 system + user，最新 RGB、短 ID overlay、最新 revision/state、当前
候选和最近有限失败；不追加 assistant 历史或旧图片。

目标输出只允许：

```json
{"selected_target_option_id":"target_...","reason_codes":["..."]}
```

边输出只允许：

```json
{"selected_candidate_id":"edge_...","backup_candidate_ids":["edge_..."],"reason_codes":["..."]}
```

未知、空、重复、过期、候选外 ID 以及任何额外动作/object/task_complete 字段都会被拒绝。
非法 JSON 只允许一次新的 format-only 请求；再次失败、连接/超时/预算错误均记录
`policy_invalid_output`，不静默改选，不解释为完成。单一候选使用真实 single-feasible 日志。

## 8. 斜角抓取

抓取默认每 5 度扫描一次，不偏好物体长边、短边或 detector yaw。代码检查 GF225 开口、
中心偏差、有效双指接触、指尖、掌部、下降和抬升。安全区间至少连续 10 度，并从内部选角；
单点角接触和窄区间被拒绝。

物理边显式记录抓取 yaw、物体相对夹爪 yaw、期望放置物体 yaw 和由代码推导的释放夹爪 yaw，
避免斜角抓取时把物体 yaw 错当成腕部 yaw。

## 9. 抓取后重新观测门控

执行拆成接近、下降、闭合、抬升到观察高度、fresh observation、抓取验证。只有同一 track 的
明显抬升/位移，或原位置清空且视图连续/辅助夹爪证据支持时才运输。夹爪状态不能单独证明
抓取；仅因腕部相机视角变化而漏检也不算成功。

失败立即写 `grasp_failed` 和 fingerprint，跳过运输/放置。成功后才运输；放置后再次观测，
目标区域或房屋结构谓词失败则写 `place_failed`，不更新完成。推动也以实际方向位移、净空
收益和 protected 稳定性验证。

## 10. 凹槽矩形处理

roof 类别只从统一 house 配置读取，至少包含 `concave_rectangle`。状态显式记录 track、groove
face、face-up、长轴、当前抓取姿态、两侧覆盖、中心偏差、余量和稳定性。直接 roof 边必须
通过 groove face、长轴、间距/高差、两侧覆盖、中心、垂直接触、路径和退回检查。

代码不把 wrist yaw 当作翻面。姿态不合格时先放 orientation staging，fresh observation 后仅在
显式证据表明可重抓时形成 `regrasp_for_orientation`；最终仍必须由新观测验证。

## 11. 三角形处理

triangle 状态显式记录 apex/base direction、face、target yaw、base contact、质心投影、support
margin 和 roof relative pose。最终边要求尖端向上、底边有效接触、face 合法、质心落在支撑
范围、余量合法，并验证路径、释放和退回。姿态不足同样走 staging/reobserve/regrasp，不接受
自然语言翻转角度。

## 12. 保留的实机安全约束

- RTX 3090 自动功耗上限是 250 W。旧 300 W 设置曾发生硬重启；systemd 和脚本当前默认 250 W。
- `base_link` 工作区唯一配置：x 0.235--0.65 m，y -0.10--0.40 m。
- D435i 在末端；每次在线拍照前刷新 TF，动作/碰撞/人工移动后绝不复用旧坐标。
- detector ID 只在当前 revision 使用；跨帧只接受 track/rebinding。
- MoveIt 成功不代表 GF225 安全，代码侧全部指尖/掌部/路径检查必须保留。
- 高位预旋转速度 0.24，近桌面/物体速度 0.03；先到安全高度再预旋转。
- 额外释放间隙固定 10 mm。
- 推动从接触侧空闲预推位置下降；松散接触只允许水平阶段。
- 推动不得进入颜色目标区、破坏 protected 或完成房屋结构。
- 相反推动方向必须有不同 fingerprint。
- 模型、JSON、连接、超时和预算错误都不是成功或完成。

## 13. 已删除的旧架构

已删除开放生成 object/action/yaw/direction/distance/place 的 VLM policy、旧 stack order、通用 task
contract/grounded task plan、旧 scene memory、旧 action adapter/semantic validator、旧 silent
fallback/replan、旧快照 LLM action compiler、旧 perception VLM 补漏、线性 stack 入口和只服务旧
schema 的测试。`snapshot_pipeline` 现在只感知。

桌面 `scripts/run_stack_task.sh` 已改为显式 `organize_blocks|build_house`，不再传旧
`--memory-json` 或线性堆叠 instruction。

## 14. 新增文件

- `config/stack_demo_planner.json`
- `robot_scene_pipeline/object_semantics.py`
- `robot_scene_pipeline/schema_validation.py`
- `tools/workflows/stack_demo/common/{config,scene_state,action_edges,action_validation,action_execution,result_verification,cycle_logging,moveit_adapter,policy_images}.py`
- `tools/workflows/stack_demo/clutter/{grasp_edges,path_safety,edge_generation,target_options,extraction_planner}.py`
- `tools/workflows/stack_demo/policy/{schemas,qwen_client,target_selector,edge_selector}.py`
- `tools/workflows/stack_demo/organize/{state,placement,completion,planner}.py`
- `tools/workflows/stack_demo/house/{roles,orientation,structure,state,placement,completion,planner}.py`
- 上述五个包的 `__init__.py`
- `tests/new_arch_fixtures.py`
- `tests/test_new_scene_edge_policy.py`
- `tests/test_new_execution_organize.py`
- `tests/test_new_house_planner.py`
- `tests/fixtures/new_arch_single_red_scene.json`

## 15. 删除文件

生产代码删除：

- `config/task_semantics.json`、`docs/LLM_STACK_BLOCKS.md`
- `robot_scene_pipeline/{action_fingerprint,house_grounded_adapter,house_task_definition,llm_scene_reasoner,llm_stack_blocks,nudge_safety,organize_scope,orientation_assets,orientation_fusion,reorientation_planner,scene_memory,stack_binding,stack_state,task_action_adapter,task_action_validation,task_dynamic_protection,task_goal_evaluator,task_schemas,task_semantic_validation,vlm_action_policy,vlm_action_validation,vlm_perception_review,vlm_replanning,vlm_stack_policy,vlm_task_policy}.py`
- `tools/planning/decision_to_execution.py`
- `tools/workflows/stack_demo/{clearance_execution,observation_scope,pick,pick_preflight,placement,post_place_observation,push_clearing,push_context,push_flow,scene,target_recovery,task_execution,task_workflow,vlm_action,vlm_action_loop}.py`

删除的测试全部直接绑定以上旧类、旧 schema、旧 silent fallback 或旧线性 stack 行为；新测试
覆盖当前受限候选协议和保留的安全事实。具体删除列表可由 `git status --short` 审查。

## 16. 测试结果

当前完成的非实机验证：

```text
python3 -m compileall robot_scene_pipeline tools/workflows tests
  PASS
python3 -m unittest discover -s tests
  Ran 113 tests; OK
python3 tools/workflows/stack_demo_pipeline.py --task-type organize_blocks \
  --offline-scene-state tests/fixtures/new_arch_single_red_scene.json \
  --output-dir /tmp/robot_stack_new_arch_dry_run_final_20260715 --no-image
  exit 0; 15 个周期日志产物齐全
git diff --check
  PASS
```

测试只使用 mock Qwen、mock executor、mock scene、离线 geometry dry-run 和既有纯单元测试。

## 17. 尚未进行的实机验证

- 真实 D435i 在抬升观察位对抓取成功提供的证据质量；
- GF225 斜角/近角部双指接触和掌部余量；
- 真实 MoveIt 连续段与代码 swept-volume 的一致性；
- 真实 pick-away/nudge、推动净空收益和邻接触；
- 完整颜色区域占用与所有积木完成；
- 六角色房屋、protected 结构和局部修复；
- 凹槽面、三角 apex/base/质心的确定性感知；
- staging/regrasp 和最终结构稳定性。

## 18. 下一次实机测试顺序

1. 驱动由操作者单独启动并确认 250 W、TF、控制器、急停和空工作区。
2. 在线 observation-only，核对 revision、track 重绑定、RGB/短 ID overlay 和坐标。
3. 单个孤立方块 MoveIt plan-only，检查斜角 yaw、相对 yaw 和 release pose。
4. 单次真实抓取至观察高度，验证失败确实阻止运输。
5. 单次颜色放置和 post-place 完成谓词。
6. 一个受控 pick-away；之后再测单方向、最短距离 nudge。
7. 小规模后完整 `organize_blocks`。
8. 左右下支撑、同侧上支撑、protected 运输顺序。
9. roof 正确面直接放置，再测试 staging/regrasp；最后 triangle。
10. 完整 `build_house` 和失败后修复。每一步使用全新输出目录并人工审查日志。

## 19. 当前实际运行命令

本次实际执行并验证的是离线命令：

```bash
python3 tools/workflows/stack_demo_pipeline.py \
  --task-type organize_blocks \
  --offline-scene-state tests/fixtures/new_arch_single_red_scene.json \
  --output-dir /tmp/robot_stack_new_arch_dry_run_final_20260715 \
  --no-image
```

在线 dry-run 和 MoveIt plan-only 的当前入口：

```bash
python3 tools/workflows/stack_demo_pipeline.py \
  --task-type organize_blocks \
  --instruction "按颜色整理积木" \
  --output-dir runtime/organize_blocks_dry_run

python3 tools/workflows/stack_demo_pipeline.py \
  --task-type build_house \
  --instruction "搭一个房子" \
  --moveit-plan-only \
  --output-dir runtime/build_house_plan_only
```

实机入口仍要求 `--execute --yes`，nudge 额外要求 `--execute-push-clearing`。本次没有执行实机
入口。桌面脚本可接受 `scripts/run_stack_task.sh organize_blocks` 或 `build_house`，同样未运行。

## 20. 已知限制

- 腕部 D435i 通常不能直接看到抬升后夹爪内部；当前抓取验证是保守的多证据判断，证据不足会失败。
- `groove_face_state`、triangle apex/base/face、质心和相对位姿仍依赖感知输入提供可靠字段；当前
  YOLO/点云链尚未经过实机证明能稳定产生全部字段。
- 平面相对 yaw 已由代码计算；错误正反面没有被伪装成 wrist-yaw 翻面。当前没有已实机验证的
  通用空中 face-flip primitive，错误面可能在 staging 后保持不可完成。
- `orientation_regrasp_feasible` 必须来自明确的感知/几何证据；缺失时不会生成假定可行的翻转边。
- 路径碰撞使用保守几何和外部 MoveIt 分段 plan-only；实际执行中间状态仍需实机逐步验证。
- 重绑定歧义、missing expected track、policy 无效或无边会安全结束当前周期，不会自动宣称完成。
- 当前颜色集合和房屋类别/尺寸阈值来自统一配置，新增类别必须先更新配置和测试。
