# RobotStackDemo 交接记录

更新时间：2026-07-15。本文件只记录当前实现、实机事实、安全踩坑、实际验证和剩余限制；不保留
旧开放式 VLM 决策方案。

## 1. 当前分支与工作区

- 仓库：`/home/wxm/code/RobotStackDemo`
- 分支：`llm-decision-explore`
- 改动保留在工作区，未 commit、未 push、未创建或切换分支。
- 本轮没有运行 `--execute`、`--yes`、`--execute-push-clearing`、夹爪命令或机器人驱动命令。

## 2. 当前架构数据流

```text
D435i 最新 RGB-D
  -> YOLO、对齐深度、内参、最新实时 TF
  -> ClutterSceneState(base_link, scene_revision, track_id)
  -> 代码生成并预检 PhysicalActionEdge
  -> QwenTargetSelector 只选择 target_option_id
  -> QwenEdgeSelector 只选择 candidate_id
  -> FinalSafetyGate 复核同一不可变 edge
  -> 执行至多一条 edge
  -> D435i fresh observation
  -> 验证抓取/推动/放置真实结果
  -> 新状态重新规划
```

四层分别是共享感知与物理能力、共享杂乱取物、任务专用规划、无状态 Qwen 候选选择。模型没有
动作参数生成接口，task completion 只来自最新观测上的代码谓词。

## 3. 0715retry4 的准确根因

任务中写的 `runtime/0715retry4` 不存在。真实原始失败目录是
`runtime/organize_blocks_plan_only_20260715`，本轮未修改或覆盖它。该目录在 2026-07-15 15:58
写入 `failure_state.json`；RGB、detector、深度、TF 和桌面几何已成功，缺少最终
`private_scene_state.json`。

安全复现得到服务端原始错误：

```text
'types.SimpleNamespace' object has no attribute 'instruction'
```

直接根因有两个：

1. `depth_geometry.build_private_state` 仍访问任务字段 `args.instruction`；持久感知服务为 snapshot
   构造的是 task-independent `SimpleNamespace`，没有该字段。
2. 旧 client 没有读取 `urllib.error.HTTPError` 正文，因此用户只看到
   `HTTP Error 500: Internal Server Error`。

原始产物排除了相机未就绪、YOLO 失败、TF 失败、tabletop 失败、输出目录失败和 client timeout。
完整证据、旧/新参数和排除项在 `runtime/0715retry4_analysis.json`。

## 4. 感知 client/server 修复

- 感知私有状态不再读取 instruction；`scene_id` 与 `frame_id/coordinate_frame` 分离。
- 新增共享 `PerceptionSnapshotRequest`、`PerceptionServerHealth` 和结构化异常类型。
- client 每次显式发送绝对 output/TF 路径、frame/mode、revision/reason、超时和 TF 最大年龄。
- server 检查 TF 文件存在、JSON/4x4 有限、frame 匹配和 timestamp 新鲜。
- server 验证 RGB/depth/camera-info frame 和 timestamp 同步，不静默改 camera frame。
- client 保留 JSON 或非 JSON HTTP 正文；连接失败和超时有独立 error code。
- fallback 默认关闭；显式开启时保留原 server error，且 subprocess 参数与请求一致。

结构化错误分类：请求/TF 错误为 400，相机未就绪或 RGB-D 不同步为 503，detector/scene/意外
内部错误为 500。任何错误都不是 stop 或 task complete。

## 5. 参数衔接事实

| 参数 | CLI/来源 | TF lookup | snapshot request | server/MoveIt |
| --- | --- | --- | --- | --- |
| output | `--output-dir` | 无 | observation 绝对目录 | 只写该目录 |
| TF JSON | `--tf-json` | 写最新文件 | 绝对路径 | 验证并读取当前文件 |
| base | `--base-frame=base_link` | base frame | `base_frame` | health 比对；MoveIt base-link |
| camera | `--camera-frame=camera_color_optical_frame` | camera frame | `camera_frame` | health/source frame 比对 |
| tool | `--tool-frame=tool0` | `--require-tool` | 不用于点坐标 | MoveIt end-effector |
| TF mode | `--tf-point-mode=direct` | 无 | `tf_point_mode` | health 比对 |
| detector | weight/imgsz/iou/device | 无 | 固定配置不动态改 | health 严格比对 |
| threshold | `--score-thresh=0.5` | 无 | 每请求动态值 | snapshot 使用 |

实际成功 health 显示：三个相机 topic ready，camera frame 为
`camera_color_optical_frame`，base 为 `base_link`，TF mode 为 `direct`，weight 为绝对
`models/yolo/weights/best.pt` 路径，imgsz 960、iou 0.45、device `cuda:0`，并包含 PID 和
config fingerprint。

## 6. `/health` 和端口规则

`/health` 返回 RGB/depth/camera-info readiness、seq、timestamp、source frame、topic、固定 frame/
detector 参数、server PID 和 fingerprint。pipeline 在 snapshot 之前比较固定配置，不一致以 409
`perception_server_config_mismatch` 停止。

- 8765：当前外部维护的日常服务；workflow 不启动、不重启、不停止。
- 8766：需要隔离验证时的专用测试端口；只能停止自己启动并保存 PID 的实例。

本轮复用了现有 8765，没有启动或停止感知、相机、UR 或 MoveIt 驱动进程。

## 7. 共享底层能力

保留 RGB-D、YOLO、内参、实时 TF、tabletop/点云尺寸和 yaw、track 重绑定、MoveIt、UR5、GF225
接口。共享物理层提供：

- revision-scoped object_ref、持久 track_id、歧义重绑定拒绝；
- 邻居、边缘余量、blocked/free side、目标/占用/空闲区域；
- GF225 指尖、上指、掌部、TCP-to-body、下降、抬升、运输、释放、退回和推动扫掠；
- 不可变 edge、最终门、动作后 fresh observation 和多证据验证；
- 方向、距离、推腕角、抓取/放置 pose 足够区分的失败 fingerprint。

配置只来自 `config/stack_demo_planner.json` 与 `config/workspace_bounds.json`。新增 135° 离散推腕
候选仍逐条经过完整几何和 MoveIt 门，不是安全放宽。

## 8. ClutterExtractionPlanner

两个任务共用 `ClutterExtractionPlanner`。任务层先给合法 track 或 `track × role`，共享层扫描
`[0,180)` 抓取 yaw，生成 direct/staging/pick-away/nudge 边。只有至少一条完整首步边的目标才
形成 TargetOption。没有 direct 完整边（包括运输、放置或 MoveIt 失败）时才展开清障，不再只看
“有没有抓取角”。

清障只针对真实占用抓取区间/扫掠的阻挡物。预推从接触侧空列下降；松散邻接触只允许水平推动；
无净空增益、越工作区、进入目标色区或破坏 protected 的候选被拒绝。

## 9. OrganizePlanner

`OrganizeTaskState` 独立保存 expected/visible/missing/completed/unresolved track、颜色区域、安全槽、
占用、当前结果和失败。单帧漏检不减少 expected；missing 时不能完成。只有所有 expected track
经最新观测确认进入正确颜色区、无非法重叠、无未解决对象时才完成。

2026-07-15 的真实 organize live 检查成功通过相机、TF、感知和参数契约，但当前杂乱场景四个
未完成对象没有完整安全边：中心抓取全部 yaw 被邻块挡住，唯一有增益的推动在预推下降列碰撞。
因此正确结束为非完成，未调用 Qwen/MoveIt，未放宽安全门。报告：
`runtime/organize_blocks_live_integration_20260715_resume2/integration_report.json`。

## 10. HousePlanner

六角色固定为：

```text
left_support_lower   right_support_lower
left_support_upper   right_support_upper
roof                 triangle_top
```

左右下支撑顺序不写死，代码先产生全部合法 `track × role`。同侧上支撑依赖稳定下支撑，roof 依赖
两个稳定上支撑，triangle 依赖已验证 roof。已完成结构自动 protected。

同一真实快照下 house 层存在通过几何和真实 MoveIt 的清障边，实际成功生成 32 条最终候选；
Qwen 选择 `track_blue_02__left_support_lower`，再选 40 mm `-x` 清障边。它只完成 plan-only，
没有实际推动。

## 11. Qwen 无状态接口

默认 `qwen3-vl:8b-instruct`，temperature 0、think false、stream false、预算有界。每次请求只有
短 system、新 user JSON、最新 RGB/短 ID overlay、当前 revision/state/candidates 和有限失败，
没有 assistant 历史或旧图片。

目标只返回 `selected_target_option_id + reason_codes`；边只返回
`selected_candidate_id + backup_candidate_ids + reason_codes`。未知/重复/过期/候选外 ID、动作参数、
object ID 或 task_complete 都被拒绝。非法 JSON 只允许一次独立 format repair；再次失败不静默
改选。

真实成功 run 中 `selected_target.json` 为 `qwen_target_selection`，`selected_edge.json` 为
`qwen_edge_selection`，两者均 `silent_code_reselection=false`。

## 12. 斜角抓取与真实 release yaw

抓取默认 5° 扫描，不偏好检测 yaw 或物体边；要求连续安全区间至少 10°并从内部选角。代码检查
GF225 开口、中心偏差、有效双指接触、指尖、掌部、下降和抬升。单点角接触和窄区间拒绝。

edge 保存物体相对夹爪 yaw，并用它推导实际 release gripper yaw；放置下降和 MoveIt 使用 release
yaw，而不是错误复用目标物体 yaw。

## 13. 抓取后重新观测与 track 生命周期

抓取拆成接近、下降、闭合、抬升、fresh observation、验证。目标明显抬升/位移，或原位置空且
其他可靠 track 证明视图连续时才继续运输；夹爪反馈只能辅助，目标单独消失不能证明成功。

抓取成功后 track 标记 `held_by_gripper`，保存抓取 revision/pose/yaw/相对姿态和最后桌面 pose；
放置后直接锚定实际 place pose，不再用旧的错误位移公式。抓取失败跳过运输/放置；staging、
pick-away、最终任务放置和 orientation staging 使用各自结果谓词。

## 14. 屋顶与三角姿态

roof 显式保存 groove face、face-up、长轴、当前抓取、支撑覆盖、中心偏差、余量和稳定性；triangle
保存 apex/base direction、face、target yaw、base contact、质心投影、support margin 和 roof
相对 pose。

代码不把 wrist yaw 当作翻面。错误面先 staging + fresh observation；只有显式
`orientation_transition.geometry_verified`、regrasp pose、staging place pose 和 expected orientation
齐全时才生成 regrasp。直接 roof/triangle 边仍要求完整结构和姿态谓词。

## 15. 保留的实机安全事实

- RTX 3090 自动功耗上限 250 W；旧 300 W 设置曾导致硬重启。
- `base_link` 工作区唯一配置：x 0.235--0.65 m，y -0.10--0.40 m。
- D435i 在末端；场景/机器人/人工位置变化后必须重新 TF + observation。
- detector ID 只属于当前 revision；跨帧只用 track/rebinding。
- MoveIt 成功不代表 GF225 安全，代码侧完整碰撞门不可省略。
- 高位预旋转 0.24，近桌面/物体 0.03；近物体不得使用高位速度。
- 额外 release gap 固定 10 mm。
- 推动下降不得压碰邻块，松散接触只允许水平阶段。
- 推动不得进入颜色区、破坏 protected 或已完成房屋。
- 模型、JSON、连接、超时、预算和无边都不是完成。

## 16. 本轮新增与主要修改

新增：

- `robot_scene_pipeline/perception_contract.py`
- `tools/workflows/stack_demo/live_integration.py`
- `tools/workflows/stack_demo/perception_client.py`
- `tools/workflows/stack_demo/common/track_lifecycle.py`
- `tests/test_perception_server_contract.py`
- `tests/test_retry4_architecture_fixes.py`

主要修改：感知 runtime/server/ROS capture/depth geometry/object tracking/GF225 swept volume；workflow
arguments/commands/app、边生成/安全/TargetOption、Qwen prompt、MoveIt adapter、执行/验证、house
orientation staging、统一 planner 配置、MoveIt preview 清理顺序、README 和本文件。

本轮没有删除生产文件。旧开放式 planner、task contract、grounded plan、silent fallback、旧 CLI
和旧 schema 已在当前架构中不存在；`rg` 只剩文档中的“已删除”说明和必要 TF/校准兼容代码。

## 17. 测试和 live 结果

最终离线验证：

```text
python3 -m compileall robot_scene_pipeline tools/workflows tests
  PASS
python3 -m unittest discover -s tests
  Ran 137 tests in 0.078s; OK
git diff --check
  PASS
```

真实无运动成功报告：

- 路径：`runtime/build_house_live_integration_20260715_resume3/integration_report.json`
- camera topics、TF、perception health、snapshot、parameter contract：全部 true
- `depth_available=true`，6 个有效 `base_link` 对象，tabletop frame `base_link`
- Qwen target 与 edge ID 选择：成功
- physical edges：32 条最终通过边
- selected edge：`edge_r1_track_blue_02_track_blue_02__left_support_lower_nudge_track_red_02_nx_40_0`
- FinalSafetyGate 与真实 MoveIt plan-only：通过
- `robot_motion_executed=false`
- `gripper_command_executed=false`
- 关节最大绝对差：`6.008148193359375e-05 rad`，来自状态噪声；workflow 未发 motion command
- 修复 ROS 2 Humble 退出清理顺序后，使用同一选中 nudge plan 单独重跑真实 MoveIt plan-only；
  pre-push/contact/push-end/retreat 四段均规划成功，输出明确为 `Mode: PLAN ONLY`、
  `Gripper: disabled`，且退出时未再出现 action wait-set 后台异常。

第一次 live 目录 `...resume1` 保留 ROS Humble action graph API 兼容失败；`resume2` 保留 organize
真实无边；`resume3` 成功。失败目录均未覆盖。

## 18. 实际命令

成功 live 命令：

```bash
/usr/bin/python3 tools/workflows/stack_demo_pipeline.py \
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

当前 organize plan-only 命令仍是：

```bash
python3 tools/workflows/stack_demo_pipeline.py \
  --task-type organize_blocks \
  --instruction "按颜色整理积木" \
  --moveit-plan-only \
  --output-dir runtime/organize_blocks_plan_only_new
```

`--live-integration-check` 绝不能与 execute/yes/push execute 一起出现。

## 19. 尚未进行的实机验证与顺序

未执行：真实夹爪闭合、抓取、运输、颜色放置、pick-away、nudge、房屋结构、roof/triangle 放置、
staging/regrasp。plan-only 成功不代表这些动作成功。

建议顺序：

1. 操作者确认 250 W、急停、TF、控制器和空工作区；驱动继续由外部维护。
2. observation-only 核对 revision/track/短 ID/坐标。
3. 孤立方块 plan-only，人工审查斜角 grasp/release pose。
4. 单次抓取至观察高度，验证失败确实阻止运输。
5. 单次颜色放置和 post-place 谓词。
6. 单个 pick-away；再测试单方向最短 nudge。
7. 小规模到完整 organize。
8. 左右支撑、protected 运输。
9. roof 正确面、staging/regrasp；最后 triangle。
10. 完整 build_house 与修复流程。每步使用新输出目录。

## 20. 已知限制

- 腕部 D435i 通常看不到夹爪内部；抓取验证依赖保守多证据，证据不足会失败。
- groove/triangle orientation 字段仍需要感知稳定提供，尚未实机证明。
- 没有通用、已验证的空中 face-flip primitive；未知/错误面可能在 staging 后仍不可完成。
- 密集场景可能真实没有安全首步；当前会安全结束而不是假边或 task complete。
- 为保证 Qwen 前的候选物理可行性，密集 house 场景会对较多候选运行真实 MoveIt plan-only，耗时
  可能明显增加；当前保持安全完整性，尚未引入会改变候选语义的缓存/裁剪。
- ROS 2 Humble 连续 plan-only 子进程曾在退出时打印后台 action wait-set RCLError；当前已改为先
  shutdown 并等待 TF listener 线程，再销毁 node。规划结果未受影响。
