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

## 0715retry4 感知 500 修复

原始失败证据位于 `runtime/organize_blocks_plan_only_20260715/`；任务中提到的
`runtime/0715retry4` 实际不存在，原目录未被覆盖。该轮已经写出 RGB、YOLO、深度、TF 和桌面
几何，却在 `private_scene_state.json` 生成前返回 HTTP 500。安全复现得到服务端原始异常：

```text
'types.SimpleNamespace' object has no attribute 'instruction'
```

根因是感知层 `build_private_state` 仍读取任务层 `args.instruction`，而持久感知服务器构造的是
与任务无关的 `SimpleNamespace`；同时旧客户端丢弃了 HTTPError 正文，只留下笼统的
`HTTP Error 500: Internal Server Error`。完整证据与排除项记录在
`runtime/0715retry4_analysis.json`。

当前感知状态不再依赖任务 instruction，`scene_id` 与坐标系字段也已分离。客户端会保留 JSON
和非 JSON 错误正文，并把相机未就绪、TF 请求错误、detector 失败和场景处理失败区分为结构化
错误；这些错误都不会被解释为任务完成。

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
三角棱柱抓取还要求两指形成稳定侧面接触：即使长边小于 GF225 开口，也禁止沿长轴闭合去夹三角尖角；抓取角必须沿长轴布置，使夹爪跨短边侧抓。
松散邻接触只允许出现在水平推动阶段。没有预计净空收益的推动不会进入候选。

放置使用代码生成的槽位或房屋角色位姿。`place_pose` 表示预期最终物体位姿；代码根据抓取时
物体相对夹爪 yaw 计算释放夹爪 yaw。`release_pose` 在目标上方 10 mm，MoveIt 预检与开爪执行
使用该夹爪姿态。张开夹爪下降、
掌部、持物运输、释放和退回均在 Qwen 调用前预检，并在选中后由最终安全门复核。

## Qwen3-VL 协议

默认模型来自 `config/stack_demo_planner.json`：`qwen3-vl:8b-instruct`。配置固定
`temperature=0`、`think=false`、`stream=false` 和有界输出预算。所有 target/edge、检测语义
复核、特殊构件语义/朝向复核、姿态分析、JSON finalizer 及格式修复请求都显式发送
`options.num_ctx=32768`；启动日志及每次 Ollama diagnostics 同时记录 requested/actual context。

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

颜色区域统一位于 `base_link` 的 `x=0.250..0.420 m`，避开默认初始杂乱区；Y 坐标为：

| 颜色 | Y 范围（m） |
| --- | --- |
| 红 | `0.230..0.265` |
| 绿 | `0.275..0.310` |
| 蓝 | `0.320..0.355` |
| 黄 | `0.365..0.400` |

每区 Y 宽 `0.035 m`，相邻区域之间保留 `0.010 m` 清障通道。完成要求颜色匹配、中心在对应
区域内，并且考虑物体 yaw 后的二维实际足迹基本位于区域内；中心刚越过边界不算完成。
颜色矩形不是碰撞体，也不会仅因推动终点进入某色区域而拒绝。推动仍检查所有普通物体、
已完成物体的带余量实际保护体、GF225 和被推动物的完整扫掠；推入自身颜色区域计为整理收益。

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
屋顶物体中心固定对准 `house.origin_center_base_m` 的 XY；其接触高度使用独立的
`house.roof_final_place_z_offset_m`，不复用普通方块的下压偏移。

梁柱直接放置会逐个检查两种相差 `90°` 的沿边夹持方向；一个方向被相邻柱体挡住时继续检查
正交方向，任一方向通过几何与 MoveIt 就直接执行，不进入中转。两种方向都失败时才允许中转。
尚未满足屋顶前置条件的屋顶若需要为梁柱清障，会优先放到距房屋中心至少 `0.15 m` 的安全点。

`rectangle`、`concave_rectangle`、`triangle` 都走完整 SE(3) 特殊构件路径。D435i mask/depth
点云通过三维 PCA 和可见面拟合给出对象长轴、面法向和对象四元数；Qwen 只在标注全景图与
逐候选裁剪图上输出受限类别、宽面/凹槽/直角边视觉证据。YOLO/Qwen 冲突、遮挡严重、证据
跨 revision 或无法恢复三维语义轴时，最终放置候选不会生成，只能重新观测或实体中转。
抓后、放后和无动作重观测会先递增 `scene_revision`，再执行该帧语义复核，因此融合证据与
最终场景属于同一 revision。特殊构件响应按对象独立校验；一个对象给出越界置信度、重复或
未知 candidate ID 时只拒绝该对象，并在 `vlm_shape_response_validated.json` 记录原因，不再
丢弃同一批中其他合法对象。

最终目标长轴取左右上层支撑实际中心连线，不使用固定世界 X 轴或固定 roof yaw。正确的宽面、
凹槽方向或指定边会保持当前完整三维面态，不再强制构造 `+45°/-45°`；仅在实测长轴等目标
朝向与支撑轴不一致时计算最小必要四元数差。抓起后先运输到距房屋至少
`0.15 m` 的安全空中点，在 GF225 实际抓取 TCP 世界 XYZ 固定于 1 mm 内时用不大于 5° 的
SLERP waypoint 完成旋转。`tool0` 会因 0.16 m TCP 偏移产生补偿位移。获得最终姿态后才运输
到目标上方，最终段保持四元数并仅垂直下降。扫掠包含持物三维包络、指尖、上指、掌部、
D435i/支架、桌面、其他对象、protected 结构和工作区。

MoveIt 对这条特殊路径按一个连续 TCP 四元数序列进行 plan-only：首段使用实时关节状态，后续
每段使用上一轨迹终点作为起点，并保留逐段碰撞检查与 1.75 rad 的 `wrist_3` 门限；执行阶段
transport 复用相同连续语义，不再把固定 TCP 旋转 waypoint 分别从 ready 状态规划。每段除检查
相邻轨迹点外还检查所有关节的未折叠起点到终点位移，拒绝用数百个小步掩盖接近整圈的远端 IK
分支。到位残差不超过 `2°` 直接接受；特殊构件放置下降不再为亚度级残差启动第二次全姿态 IK。

正确宽面/凹槽/指定边已就位时保留当前面态，只修正真正不一致的目标朝向；表面错误时先做
桌面 staging、重新观测和重新抓取，禁止在空中直接换面。最终姿态确实倾斜时释放额外提高
`4 mm`，释放后保持最终四元数垂直撤离，不在屋顶位置转回 downward。

绿色三角件还使用积木的标定比例作为独立三维证据：斜边为方块边长 `2s`，三角面高和棱柱
厚度均为 `s`；点云量测允许配置误差，但期望比例固定为 `2:1:1`，从而与约 `1:1:1` 的绿色
方块区分。无遮挡三角形若 VLM 对直角方向偶发返回 `unknown`，代码只在高置信宽面证据和完整
绿色三顶点轮廓同时成立时，用轮廓中唯一的 90° 角作确定性方向兜底；图像中质心到直角顶点
的方向必须经实时 camera→base_link 旋转投影到三角面，不能把 `up/right` 合并为同一符号。
裁边、严重遮挡或非三角轮廓仍拒绝生成姿态。放后验收必须由新图像确认直角顶点向上，已
下发的刚性目标四元数只能作为命令审计，不能替代实际姿态证据。
若 YOLO 把同一绿色三角棱柱切成两个方块框，感知层只在两片同色、mask 连通且有斜外边、
深度顶部连续、局部支撑面一致、单片均为近方块而合并点云呈长构件时合并；随后必须用联合
mask 重新构造点云、尺寸和完整三维姿态，再做 `2:1:1` 判定。两个相邻但轮廓独立的绿色方块
不会合并，原始检测框和合并证据分别保留在检测及 tabletop 日志中。
每个新鲜帧都可独立判断三角块是否已直角朝上，不依赖同一进程中的翻转历史：必须同时满足
`2:1:1` 度量比例、姿态置信度、斜边近水平、可见面朝上和物体底部接触局部桌面支撑。D435i
斜边 Z 分量容差为 `0.20`（约 11.5°），覆盖实测 `0.12–0.16` 抖动；全部通过后禁止再生成
45° 调整，任一物理证据缺失则仍走“翻一次、放桌面、重新检测”。

`--continue-from-cycle` 只恢复已验证的 track/角色连续性，下一进程仍必须采集新图像。已开爪
但因单帧点云/PCA 噪声失败的屋顶可在续跑中以新鲜接触与结构关系确认；已释放但顶点方向
错误或未知的三角件只恢复为 `REPAIRABLE`，绝不会因上一条目标姿态被标成完成。确认后六个
角色全部成为 protected，不再生成拆屋顶或重复抓顶件的动作。总高度验收会把两层合并柱
拆成两个真实层，并使用 `roof_nominal_thickness_m`，不会把约 62mm 的桌面到屋顶顶面高度
误当成屋顶厚度。

没有 continuation 的新进程也不会拆掉画面中已经完成的房屋：当前帧长方形屋顶中心必须位于
代码定义的房屋位置容差内，并由顶部深度簇与局部桌面/支撑面独立测得
`top_z - support_z = 60–67 mm`，才恢复并保护屋顶以及被它遮挡的四个支撑角色。桌面上单独的
约 14 mm 厚长方体不满足该实测高度门，不会触发恢复。

若所有合法安全空中翻转点都被当前物体占用，规划器会把具体阻挡 `track_id` 写入失败预检，
只生成移开这些物体的 `pick_away_blocker`/安全清障候选；不会把同一个屋顶件在 staging 点间
来回搬运。一次成功的姿态中转若没有带来新的可执行三维证据，该屋顶件不会再次中转。

普通积木保持 downward、roll/pitch 不变，但抓取 yaw 与释放 yaw 通过物体相对夹爪 yaw
解耦。它会在远离目标/protected 结构且高于最高障碍物的位置先完成 yaw-only 转正，再保持
目标 yaw 运输和垂直下降；房柱对齐实际房屋轴，颜色整理对齐槽位 X/Y 轴。

房屋角色状态为 `UNPLACED / REPAIRABLE / COMPLETED_VISIBLE / COMPLETED_OCCLUDED /
INVALIDATED`。完成或可局部修复的结构均受保护；单帧漏检不撤销完成状态，结构位置重绑定可
接纳新的 track ID。最终完成同时检查六角色、总高度、左右柱顶、屋顶双侧承托、三角顶接触、
支撑图连通和 unresolved missing tracks。这部分新架构尚未经过实机执行验证。

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
- 实测 `tool0 -> GF225 grasp TCP` 平移唯一值为 `[0.0, 0.0, 0.16] m`。planner 配置、CLI 和
  MoveIt 调用必须一致；位姿换算只在 MoveIt 目标转换时应用一次，不改物体或抓取高度。
- GF225、抓取、推动、速度、10 mm 释放间隙和房屋参数：
  `config/stack_demo_planner.json`。
- RTX 3090 自动功耗上限为 250 W；systemd unit 和脚本均不设置 300 W。
- D435i 安装在末端，每次场景获取前刷新实时 TF；动作后必须重新观测。
- 高位预旋转使用 0.24，近物体移动使用 0.03；MoveIt 执行器先抬到安全高度再预旋转。
- MoveIt 成功不是 GF225 安全证明；代码侧指尖、掌部和分阶段扫掠检查不可省略。
- 实机驱动由外部维护；本入口不会启动或重复启动 UR5、RealSense、MoveIt 或 GF225 驱动。

## 感知 client/server 契约

每次快照请求显式发送绝对 `output_dir`、绝对 `tf_json`、`base_frame`、`camera_frame`、
`tf_point_mode`、`score_thresh`、request ID、capture reason、scene revision、客户端超时和 TF
最大年龄。服务端先验证 TF 文件存在、JSON/4x4 矩阵有限、parent/child 匹配且时间新鲜，再使用
当前请求参数处理同一组同步 RGB-D 帧，不再依赖服务启动时残留的 TF 路径。

`GET /health` 返回 RGB/depth/camera-info readiness、序号/时间戳/来源 frame、三个 topic、固定
base/camera/frame mode、detector weight/imgsz/iou/device、低置信度候选阈值与语义复核候选池能力、
server PID 和配置 fingerprint。pipeline
在 `/snapshot` 前逐项比较固定配置；不一致会以
`perception_server_config_mismatch` 拒绝继续。`score_thresh` 和当前 `tf_json` 是每次 snapshot
的动态参数，不是服务端固定配置。服务还在 `/health` 发布感知管线版本；感知行为修改后，
工作流会拒绝仍在内存中运行的旧服务，必须重启服务后才能继续。

搭房子时，初始、抓后、放后及无候选重观测都会对主检测和具备深度几何的低置信度候选执行
VLM 图像语义复核。VLM 可修正已有候选类别并提升真实低分候选，但不能凭空创建检测框或三维
坐标；重叠低分重复框由代码确定性抑制。旧感知服务若不提供候选池能力会被配置契约拒绝，
必须重启加载新版服务。
建房融合会保留检测器实际提供、轮廓未裁边且标定三维尺寸符合 `2s × s × s` 的绿色方块/三角
候选，即使单帧视觉复核错误否决也不会丢失三角件；任一三维尺度小于 `8 mm` 的深度薄片会被
拒绝，避免局部图像边缘成为 MoveIt 碰撞物体。
三角顶放后若俯视投影呈四边形、视觉模型无法指出直角顶点，代码使用同一新鲜 mask/depth
点云沿长轴的竖直截面复核：只有下部宽度显著大于上部，且三角件满足 `2:1:1`、屋顶接触、
中心承托和长边对齐时才确认直角向上；执行目标姿态本身仍不能代替观测证据。
绿色 YOLO 方块不会仅凭大模型一次 `square/high` 进入梁柱候选：轮廓贴图像边界、方块/三角复核
缺失，或桌面轮廓比例明显不符合方块时，代码硬门会拒绝该次候选并等待重新观测。
`annotated_detector_raw.jpg` 保留原始 YOLO 标签；完成两级复核后，默认
`annotated_detector.jpg` 重绘融合标签，因此 2:1:1 绿色三角件显示为 `triangle green`，不会再让
原始 `square green` 文本被误认为最终规划类别。

结构化错误使用 `ok=false`、HTTP status、`error_code`、`error_type`、`error`、`request_id` 和
`effective_config`。请求/TF 错误通常为 400；`camera_not_ready`、RGB-D 不同步为 503；
`detector_failed`、`scene_processing_failed` 和意外内部错误为 500。

snapshot subprocess fallback 默认关闭。只有显式给出
`--allow-snapshot-subprocess-fallback` 才会启用，并保留原服务端错误、记录实际 fallback 命令，
且使用与 server 请求相同的 TF/frame/detector/topic 参数。

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
  --task-type organize_blocks \
  --instruction "按颜色整理积木" \
  --moveit-plan-only \
  --tcp-offset-tool 0 0 0.16 \
  --output-dir runtime/organize_blocks_region_tcp_fix_test
```

使用真实相机、最新 TF、真实 Ollama 和已有 MoveIt 做一次中央强制无运动集成检查：

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

`--live-integration-check` 与 `--execute`、`--yes`、`--execute-push-clearing` 互斥，并强制
`execute=false`、`moveit_plan_only=true`、`max_task_steps=1`。所有子进程还会由中央守卫再次拒绝
执行/夹爪 flag；它不执行 ready pose、轨迹、nudge、pick-away 或动作后模拟。

已有 8765 服务由外部维护，不应为了测试重启。需要隔离验证新 server 时可在 8766 启动专用
进程，并在测试后只停止自己启动的 PID：

```bash
python3 -m robot_scene_pipeline.perception_server \
  --host 127.0.0.1 --port 8766 \
  --camera-source ros-topic \
  --color-topic /camera/camera/color/image_raw \
  --depth-topic /camera/camera/aligned_depth_to_color/image_raw \
  --camera-info-topic /camera/camera/color/camera_info \
  --detector-weight models/yolo/weights/best.pt \
  --detector-imgsz 960 --detector-iou 0.45 --detector-device cuda:0 \
  --use-tf --base-frame base_link \
  --camera-frame camera_color_optical_frame --tf-point-mode direct \
  --estimate-tabletop
```

实机命令保留如下，但本次重构未执行：

```bash
python3 tools/workflows/stack_demo_pipeline.py \
  --task-type organize_blocks \
  --instruction "按颜色整理积木" \
  --tcp-offset-tool 0 0 0.16 \
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

本次通过真实 D435i、实时 TF、`qwen3-vl:8b-instruct` 和已有 MoveIt 完成了一次
`build_house --live-integration-check`。报告位于
`runtime/build_house_live_integration_20260715_resume3/integration_report.json`：相机、TF、感知契约、
Qwen 两级 ID 选择、32 条通过边和最终 MoveIt plan-only 均成功；前后关节最大差
`6.01e-05 rad`，`robot_motion_executed=false`、`gripper_command_executed=false`。

同一场景的 organize 安全检查位于
`runtime/organize_blocks_live_integration_20260715_resume2/integration_report.json`。感知链路全部通过，
但当前四个未完成目标没有完整安全边，流程正确地以“非完成、重新观测”结束，没有调用 Qwen
或 MoveIt，未放宽碰撞门。尚未执行真实抓取、放置、清障或房屋结构搭建；下一步仍应按
“孤立目标抓取门控 -> 单次颜色放置 -> 受控清障 -> 完整整理 -> 支撑 -> 屋顶 -> 三角顶”验证。

更详细的模块和日志说明见
[`tools/workflows/stack_demo/README.md`](tools/workflows/stack_demo/README.md)，感知说明见
[`robot_scene_pipeline/README.md`](robot_scene_pipeline/README.md)，交接事实见 [`HANDOFF.md`](HANDOFF.md)。
