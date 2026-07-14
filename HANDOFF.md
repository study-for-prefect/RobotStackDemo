# RobotStackDemo 新会话交接文档

更新时间：2026-07-14（Asia/Shanghai）

这是一份给“完全没有此前对话上下文”的新会话使用的权威交接。旧交接中“只做过
plan-only、尚未实机”的结论已经过时；以本文为准。

## 1. 开始前必须知道

- 仓库：`/home/wxm/code/RobotStackDemo`
- 分支：`llm-decision-explore`
- 主入口：`tools/workflows/stack_demo_pipeline.py`
- 机器人：UR5
- 夹爪：GF225 / DH gripper
- 相机：RealSense D435i
- ROS：Ubuntu 22.04 + ROS 2 Humble
- GPU：RTX 3090，调试期间保持 250 W 功耗上限
- 默认 VLM：`qwen3-vl:8b-instruct`
- 30B 对照模型：`qwen3-vl:30b-a3b-instruct`
- 机械臂、MoveIt、相机和感知服务由用户通过桌面
  `Start_Robot_Stack.desktop` 启动。

先完整阅读：

```text
AGENTS.md
README.md
tools/workflows/stack_demo/README.md
本 HANDOFF.md
```

`AGENTS.md` 的核心约束：真实运动必须显式授权并支持 dry-run；不要混合感知、推理、
几何和执行层；行为或命令变化必须同步 README；补丁要小且可审查。当前
`tools/workflows/stack_demo/task_workflow.py` 已经明显过大，是技术债，但当前实机阶段不要
为了重构而同时改变行为。先完成整理，再处理房子，之后再按职责拆文件。

## 2. 我们到底在做什么

最终目标是从一堆任意散落、互相遮挡或挤在一起的积木开始，闭环完成：

1. **按颜色整理积木**：每种颜色一个目标区域，同色积木放在一起并排成一行；
2. **搭房子**：从散落积木中取材，搭成四个方块支撑、一个屋顶、一个三角顶的六角色房子。

中途遇到障碍不能因为一次抓取不可行、碰撞预检失败、MoveIt 失败或 VLM 重复动作就退出。
统一恢复逻辑是：

```text
当前存在能安全抓取、且能推进任务的对象
    -> 抓取
    -> organize：直接放到该对象自己的同色行
    -> house：用于房屋角色，或把障碍抓到安全临时区

当前全部对象没有可靠抓取角度
    -> 清障
    -> 优先抓走可抓障碍
    -> 仍不能抓才从障碍物侧面空处下降，水平推 3–5 cm
    -> 重新观察
    -> 再次执行“能抓先抓，否则清障”
```

水平推动时碰到其他未保护散乱积木可以作为可恢复接触；下降时不能压到任何积木。桌面、
支撑物、已放好的任务结构、protected 对象和最终放置区域不能被随意撞击或破坏。

用户明确要求：当前实机阶段只处理会阻止任务完成的问题：

- 感知错误；
- 目标位姿错误；
- 抓取失败；
- 碰撞预检错误；
- MoveIt 失败；
- 执行后状态未更新；
- VLM 重复无效动作。

不要继续扩展通用任务 Schema。不要同时调整理和房子。严格先完成整理，再开始房子。

## 3. 用户已经确认的实机事实

- 工作区为 `base_link` 米制坐标：
  - `x=0.235..0.65 m`
  - `y=-0.10..0.40 m`
- 每次观测前机械臂回同一标准关节位：`config/rectangle_ready_pose.json`。
- 驱动已经稳定，不要修改机械臂驱动、相机驱动或桌面启动文件，也不要重复启动它们。
- 清障必须开启。
- 整理不应死板地先抓某一种颜色；每一步选择当前最好抓、能推进任务的对象。
- 每种颜色一个区域，同色积木放在同色旁边，一种颜色一排。
- 放置角度不应靠固定“转 90°”规则；如果完整碰撞检测能找到安全角度/落点，就由几何检查决定。
- 放置高度不能过低；当前整理释放默认在名义中心上方增加 10 mm。
- 推障必须从物体旁边空处下降，不能先到物体正上方再水平推。
- 松散积木之间发生接触不等于任务失败；只有保护对象、桌面、支撑结构等必须硬拒绝。
- 当前 250 W GPU 上限必须保留。此前改到 300 W 仍发生过硬重启。

## 4. 当前代码已经完成了什么

### 4.1 模型与调用速度

- 默认使用 `qwen3-vl:8b-instruct`，真实结构化动作推理通常为数秒，不再需要三四分钟。
- 不要用 `qwen3-vl:8b` 或 `qwen3-vl:30b` thinking 标签替代 instruct；它们可能把预算全耗在
  thinking 而没有 JSON content。
- `qwen3-vl:30b-a3b-instruct` 只用于后续对照，不是当前整理阻塞项。
- Ollama thinking/content、finalizer、budget/backend retry 已统一处理。
- 动作输入已压缩物理阻挡信息，避免重复失败后触发
  `INPUT_CONTEXT_BUDGET_EXCEEDED`。

### 4.2 感知和类别恢复

- YOLO 负责检测框与 RGB-D 三维几何。
- 低置信检测、预期对象漏检、类别不可靠时，VLM 可复核类别；VLM 不生成几何坐标。
- 视觉颜色优先于容易误标的 detector label，仍保留 label 兼容。
- 近重复 3D 检测会合并；跨帧使用 `track_id`，当前帧使用
  `object_ref=scene_<revision>:obj_<id>`。
- detector `object_id` 不是永久身份，绝不能跨帧直接复用。

### 4.3 TF 与工作区

- `config/workspace_bounds.json` 已写入用户确认的 x/y 边界和独立
  `organize_layout_bounds`。
- 每次实际快照前使用现场 TF：
  - `base_link <- camera_color_optical_frame`
  - `base_link <- tool0`
- 不要因为机械臂初始位置变化而手工加坐标偏移；先保证标准观测关节位和 fresh TF。
- 缺少工作区或 TF 是明确系统错误，不让 VLM 猜。

### 4.4 整理任务的抓取优先决策

核心文件：

- `tools/workflows/stack_demo/task_workflow.py`
- `robot_scene_pipeline/vlm_task_policy.py`
- `robot_scene_pipeline/grasp_yaw_search.py`

当前每轮会对所有目标行外对象运行物理抓取角扫描，并把精简结果放入动作决策输入：

- 有可靠可抓对象：必须 `pick_place`，不允许 `nudge`、`pick_away`、`reobserve` 或 `stop`；
- VLM 选到不可抓对象或直接放弃：代码改选可靠可抓候选；
- 按连续可行 yaw 区间大小排序，不按颜色固定顺序；
- 同一个失败物理动作会被指纹黑名单拒绝；反方向推动是不同物理策略，不会被误杀；
- organize 未完成时，`stop` 不会被接受为完成。

最新实机又增加了一条关键门：真实抓取必须存在至少 **10° 连续安全 yaw 区间**，并从区间
内部选择角度。最近碰撞的蓝块只有 `23°–27°` 共 4° 狭窄区，旧代码仍选择边界
`22.949°`；现在该抓取会判为不可抓，必须换目标或清障。

### 4.5 整理目标区域与落点

- 每个观察到的颜色只生成一个非重叠行区域。
- 同一任务内首次确定的目标行保持不变，不再每次观测重新漂移几毫米。
- 目标颜色/group 由当前对象真实视觉颜色绑定，模型不能把红块放到蓝区。
- VLM 把目标位置复制成源位置时，代码会在目标颜色区域内生成并完整校验空槽。
- 小于 15 mm 的搬动视为无效，不允许以几毫米“挪一下”冒充整理进展。
- 同色已有成员时，下一块优先放在同色成员旁边，并检查最小间距和行对齐。
- 动作后进度判断允许最多 3 mm 的感知足迹误差，避免方块 yaw 抖动导致刚放好的块被再次抓走。
- 目标区域是在散乱区之外选择的任务区域；推障终点尽量避开所有最终目标区域。

### 4.6 放置夹爪避障（最新修复，尚未实机复验）

核心文件：`robot_scene_pipeline/task_action_validation.py`。

此前整理只检查“被放积木的本体足迹”是否与其他积木重叠，没有检查张开的 GF225 手指。
因此蓝块目标中心与红块中心相距约 33 mm，积木本体不重叠，但夹爪在下降/释放时撞红块。

现在：

- 按目标 yaw 构造张开夹爪的两根实体手指；中间 49 mm 开口不是实体；
- 放置目标必须清除所有当前可见对象，包括前几步已经放好的颜色行；
- 被阻挡时返回 `target_pose_gripper_clearance_blocked`；
- 在同一颜色区域内搜索另一个满足对象足迹、行对齐、间距、工作区和夹爪清障的位置；
- 用现场第二步数据重放：旧蓝块落点 `[0.335, 0.2524]` 被红块拒绝，自动改为
  `[0.29105, 0.2524]` 后通过全部语义与夹爪几何检查。

这项修复已经单元测试和现场数据离线重放，但用户要求先停止测试，因此**还没有进行修复后的
下一次真实运动验证**。

### 4.7 清障

核心文件：

- `tools/workflows/stack_demo/clearance_execution.py`
- `robot_scene_pipeline/tool_swept_volume.py`
- `tools/robot/moveit_preview/push.py`

已实现：

- 清障始终需要 `--execute-push-clearing` 明确开启；
- 当前全部抓取均被物理扫描判为不可行时，无需先浪费一轮失败 pick，允许立即清障；
- VLM nudge 不可执行或重复时，代码在当前散乱对象、四个基坐标方向、0°/90° 腕角中做有界搜索；
- 默认推 4 cm，符合用户要求的 3–5 cm；
- 每个候选依次通过语义、工作区、分段 GF225 扫掠体和 MoveIt plan-only；
- 推障从接触侧的空位置下降；下降/接触阶段不能用“松散接触允许”绕过顶部碰撞；
- 水平推动阶段可允许受控松散连锁接触，随后必须重新观察；
- 推动尽量增加障碍与被阻塞目标的距离，并避开最终颜色区。

仍未完成的实机事实：到目前为止没有在最新逻辑上完整执行并验证一次自动清障闭环。

### 4.8 运动参数

- 标准位复位：`0.08/0.08`
- 接近/平移：`0.08/0.08`
- 安全高位末端 Z 轴预旋转：默认 `3 ×` 主速度，即 `0.24/0.24`
- 预旋转仍可显式用 `--pre-rotate-velocity`、`--pre-rotate-acceleration` 覆盖
- 放置下降：保留较低的 `0.03/0.03`
- 整理释放间隙：10 mm

0.24 只用于高位原地 wrist/Z 预旋转，不要把近物体下降速度一起提高。

## 5. 今天真实机器人实际跑到了哪里

### 5.1 有效实机进展

运行目录：

```text
runtime/organize_blocks_decision_fix_live_250w_20260714
```

该轮真实完成：

1. 第一块红块：抓取成功、搬到红色行、松开成功、退回高位、重新观测成功；
2. 第二块蓝块：抓取成功、搬到蓝色行、松开成功；
3. 预旋转命令确认使用 `0.24/0.24`，约 90° 规划时长从此前约 7–8 秒降到约 2.6 秒；
4. 红块放置后进度正确更新，第二轮没有再次抓红块，而是选择蓝块。

### 5.2 实机暴露的安全问题

用户在第二步及时人工推开障碍，说明蓝块的局部抓取角判断过于乐观。随后蓝块放置时张开的
夹爪碰到第一步红块。根因不是 MoveIt IK，而是：

- MoveIt 规划场景没有完整加入这些桌面散乱小块，不能独自承担桌面碰撞判断；
- 抓取算法接受了只有 4° 宽、且取在边界上的脆弱 yaw；
- 放置语义只检查积木足迹，没有检查张开夹爪的手指。

这三个事实必须长期保留在设计中，不能因为 MoveIt `Execution result: True` 就认为真实路径没有
碰撞。

### 5.3 停止时的机器人状态

第二步蓝块已经松开，夹爪为空，程序在回标准位前被停止。随后已只读检查控制器：

- `scaled_joint_trajectory_controller` active；
- TF 正常；
- 当时末端在约 `[0.405, 0.282, 0.207]` 高位；
- 已执行标准位复位并成功到达 ready pose。

之后曾启动修复后的新实机命令，但用户在约 9 秒内中断，要求先做文档，不再测试。新会话仍应
先检查是否有残留 pipeline/moveit 进程、当前关节位和夹爪状态，不要盲信本文描述的瞬时状态。

## 6. 当前卡在哪里

当前唯一应聚焦的任务是 `organize_blocks`。卡点不是 VLM 调用速度，也不是驱动，而是最新两项
安全修复尚未通过新一轮实机验证：

1. 10° 连续抓取 yaw 门是否会正确拒绝之前那种狭窄蓝块，并改抓其他块或进入清障；
2. 放置夹爪几何是否会在真实场景中把目标从邻近已放红块的位置改到安全位置；
3. 全部不可抓时，自动侧推是否能真实执行一次、重新观察并创造抓取角；
4. 整理尚未完整跑到 `task_complete=true`；
5. 房子尚未进入最终实机闭环调试，必须等整理完成后再开始。

最近一次聚焦回归（在文档修改前）为：

```text
Ran 125 tests
OK (skipped=1)
```

用户当前明确要求“先不做测试”，因此本次文档更新后没有再运行测试或机械臂。

## 7. 下一步计划（严格按顺序）

### 第一步：安全恢复检查，不改代码

1. 用户用 `Start_Robot_Stack.desktop` 启动驱动；不要自行修改/重启驱动文件。
2. 确认没有遗留 `stack_demo_pipeline.py` 或 `moveit_plan_preview.py` 运动进程。
3. 确认控制器 active、夹爪为空、末端在安全高位或 ready pose。
4. 刷新并验证相机/末端 TF。
5. 确认 `nvidia-smi` 功耗上限仍是 250 W。
6. 当前积木因用户人工推开和上次碰撞已变化，必须 fresh observation，不能复用旧动作坐标。

### 第二步：只验证整理的最新两个安全门

1. 从 fresh scene 做一次 plan-only/预检，查看 `physical_grasp_validation`：
   - `robust_feasible_yaw_intervals_deg` 必须至少 10°；
   - 4° 窄区必须出现在 `rejected_narrow_feasible_yaw_intervals_deg`；
2. 查看放置目标验证：
   - 旧行成员必须出现在 place gripper obstacles 中；
   - 被挡时必须得到 `target_pose_gripper_clearance_blocked` 和同色区安全替代点；
3. 只在这两项日志正确时执行一个低风险实机步骤；
4. 观察抓取、放置、复位、fresh observation 和 track 更新完整结束。

### 第三步：完成整场颜色整理

1. 不按颜色固定顺序，持续“能抓先抓”；
2. 全部不可抓时必须执行一次真实自动清障，不准退出；
3. 清障后重新观察，抓走已创造角度的对象；
4. 同色成员放在同一行，放置夹爪不碰其他颜色行；
5. 直到全部积木进入对应区域、间距/行对齐满足且 `task_complete=true`；
6. 保存完整成功 runtime，作为房子调试前的基线。

### 第四步：单独开始房子

整理完整成功后，才开始 `build_house`。沿用同一原则：

- 能抓角色块先抓；
- 可抓障碍先 `pick_away`；
- 不能抓才侧推；
- 已完成支撑/屋顶立即 protected；
- 每一步 fresh observation、重新绑定角色、重新计算谓词；
- 屋顶和三角形翻面使用现有四元数/SLERP，不要固定写 45° 或只绕 yaw。

## 8. 实机命令

驱动由桌面启动后，新终端：

```bash
cd /home/wxm/code/RobotStackDemo
source /opt/ros/humble/setup.bash
source /home/wxm/ros2_ws/install/setup.bash
```

整理完整实机（清障必须开启）：

```bash
python3 tools/workflows/stack_demo_pipeline.py \
  --instruction "按颜色整理积木" \
  --model qwen3-vl:8b-instruct \
  --execute --yes --execute-push-clearing \
  --output-dir runtime/organize_blocks_<new_unique_name>
```

房子命令只在整理成功后使用：

```bash
python3 tools/workflows/stack_demo_pipeline.py \
  --instruction "搭一个房子" \
  --model qwen3-vl:8b-instruct \
  --execute --yes --execute-push-clearing \
  --output-dir runtime/build_house_<new_unique_name>
```

每次都用新的 output dir。不要覆盖历史 runtime，它们是定位实机问题的证据。

## 9. 重要运行目录

```text
# 最近一次真实红/蓝抓放，并暴露蓝抓取/放置碰撞
runtime/organize_blocks_decision_fix_live_250w_20260714

# 最新完整夹爪清障修复启动后很快被用户中断，可能只有部分启动产物
runtime/organize_blocks_full_gripper_clearance_live_250w_20260714

# 较早：红块成功抓放后，重复把已放红块当成待整理对象
runtime/organize_blocks_no_exit_clearance_live_250w_20260714

# 较早：重复动作/context 问题
runtime/organize_blocks_grasp_first_live_250w_20260714

# 离线抓取优先重放
runtime/organize_blocks_grasp_first_offline_replay_20260714

# 曾成功执行绿色侧推的清障记录
runtime/organize_blocks_empty_side_clearance_250w_execute_20260714
```

调实机问题时先读最新 cycle 的：

```text
task_action_attempt_XX_input.json
task_action_attempt_XX_output.json
task_action_attempt_XX_validation.json
task_action_preflight.json
task_step_XX_grasp_validation.json
task_step_XX_pick_plan.json
task_step_XX_place_plan.json
task_action_execution.json
observation_after_*/private_scene_state.json
failure_ledger.json
replanning_context.json
automatic_clearance_search.json
```

## 10. 绝对不要再踩的坑

### 驱动和现场

1. **不要修改或重复启动机械臂/相机驱动。** 用户用桌面启动器管理它们。
2. **不要复用碰撞前或人工移动前的动作坐标。** 场景变化后必须 fresh observation。
3. **不要在不清楚夹爪是否持物时直接 Ctrl-C。** 终端输出可能有延迟；此前一次以为机械臂仍在
   ready，实际已经抓起红块并在高位预旋转。若持物，先安全完成放置或退回高位。
4. **不要因为 MoveIt plan/execution 成功就认定桌面小块无碰撞。** 当前 MoveIt planning scene
   没有完整表达所有检测积木，必须保留代码侧 GF225/积木几何门。
5. **不要把 0.24 用到近物体下降。** 它只用于安全高位腕部预旋转。

### 抓取和放置

6. **不要接受只有几度宽的抓取角缝隙。** 至少 10° 连续可行且选区间内部。
7. **不要只检查被放积木的本体足迹。** 必须检查张开夹爪下降和释放是否碰当前所有对象。
8. **不要在同色/异色行之间只留“积木不重叠”的距离。** 还要给 GF225 张开手指留空间，必要时
   在同一区域内改变 x。
9. **不要把几毫米位移当任务进展。** 目标点复制源点或小于 15 mm 必须重新选槽。
10. **不要把刚放好的块因亚毫米 yaw/足迹抖动又判为区域外。** 保留 3 mm 观测容差和固定目标行。
11. **不要把放置高度降回贴桌。** 当前 10 mm release gap 是实机碰桌后的修复。

### 清障

12. **不要因“选中的目标不好抓”就退出。** 先尝试其他可靠抓取；全不可抓立即清障。
13. **不要先移动到障碍物正上方再推。** 从接触侧空处下降，再水平推。
14. **不要用 loose-contact 规则允许下降压到邻块。** 受控接触只用于水平推动阶段。
15. **不要把普通未保护红/蓝块永久当作不可碰结构。** 但已经进入最终行、房屋支撑或 protected
    对象必须保护。
16. **不要因为一个推方向失败就禁止反方向。** 相反接触侧是不同物理策略。
17. **不要把清障推向最终颜色区。** 初始规划要让最终区与散乱区保持距离，推障也要避开它。

### 感知、身份和决策

18. **不要把 detector id 当跨帧身份。** 当前帧用 object_ref，跨帧用 track_id。
19. **不要让 VLM 生成/覆盖 TF、工作区或三维坐标。** VLM 辅助类别与任务决策，几何来自代码。
20. **不要严格按颜色顺序抓。** 每轮选择当前最可靠可抓、能推进任务的对象。
21. **不要让 VLM 重复同一无效动作 8 次。** 有可抓对象时代码选可靠抓取；全不可抓时有界清障。
22. **不要把模型/连接/预算失败伪造成 stop。** Backend、Budget、Action 计数保持分离。
23. **不要继续扩大 prompt 和失败历史。** 物理阻挡只传精简 id，避免再次超出上下文。

### 范围和电源

24. **不要同时调整理和房子。** 当前只完成整理。
25. **不要继续扩展通用 Schema。** 只修阻止当前任务完成的错误。
26. **不要把 GPU 功耗恢复到 300/370 W。** 当前保持 250 W；硬重启不是普通软件异常。
27. **不要使用 `qwen3-vl:8b`/`qwen3-vl:30b` thinking 标签做严格动作 JSON。** 使用 instruct。
28. **不要执行 destructive git 操作。** 工作树有大量用户与本轮未提交修改，先审阅再提交。

## 11. 当前工作树和测试

工作树包含大量未提交修改，涉及感知恢复、VLM 决策、追踪、整理几何、清障、运动参数和测试。
不要 `git reset --hard`，不要覆盖用户改动。先运行：

```bash
git status --short
git diff --check
```

本轮聚焦回归在最新代码修复后、文档修改前通过 125 项（1 skip）。用户随后明确说先不测试，
因此新会话不要自动重跑全套；等用户恢复验证工作后，再按风险选择聚焦测试和一次新场景 plan-only。

## 12. 最重要的一句话

**当前不要宣布完成。下一步只验证颜色整理的两个最新实机安全门：拒绝狭窄抓取 yaw，以及放置时
张开夹爪避开已放好的颜色行；随后必须真实完成一次自动清障和整场颜色分类。整理成功后，才开始
房子。任何单次不可抓、碰撞预检或 VLM 坏选择都应触发换抓取/抓走障碍/侧推清障，而不是退出。**
