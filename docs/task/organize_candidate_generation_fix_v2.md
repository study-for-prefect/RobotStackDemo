# 整理积木候选生成、接触侧、连锁推动与角度连续化修改任务

当前分支：`llm-decision-explore`

先阅读仓库根目录 `AGENTS.md`，确认当前分支和工作区状态，再检查相关实现与运行记录。不要依据本文伪代码直接重写；先定位现有代码路径、复用现有数据结构和碰撞模型，只修改与本任务直接相关的模块。

重点分析：

```text
runtime/organize_execute_20260716_094729
runtime/organize_execute_20260716_100336
```

本轮优先级：

```text
P0：修复未完成物体存在时，直接抓取和推动候选错误生成为空的问题
P1：推动候选显式建模 push_direction 与 contact_side
P2：允许并验证受控连锁推动，不把普通未完成积木接触一律判为碰撞失败
P3：规范两指夹爪抓取轴角到 [-90°, 90°)，消除无意义的 180° 旋转
P4：保留正常 pick-place 的高位姿态调整，并选择关节变化最小的 180° 等价表示
```

当前阶段不要求最终放置 yaw 精确，不增加 `orientation_pending`、二次抓取或二次姿态修正任务。

---

## 1. 先定位第二次运行候选为空的真实原因

第二次运行存在多个未完成物体、明显可用的直接抓取角度和推动方向，但最终得到：

```json
{
  "physical_action_edges": [],
  "target_options": [],
  "decision_source": "reobserve"
}
```

必须定位具体代码路径，重点检查：

```text
提前 return
错误 continue
错误区域过滤
直接抓取失败后跳过推动
某个物体失败后停止检查其他物体
某个方向失败后停止检查其他方向
候选在列表重建时静默丢失
candidate_id / edge_id / track_id 不一致
TargetOption 构造时错误过滤
MoveIt 预检结果错误覆盖几何候选
异常被吞掉后返回空列表
```

修改前先用两次运行包中的 scene、task state 和中间 JSON 复现候选生成，记录修改前各阶段数量。

---

## 2. 候选生成总体结构

直接抓取和推动必须对每个未完成物体独立生成，不能互相阻断。

正确结构应等价于：

```python
all_edges = []

for obj in unresolved_objects:
    direct_grasp_edges = generate_direct_grasp_edges(obj, scene)
    push_edges = generate_push_edges(obj, scene)

    all_edges.extend(direct_grasp_edges)
    all_edges.extend(push_edges)
```

禁止：

```text
直接抓取失败
→ 不生成推动候选
```

禁止：

```text
某个物体无可行候选
→ 不检查其他物体
```

禁止：

```text
某个推动方向失败
→ 不检查该物体其他推动方向
```

禁止：

```text
某一阶段结果为空
→ 在未完成完整扫描前提前返回空动作图
```

候选必须经过明确阶段：

```text
raw_generated
→ geometry_rejected / geometry_passed
→ moveit_rejected / accepted
→ physical_action_edge
→ target_option
```

任何候选不得无拒绝记录地消失。

---

## 3. 直接抓取候选

### 3.1 规范两指夹爪抓取轴角

所有两指夹爪抓取轴角统一规范到：

```text
[-90°, 90°)
```

统一函数：

```python
def normalize_gripper_yaw_deg(yaw_deg: float) -> float:
    return (yaw_deg + 90.0) % 180.0 - 90.0
```

示例：

```text
145°  → -35°
100°  → -80°
-120° → 60°
90°   → -90°
```

所有 raw grasp yaw 在进入碰撞检查、IK、MoveIt 和动作参数生成前完成规范化。后续 approach、grasp、close、lift 使用同一连续表示，禁止中途恢复为原始的 `145°` 等表示。

### 3.2 每个未完成物体必须完整扫描抓取角

保留当前有效的角度采样分辨率；若当前没有完整扫描，至少覆盖：

```python
candidate_yaws_deg = range(-90, 90, 10)
```

每个角度独立检查：

```text
接近下降通道
GF225 指尖碰撞
GF225 掌部碰撞
闭爪空间
抓取后垂直抬升通道
运输初始通道
工作空间
IK
MoveIt
```

以下信息不能直接取消整个物体的角度扫描：

```text
blocked_sides 非空
存在邻接物体
orientation_confidence = 0
某一个抓取角失败
某一个 IK 分支失败
```

`blocked_sides` 只能用于快速排序或提示，不能替代完整夹爪几何验证。

---

## 4. 推动候选必须显式包含接触侧

每个推动候选必须同时包含：

```text
push_direction
contact_side
```

定义：

```text
push_direction：被推动积木的运动方向
contact_side：GF225 接触目标积木的一侧
```

两者严格相反：

```text
push_direction = +x  → contact_side = -x
push_direction = -x  → contact_side = +x
push_direction = +y  → contact_side = -y
push_direction = -y  → contact_side = +y
```

统一映射：

```python
OPPOSITE_SIDE = {
    "+x": "-x",
    "-x": "+x",
    "+y": "-y",
    "-y": "+y",
}

contact_side = OPPOSITE_SIDE[push_direction]
```

候选至少保存：

```json
{
  "candidate_id": "push:track_xxx:+x:0.050",
  "action_type": "push",
  "target_track_id": "track_xxx",
  "push_direction": "+x",
  "contact_side": "-x",
  "push_distance_m": 0.05
}
```

不得只保存推动方向而不保存接触侧。

---

## 5. 每个未完成物体检查四个推动方向

对每个未完成物体独立检查：

```text
+x
-x
+y
-y
```

每个方向生成多个推动距离。优先复用当前距离生成策略；若当前只生成单一距离，至少覆盖：

```text
0.03 m
0.05 m
0.07 m
0.09 m
```

正确语义：

```text
每个物体必须检查四个方向
≠
每个物体必须无条件接受四个方向
```

是否接受由接触侧空间、夹爪几何、推动扫掠、连锁推动、安全边界和 MoveIt 决定。

候选生成必须按以下组合独立执行：

```text
每个未完成物体
× 每个推动方向
× 每个推动距离
```

一个组合失败不得阻断其他组合。

---

## 6. 接触侧夹爪空间检查

### 6.1 接触侧被挡时，只拒绝该接触方向

在生成正式推动边前，根据 `contact_side` 构造：

```text
pre-push 位姿
接触位姿
GF225 指尖占用体
GF225 掌部占用体
pre-push 垂直下降通道
```

如果接触侧存在周边积木，使 GF225 无法进入，则该方向拒绝：

```text
pre_push_contact_side_blocked
```

示例：目标 A 的右侧存在 B。

```text
push_direction = -x
contact_side = +x
```

夹爪需要从 A 的右侧进入，因此该候选拒绝。

记录：

```json
{
  "target_track_id": "track_A",
  "push_direction": "-x",
  "contact_side": "+x",
  "blocking_track_ids": ["track_B"],
  "rejection_stage": "contact_side_validation",
  "rejection_reason": "pre_push_contact_side_blocked"
}
```

### 6.2 接触侧被挡不代表推动方向相反的候选也失败

同一场景 A 在左、B 在右：

```text
从 A 右侧接触、向左推 A
→ 接触侧被 B 挡，拒绝

从 A 左侧接触、向右推 A
→ 继续验证，可允许 A 顺带推动 B

从 B 右侧接触、向左推 B
→ B 作为独立目标继续验证
```

不能因为 A 的一个接触侧被挡，就把 A、B 或整个局部场景判为无候选。

### 6.3 禁止在实际间隙不足时从两个积木之间插入夹爪

必须使用 GF225 实际分段几何，而不是只用目标中心到邻居中心的距离。

当前关键参数：

```text
GF225 张开外轮廓约 0.112 m
指尖几何
掌部几何
tool0 → TCP = 0.16 m
碰撞安全余量
```

若两积木之间的可用间隙小于接触位姿所需完整夹爪占用范围，则拒绝中间插入候选。

`blocked_sides` 只能作为加速提示，最终判断必须基于真实几何。

---

## 7. 推动过程允许受控连锁接触

实际杂乱场景中经常出现：

```text
A 推动 B
B 接触 C
C 随之发生位移
```

这种普通未完成积木之间的接触不能一律判为碰撞失败。

必须明确区分三类对象：

```text
protected_completed_objects：已完成整理、需要保护的实际物体
movable_unfinished_objects：普通未完成、允许被推动或顺带移动的积木
fixed_obstacles：桌面边界、容器、固定结构、机器人禁区等不可移动障碍
```

处理规则：

```text
推动接触已完成保护物体
→ 拒绝

推动接触固定障碍或越过工作空间边界
→ 拒绝

推动接触普通未完成积木
→ 进入连锁推动分析，不直接拒绝
```

删除或改写所有等价绝对规则：

```text
被推动物体扫掠体碰到任意普通积木
→ 一律拒绝
```

不得再把普通未完成积木之间的可控接触统一记录为：

```text
ordinary_object_collision
pushed_object_swept_volume_collision
```

应记录为：

```text
controlled_secondary_contact
push_chain_generated
```

只有碰撞已完成物体、固定障碍或违反安全边界时才拒绝。

---

## 8. 构建并验证连锁推动集合

### 8.1 连锁传播

对主推动目标 A，沿 `push_direction` 和 `push_distance_m` 计算可能被接触的未完成物体：

```text
A → B → C → ...
```

建立：

```python
chain_track_ids = ["track_A", "track_B", "track_C"]
```

判断必须基于：

```text
物体实际 footprint
推动方向
推动距离
接触间隙
安全膨胀尺寸
```

不能只根据中心点距离判断。

可以采用等价于以下逻辑的实现，但必须结合现有几何模块：

```python
def build_push_contact_chain(
    primary_object,
    push_direction,
    push_distance_m,
    movable_objects,
):
    chain = [primary_object]
    seen = {primary_object.track_id}
    frontier = [primary_object]

    while frontier:
        current = frontier.pop()
        swept = swept_object_geometry(
            current,
            push_direction,
            push_distance_m,
        )

        for other in movable_objects:
            if other.track_id in seen:
                continue
            if swept.intersects(other.geometry):
                seen.add(other.track_id)
                chain.append(other)
                frontier.append(other)

    return chain
```

避免重复加入和循环。

### 8.2 不设置“连锁数量达到 N 就必然拒绝”的硬规则

不得仅因连锁包含 2、3 或更多未完成积木而拒绝候选。

连锁数量可以增加风险代价和不确定性代价，但不是独立的硬失败条件。

仅在以下实际安全条件失败时拒绝：

```text
联合扫掠体碰撞已完成保护物体
联合扫掠体碰撞固定障碍
超出工作空间或相机有效操作区
产生明确楔紧、夹死或不可释放风险
GF225 推动扫掠路径不可行
IK 或 MoveIt 不可行
```

### 8.3 联合扫掠体

对主目标和预计被顺带移动的物体建立联合扫掠体：

```python
chain_swept_volume = union(
    swept_object_geometry(
        obj,
        push_direction,
        estimated_displacement(obj),
    )
    for obj in chain_objects
)
```

若当前模型不能可靠估计每个次级物体的精确位移，可使用保守位移上界或方向一致的安全包络；不要伪造高精度预测。

联合扫掠体只对以下对象执行硬拒绝：

```text
protected_completed_objects
fixed_obstacles
workspace_boundary
robot_exclusion_geometry
```

尚未加入 chain 的普通未完成物体若被联合扫掠体接触，应继续扩展 chain，而不是立即拒绝。

### 8.4 连锁推动评分

连锁推动允许生成，但通过评分降低高风险候选优先级：

```python
score = (
    primary_clearance_gain
    + secondary_clearance_gain
    + organization_gain
    - chain_size_penalty
    - secondary_displacement_penalty
    - uncertainty_penalty
)
```

要求：

```text
连锁移动能同时清开多个物体
→ 可增加收益

连锁对象越多、次级位移越大、结果越不确定
→ 降低优先级

不确定性较高
→ 降权，不因普通连锁接触本身直接拒绝
```

### 8.5 每次连锁推动执行后强制重新观测

每次只执行一个推动动作：

```text
执行单次推动
→ 停止
→ 重新观测
→ 重新绑定 track
→ 更新所有物体位置
→ 重新生成全部候选
```

不得一次规划多个依赖连锁结果精确位置的后续动作。

---

## 9. 推动候选完整检查顺序

每个推动候选按以下顺序验证：

```text
1. push_direction 与 contact_side 映射合法
2. contact_side 是否有足够的 GF225 进入空间
3. pre-push 垂直下降通道是否无碰撞
4. 指尖能否到达接触面
5. 掌部接触位姿是否碰撞周边物体
6. 主目标推动扫掠体接触对象分类
7. 普通未完成物体接触时构建完整连锁集合
8. 连锁联合扫掠体是否碰撞已完成保护物体
9. 连锁联合扫掠体是否碰撞固定障碍或越界
10. GF225 从 contact 到 push_end 的水平扫掠体是否安全
11. 工作空间和机器人禁区
12. IK
13. MoveIt
```

等价伪代码：

```python
for obj in unresolved_objects:
    for push_direction in ("+x", "-x", "+y", "-y"):
        contact_side = opposite_side(push_direction)

        for push_distance_m in push_distances:
            candidate = build_push_candidate(
                obj=obj,
                push_direction=push_direction,
                contact_side=contact_side,
                push_distance_m=push_distance_m,
            )

            if not contact_side_is_clear(candidate, scene, gripper_model):
                reject(candidate, "pre_push_contact_side_blocked")
                continue

            if not pre_push_descent_is_clear(candidate, scene):
                reject(candidate, "pre_push_descent_collision")
                continue

            if gripper_contact_pose_collides(candidate, scene):
                reject(candidate, "gripper_contact_pose_collision")
                continue

            contacts = classify_push_contacts(candidate, scene)

            if contacts.protected_completed_objects:
                reject(candidate, "protected_completed_object_collision")
                continue

            if contacts.fixed_obstacles:
                reject(candidate, "fixed_obstacle_collision")
                continue

            chain = build_push_contact_chain(
                candidate,
                contacts.movable_unfinished_objects,
            )

            if not chain_sweep_is_safe(candidate, chain, scene):
                reject(candidate, chain_failure_reason)
                continue

            if gripper_push_sweep_collides_with_protected_or_fixed(
                candidate,
                scene,
            ):
                reject(candidate, "gripper_swept_volume_collision")
                continue

            accepted_edges.append(candidate.with_chain(chain))
```

伪代码不要求照抄，必须与当前现有架构一致。

---

## 10. 颜色目标区域与实际物体保护

保留并确认此前要求：

```text
每个颜色区域 Y 方向宽度 = 0.035 m
颜色区域之间保留明确清障通道
颜色区域不覆盖默认初始杂乱区
```

删除所有等价逻辑：

```python
if push_end_inside_any_target_region:
    reject(...)
```

颜色区域只用于：

```text
任务语义
放置目标生成
完成状态判断
整理收益计算
```

空颜色区域不是推动禁区。

只保护实际已完成物体：

```python
protected_completed_objects = [
    obj
    for obj in scene.objects
    if obj.is_completed
]
```

处理规则：

```text
推动进入自身颜色区域
→ 允许，并可计为直接整理收益

推动进入其他颜色空区域
→ 允许，可降低优先级

推动或连锁扫掠碰撞其他颜色区域内的实际已完成物体
→ 由真实几何碰撞拒绝
```

不得用颜色矩形代替物理碰撞体。

---

## 11. 正常 pick-place 保留高位姿态调整

保留正常动作流程：

```text
按可行角抓取
→ 保持抓取姿态垂直抬升
→ 到安全高度后旋转到首选释放姿态
→ 运输
→ 下降
→ 放置
```

不得简单改成运输和放置始终保持抓取角。

当前颜色整理可以保留首选最终物体角，例如：

```text
preferred_object_yaw_deg = 0°
```

但当前完成判定仍只要求：

```text
颜色正确
完整 footprint 位于对应颜色区域
```

最终实际 yaw 暂时不参与 `completed_tracks`，不生成二次姿态修正。

---

## 12. 从 180° 等价表示中选择关节变化最小的解

两指夹爪轴向具有 180° 等价性。目标释放角生成等价集合：

```python
base_yaw_deg = normalize_gripper_yaw_deg(raw_target_yaw_deg)

equivalent_yaws_deg = [
    base_yaw_deg - 180.0,
    base_yaw_deg,
    base_yaw_deg + 180.0,
]
```

不能只按 yaw 数值差选择。对等价目标分别求 IK 或规划，并根据当前关节状态选择代价最小的可行解。

建议代价：

```python
cost = (
    total_weighted_joint_delta
    + wrist_3_weight * abs(wrist_3_delta)
    + ik_branch_switch_penalty
)
```

优先级：

```text
可规划
无碰撞
总关节变化最小
wrist_3 变化最小
避免异常 IK 分支切换
```

必须避免第一次运行中的：

```text
grasp_yaw = 145°
release_yaw = -35°
→ 对两指夹爪几何等价，却执行无意义 180° 旋转
```

正确处理：

```text
raw grasp yaw 145°
→ 先规范为 -35°
→ 对释放目标的 180° 等价表示分别规划
→ 选择相对当前关节状态变化最小的解
```

---

## 13. wrist_3 变化约束

保留：

```text
max_wrist_3_start_goal_delta_rad = 1.75
```

原因：

```text
1.2 rad ≈ 68.8°
无法覆盖规范夹爪方向范围内接近 90° 的必要变化
```

`1.75 rad` 只用于在完成 180° 等价最小化之后阻止异常关节跳变，不能替代角度规范化和 IK 选解。

所有关键运动段统一检查：

```text
current → approach
approach → grasp
grasp → lift
lift → high-level orientation adjustment
orientation adjustment → transport
transport → place
place → retreat
```

不要把 `wrist_3_joint` 的绝对关节值强制限制在 `[-90°, 90°]`。`[-90°, 90°)` 仅用于夹爪轴向 yaw 的规范表示。

---

## 14. tool0 → TCP 补偿

保留并统一：

```text
tool0 → TCP = [0.0, 0.0, 0.16] m
```

本轮不修改其他抓取高度算法、物体高度、hover、approach、lift、place 或 table clearance。

确认系统中只有一个明确的 TCP 补偿来源，不能重复叠加 `0.16 m`。

运行时输出：

```text
tool_frame=tool0
tcp_offset_tool=[0.0, 0.0, 0.16]
```

---

## 15. reobserve 不得作为第一次空候选后的立即退出

当前错误行为：

```text
无候选
→ reobserve
→ 当前程序直接结束
```

应改为同一次任务内部重试：

```text
保存全部候选统计和拒绝原因
→ 重新观测
→ 重建 scene state
→ 重新绑定 track
→ 重算完成状态
→ 重新执行直接抓取角扫描
→ 重新执行四方向、各距离、各接触侧推动扫描
→ 重新生成 physical_action_edges 和 target_options
```

设置集中配置，例如：

```python
max_consecutive_reobserve = 3
```

只有连续多次完整扫描后仍无可行边，且新旧观测没有有效变化，才允许终止。

如果出现：

```text
unresolved_object_count > 0
raw_direct_grasp_candidates_generated == 0
raw_push_candidates_generated == 0
```

应报：

```text
candidate_generation_internal_error
```

不能伪装成正常场景不可行。

重新观测后不得复用上一帧空候选结果。

---

## 16. 候选诊断文件

每个 cycle 必须保存：

```text
candidate_generation_summary.json
candidate_rejections.json
```

### 16.1 candidate_generation_summary.json

至少包含：

```json
{
  "unresolved_object_count": 0,
  "direct_grasp_objects_scanned": 0,
  "raw_direct_grasp_candidates_generated": 0,
  "direct_grasp_candidates_geometry_passed": 0,
  "direct_grasp_candidates_accepted": 0,
  "push_objects_scanned": 0,
  "push_directions_scanned": 0,
  "raw_push_candidates_generated": 0,
  "push_candidates_geometry_passed": 0,
  "push_candidates_accepted": 0,
  "contact_side_blocked_count": 0,
  "push_chain_generated_count": 0,
  "controlled_secondary_contact_count": 0,
  "physical_action_edge_count": 0,
  "target_option_count": 0,
  "rejections_by_reason": {}
}
```

### 16.2 candidate_rejections.json

每条至少包含：

```json
{
  "candidate_id": "",
  "action_type": "direct_grasp_or_push",
  "target_track_id": "",
  "grasp_yaw_deg": null,
  "push_direction": null,
  "contact_side": null,
  "push_distance_m": null,
  "chain_track_ids": [],
  "rejection_stage": "",
  "rejection_reason": "",
  "blocking_track_ids": [],
  "wrist_3_start_goal_delta_rad": null,
  "wrist_3_limit_rad": 1.75,
  "moveit_error": null
}
```

推动方向统计应能看到每个物体四个方向及对应接触侧：

```json
{
  "track_id": "track_A",
  "push_directions": {
    "+x": {
      "contact_side": "-x",
      "generated_count": 4,
      "accepted_count": 2
    },
    "-x": {
      "contact_side": "+x",
      "generated_count": 4,
      "accepted_count": 0,
      "rejection_reason": "pre_push_contact_side_blocked",
      "blocking_track_ids": ["track_B"]
    },
    "+y": {
      "contact_side": "-y",
      "generated_count": 4,
      "accepted_count": 1
    },
    "-y": {
      "contact_side": "+y",
      "generated_count": 4,
      "accepted_count": 0
    }
  }
}
```

### 16.3 拒绝和状态原因

至少区分：

```text
pre_push_contact_side_blocked
pre_push_descent_collision
gripper_contact_pose_collision
finger_collision
palm_collision
gripper_swept_volume_collision
protected_completed_object_collision
fixed_obstacle_collision
push_chain_workspace_violation
push_chain_wedge_risk
workspace_violation
ik_failed
moveit_plan_failed
joint_delta_exceeded
invalid_track_reference
candidate_generation_internal_error
```

允许状态，不应作为普通碰撞拒绝：

```text
controlled_secondary_contact
push_chain_generated
```

---

## 17. 数据结构要求

推动候选增加并贯穿后续阶段：

```json
{
  "target_track_id": "track_A",
  "push_direction": "+x",
  "contact_side": "-x",
  "push_distance_m": 0.05,
  "chain_track_ids": [
    "track_A",
    "track_B",
    "track_C"
  ],
  "chain_object_count": 3,
  "secondary_contact_expected": true,
  "estimated_displacements_m": {
    "track_A": 0.05,
    "track_B": 0.03,
    "track_C": 0.01
  }
}
```

若无法可靠估计精确位移，允许将 `estimated_displacements_m` 标记为保守上界或 `null`，同时记录估计方法和不确定性；不能伪造精确值。

确保以下标识在各阶段一致：

```text
candidate_id
edge_id
target_track_id
chain_track_ids
feasible_first_step_edge_ids
selected_edge_ids
```

---

## 18. 测试要求

至少增加或更新以下测试。

### 18.1 候选生成

```text
1. 每个未完成物体都会进入直接抓取角扫描。
2. 每个未完成物体都会扫描四个推动方向。
3. 每个推动方向会生成配置的多个推动距离。
4. 直接抓取全部失败时，推动候选仍然生成。
5. 某个物体无候选时，不影响其他物体生成候选。
6. 某个推动方向失败时，不影响同一物体其他方向。
7. unresolved_object_count > 0 且 raw 候选总数为零时，产生 candidate_generation_internal_error。
```

### 18.2 接触侧

```text
8. push_direction=+x 时 contact_side=-x。
9. push_direction=-x 时 contact_side=+x。
10. push_direction=+y 时 contact_side=-y。
11. push_direction=-y 时 contact_side=+y。
12. 目标右侧被挡时，从右侧接触的候选被拒绝。
13. 目标右侧被挡时，从左侧接触、向右推动的候选仍继续验证。
14. 邻接物体本身仍生成独立推动候选。
15. 两物体间隙不足时，不生成从中间插入 GF225 的候选。
```

### 18.3 连锁推动

```text
16. A 从左侧接触向右推，顺带接触 B 时，不因 B 是普通未完成积木而直接拒绝。
17. A 推 B、B 接触 C 时，能够生成包含 A/B/C 的连锁候选。
18. 连锁对象数量本身不作为硬拒绝条件。
19. 连锁联合扫掠体碰撞已完成物体时拒绝。
20. 连锁联合扫掠体碰撞固定障碍时拒绝。
21. 连锁推动越过工作空间时拒绝。
22. 普通次级接触记录为 controlled_secondary_contact 或 push_chain_generated。
23. 连锁推动执行后强制重新观测并重建候选。
```

### 18.4 颜色区域与保护

```text
24. 每个颜色区域 Y 宽度为 0.035 m。
25. 颜色区域之间存在清障通道。
26. 空颜色区域不会拒绝推动。
27. 推动进入自身颜色区域允许并产生整理收益。
28. 只在实际扫掠碰撞已完成物体时触发保护拒绝。
29. 不存在颜色矩形级推动禁区。
```

### 18.5 角度与运动

```text
30. 145° 抓取角规范为 -35°。
31. 所有直接抓取候选角位于 [-90°, 90°)。
32. 145° 与 -35° 不产生无意义 180° 高位旋转。
33. 正常 pick-place 仍可在安全高度旋转到首选释放姿态。
34. 多个 180° 等价 IK/规划结果中选择关节变化最小的可行解。
35. wrist_3 变化接近 90° 时不会被旧 1.2 rad 阈值错误拒绝。
36. 最终放置 yaw 暂时不参与整理完成判定。
```

### 18.6 TCP 与重观察

```text
37. CLI、配置、workflow 和 MoveIt 使用同一个 tool0 → TCP = 0.16 m。
38. 不重复应用 TCP 补偿。
39. 无候选时 reobserve 在同一次任务内重试。
40. reobserve 后重新生成候选，不复用上一帧空结果。
```

---

## 19. 执行验证

先运行静态检查和测试：

```bash
python3 -m compileall tools/workflows/stack_demo tools/robot
pytest -q
```

再运行不驱动机械臂的 plan-only：

```bash
python3 tools/workflows/stack_demo_pipeline.py \
  --task-type organize_blocks \
  --instruction "按颜色整理积木" \
  --moveit-plan-only \
  --tcp-offset-tool 0 0 0.16 \
  --output-dir runtime/organize_candidate_contact_chain_fix
```

若当前入口参数与上述命令不同，以实际代码为准，但不得切换回旧流程。

---

## 20. 最终汇报

最终必须报告：

```text
1. 第二次运行候选为空的真实原因和完整代码路径。
2. 修改文件列表。
3. 是否存在提前 return、错误 continue、错误过滤或候选 ID 丢失。
4. 每个未完成物体生成的 raw 直接抓取候选数量和 accepted 数量。
5. 每个未完成物体四个推动方向、对应 contact_side、各距离候选数量。
6. 因 contact_side 被挡拒绝的候选数量及 blocking_track_ids。
7. 进入连锁推动分析的候选数量。
8. 连锁推动被接受和被拒绝的具体原因。
9. 修改前后的 physical_action_edges 和 target_options 数量。
10. 是否已删除颜色区域级推动禁区。
11. 145° → -35° 的规范化代码位置。
12. 180° 等价 IK/规划结果的最小关节变化选择方式。
13. 正常高位姿态调整仍保留的代码路径。
14. reobserve 内部重试流程。
15. tool0 → TCP 0.16 m 的唯一应用位置。
16. 测试结果和 plan-only 输出目录。
17. 尚需实机验证、无法由 plan-only 证明的内容。
```

---

## 最终不可违反的原则

```text
夹爪接触侧被挡
→ 只拒绝该接触方向

目标推动后接触普通未完成积木
→ 构建并验证连锁推动，不直接拒绝

推动或连锁扫掠碰撞实际已完成物体、固定障碍或越界
→ 拒绝

直接抓取失败
→ 仍必须生成推动候选

单个物体或单个方向失败
→ 不影响其他物体和方向

空颜色区域
→ 不是推动禁区

两指夹爪角
→ 先规范到 [-90°, 90°)，再从 180° 等价规划结果中选关节变化最小者

最终放置 yaw
→ 当前阶段不作为完成条件
```
