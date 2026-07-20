你当前位于 RobotStackDemo 仓库的 `llm-decision-explore` 分支。

直接修改代码，不要只分析。先阅读根目录 `AGENTS.md`、当前 README、配置文件和相关测试。不要缩小修改范围，不要用临时补丁掩盖架构问题。

# 一、任务目标

修复搭房子、三维姿态调整、视觉语义复核、普通积木放置角度和房屋完成判定中的系统性问题。

必须实现：

1. `rectangle`、`concave_rectangle`、`triangle` 三类特殊构件全部支持完整三维姿态调整，不是只转 yaw。
2. 三类特殊构件放置时，物体长轴必须沿左右两根上层房柱中心的实际连线。
3. 三类构件抓起后，先远离现有房屋结构，到达安全空中姿态调整位。
4. 在安全空中姿态调整位保持 GF225 实际抓取 TCP 的世界坐标 XYZ 不变，完成三维旋转，获得最终放置姿态。
5. 三维姿态调整完成后，保持最终姿态移动到最终目标上方。
6. 进入房屋目标区域后禁止继续执行大角度三维旋转，只允许小范围轨迹连续性修正和垂直下降。
7. 三类构件最终语义姿态不同：

   * `rectangle`：长轴沿房梁轴，展示长×宽的宽面，不能以厚度面或窄侧面作为主要屋顶面。
   * `concave_rectangle`：长轴沿房梁轴，凹槽开口方向朝下。
   * `triangle`：长边沿房梁轴，指定直角边朝上。
8. YOLO 对这三类构件容易误分类。增加 Qwen3-VL 辅助分类和朝向识别，但大模型只能输出视觉语义证据，不能直接生成机械臂轨迹、欧拉角、坐标或动作。
9. 修复抓取候选排序、长方形物理检验、实体中转逻辑和房屋完成状态，防止已完成结构再次被抓起。
10. 普通积木保持末端 downward，禁止 roll/pitch，但抓取角与放置角必须解耦。
11. 普通积木必须在远离目标结构的安全高度完成 yaw-only 转正，然后保持目标 yaw 移动、下降和释放。
12. Ollama/Qwen所有调用的有效上下文统一修改为32K，即 `32768`。
13. 颜色整理、普通方块抓放、推动和清障流程不得因三维姿态重构发生回归。

# 二、已确认的上下文配置问题

当前分支不是32K。

当前代码中至少存在：

```json
"policy": {
  "num_ctx": 12288
}
```

并且：

```python
POLICY_GENERATION_CONFIG = {
    "target_selection": {"num_ctx": 12288, ...},
    "edge_selection": {"num_ctx": 12288, ...},
    "orientation_analysis": {"num_ctx": 24576, ...},
    "semantic_detection_review": {"num_ctx": 12288, ...},
    "final_json_generation": {"num_ctx": 12288, ...},
}
```

`app.py`会在未提供CLI覆盖时执行：

```python
args.vlm_num_ctx = int(policy["num_ctx"])
```

随后 `_generation_budget()` 优先使用 `args.vlm_num_ctx`，所以默认情况下包括姿态分析在内的请求实际都是 `12288`，不是 `24576`，更不是32K。

## 2.1 修改要求

将统一权威默认值改为：

```python
DEFAULT_VLM_NUM_CTX = 32768
```

修改：

```json
"policy": {
  "num_ctx": 32768
}
```

所有以下调用默认有效上下文必须为32768：

```text
target_selection
edge_selection
orientation_analysis
semantic_detection_review
special_shape_semantic_review
final_json_generation
格式修复请求
```

不要继续在多个模块中维护互相冲突的 `num_ctx`。

优先重构为：

```python
def effective_vlm_num_ctx(args, config) -> int:
    value = int(args.vlm_num_ctx or config.policy.num_ctx)
    if value != 32768:
        raise ValueError(
            f"stack_demo requires vlm num_ctx=32768, got {value}"
        )
    return value
```

也可以保留调试覆盖，但正式默认路径必须为32768，且所有调用必须记录实际值。

`POLICY_GENERATION_CONFIG`只维护不同策略的 `num_predict`。若仍保留每策略 `num_ctx` 字段，则全部必须为32768，并增加一致性断言。

Ollama请求继续明确发送：

```python
payload["options"]["num_ctx"] = 32768
```

不能只依赖Ollama服务默认值。

## 2.2 运行时验证

程序启动时输出：

```text
effective_vlm_num_ctx=32768
effective_vlm_num_predict=...
model=qwen3-vl:8b-instruct
```

每次调用的诊断文件必须包含：

```json
{
  "requested_num_ctx": 32768,
  "actual_num_ctx": 32768,
  "prompt_eval_count": 0,
  "eval_count": 0
}
```

保留并检查现有：

```text
ollama_calls/*/ollama_request.json
ollama_calls/*/ollama_diagnostics.json
model_runtime_diagnostics.json
```

在模型加载并完成至少一次请求后执行：

```bash
ollama ps
```

确认对应模型的 `CONTEXT` 为32768。

若不是32768，必须先检查：

```text
请求options.num_ctx是否实际为32768
是否存在旧模型常驻实例
是否存在其他启动脚本或环境变量覆盖
是否请求发送到不同Ollama实例
```

必要时卸载旧常驻模型，再重新发出32K请求。

## 2.3 上下文测试

新增测试验证：

1. planner配置默认值是32768。
2. `app.py`最终得到的 `args.vlm_num_ctx` 是32768。
3. target、edge、semantic review、orientation和finalizer请求都包含：

```json
"options": {
  "num_ctx": 32768
}
```

4. 任何旧的12288或24576默认值都会导致测试失败。
5. 日志中的 `actual_num_ctx` 必须是32768。
6. 语义复核的图像批次不会因输入过大超过：

```text
estimated_input + output_budget + reserve <= 32768
```

# 三、当前错误实现必须删除

重点检查并修改：

* `config/stack_demo_planner.json`
* `tools/workflows/stack_demo/app.py`
* `tools/workflows/stack_demo/arguments.py`
* `tools/workflows/stack_demo/perception_semantic_review.py`
* `tools/workflows/stack_demo/policy/qwen_client.py`
* `robot_scene_pipeline/ollama_policy_client.py`
* `tools/workflows/stack_demo/clutter/edge_generation.py`
* `tools/workflows/stack_demo/clutter/grasp_edges.py`
* `tools/workflows/stack_demo/clutter/path_safety.py`
* `tools/workflows/stack_demo/house/placement.py`
* `tools/workflows/stack_demo/house/state.py`
* `tools/workflows/stack_demo/house/completion.py`
* `tools/workflows/stack_demo/common/result_verification.py`
* `tools/workflows/stack_demo/common/action_edges.py`
* `tools/workflows/stack_demo/common/action_execution.py`
* 实际调用 `tools/robot/moveit_plan_preview.py` 的命令构造和执行代码
* 感知、Qwen调用、场景对象构建和跨帧跟踪代码
* 对应配置文件、README和测试

删除以下错误语义：

```python
FULL_3D_ORIENTATION_SHAPES = {
    "triangle",
    "concave_rectangle",
}
```

改为：

```python
FULL_3D_ORIENTATION_SHAPES = {
    "rectangle",
    "concave_rectangle",
    "triangle",
}
```

但不能只增加字符串，必须完成真实三维轨迹支持。

删除特殊构件最终执行链中的以下伪实现：

```python
orientation_policy = "full_3d_allowed"
release_pose["yaw_deg"] = ...
motion_role = "special_shape_orientation_at_destination"
```

若这些字段背后仍然只有yaw，则属于错误实现。

禁止特殊构件继续使用：

```text
抓取
→ 保持抓取yaw运输
→ 到房屋上方改变release yaw
→ 下降释放
```

禁止使用：

```text
移动到房柱或屋顶附近
→ 原地大角度旋转约45°
→ 下降
```

该路径容易通过持物扫掠、GF225掌部、D435i或UR5腕部破坏已经完成的房屋结构。

# 四、区分空中姿态调整位和实体中转区

必须明确区分两个概念。

## 4.1 空中安全姿态调整位

正常路径使用。

特点：

* 不松开夹爪；
* 不改变抓持关系；
* 位于房屋结构和杂乱积木之外；
* 有足够三维旋转扫掠空间；
* 在此处完成最终三维姿态；
* 姿态完成后再移动到最终放置位置。

动作类型可命名为：

```text
airborne_orientation_adjustment
```

正常路径：

```text
抓取
→ 垂直抬升
→ 运输到安全空中姿态调整位
→ 固定GF225抓取TCP的XYZ
→ 完成完整三维姿态调整
→ 保持最终姿态运输到目标上方
→ 垂直下降
→ 释放
```

## 4.2 实体中转区

仅作为备用路径。

特点：

* 放下物体；
* 重新观察；
* 必要时重新抓取；
* 用于当前抓持关系不能获得最终姿态、视觉证据不足或三维旋转不可行的情况。

动作类型继续使用或重构为：

```text
extract_to_staging
regrasp_for_orientation
```

使用条件：

```text
当前抓持关系无法得到目标姿态
安全空中三维旋转扫掠失败
UR5 IK不连续
wrist关节变化超限
当前抓持点导致旋转半径过大
Qwen无法可靠判断宽面、凹槽或直角边
深度几何无法恢复三维语义轴
```

不能把实体中转区当作三维旋转的默认步骤。

# 五、重构姿态数据模型

## 5.1 区分物体姿态和夹爪姿态

禁止继续用一个 `yaw_deg` 同时表示：

* 物体长轴方向；
* GF225夹爪方向；
* tool0方向；
* 抓取方向；
* 最终释放方向。

增加明确数据：

```python
observed_object_pose = {
    "frame_id": "base_link",
    "position_m": [x, y, z],
    "orientation_xyzw": [qx, qy, qz, qw],
}

grasp_tcp_pose = {
    "frame_id": "base_link",
    "position_m": [x, y, z],
    "orientation_xyzw": [qx, qy, qz, qw],
}

target_object_pose = {
    "frame_id": "base_link",
    "position_m": [x, y, z],
    "orientation_xyzw": [qx, qy, qz, qw],
}

release_grasp_tcp_pose = {
    "frame_id": "base_link",
    "position_m": [x, y, z],
    "orientation_xyzw": [qx, qy, qz, qw],
}
```

`yaw_deg`只能作为普通平面物体的兼容字段和日志字段，不能作为特殊构件姿态求解的权威输入。

## 5.2 保存完整抓持刚体关系

抓取成功时保存：

```python
T_grasp_tcp_object = (
    inverse(T_world_grasp_tcp_at_grasp)
    @ T_world_object_at_grasp
)
```

目标物体姿态确定后反算释放夹爪姿态：

```python
T_world_grasp_tcp_release = (
    T_world_object_target
    @ inverse(T_grasp_tcp_object)
)
```

不能再只保存：

```python
grasp_object_relative_yaw_deg
```

该字段只允许用于普通yaw-only兼容路径和日志。

## 5.3 使用四元数或旋转矩阵

增加统一三维姿态工具模块，例如：

```text
tools/workflows/stack_demo/common/pose3d.py
```

至少实现：

```python
normalize_vector()
quaternion_normalize()
quaternion_multiply()
quaternion_inverse()
quaternion_from_axis_angle()
quaternion_to_matrix()
matrix_to_quaternion()
pose_to_transform()
transform_to_pose()
slerp()
rotation_error_deg()
axis_alignment_error_deg()
transform_point()
transform_vector()
```

内部统一使用：

```text
右手坐标系
四元数顺序xyzw
明确world/object/grasp_tcp/tool0坐标含义
```

禁止在核心求解中硬编码：

```python
roll_deg = 45
pitch_deg = 45
```

房梁方向改变后，绕物体长轴旋转会表现为不同的roll/pitch组合。

# 六、目标三维姿态求解

## 6.1 计算真实房梁轴

从当前场景中左右两根稳定上层支撑的几何中心计算：

```python
beam_axis_world = normalize([
    right_upper.x - left_upper.x,
    right_upper.y - left_upper.y,
    0.0,
])
```

定义水平正交轴：

```python
horizontal_cross_axis_world = normalize(
    cross([0.0, 0.0, 1.0], beam_axis_world)
)
```

禁止继续固定：

```python
roof_yaw_deg = 0.0
```

禁止只使用配置中的名义房屋方向。

若两根上层支撑尚未完成、被判定不稳定、中心无法可靠恢复或高度差超限，则不生成最终屋顶放置候选。

## 6.2 三类构件共同约束

三类构件最终必须满足：

```text
object_long_axis ∥ beam_axis_world
```

允许平行和反平行的180°等价姿态。

从等价姿态中选择：

```text
语义姿态正确
旋转扫掠安全
IK连续
总关节变化最小
wrist_2和wrist_3变化最小
MoveIt路径最短
```

## 6.3 名义±45°和实际旋转量

`±45°`表示最终结构的名义倾角候选，不表示每次机械地从当前姿态再旋转固定45°。

必须根据当前被夹持物体姿态和目标姿态计算真实旋转差：

```python
q_delta = quaternion_multiply(
    q_target_object,
    quaternion_inverse(q_current_held_object),
)
```

实际执行角可能是：

```text
20°
35°
45°
60°
或其他由当前姿态决定的角度
```

代码可以基于名义 `+45°` 和 `-45°` 构造两个最终目标姿态，再分别计算从当前姿态到目标姿态的真实旋转。

## 6.4 长方形

长方形属于完整三维旋转构件。

目标要求：

```text
长轴沿房梁轴
长×宽的宽面作为主要可见屋顶面
不能让厚度面朝上
名义倾角约+45°或-45°
```

生成两个目标姿态候选：

```python
nominal_roof_tilt_deg in (+45.0, -45.0)
```

保留满足以下条件的候选：

```text
宽面法向具有正确向上分量
厚度轴不是主要向上轴
长轴与房梁轴误差在容差内
左右两根上层支撑都有有效承托区域
放置后质心投影处于支撑区域内
```

不要通过类别名称直接假定宽面方向。必须结合Qwen视觉语义和D435i深度几何恢复宽面法向。

## 6.5 凹槽矩形

目标要求：

```text
长轴沿房梁轴
名义倾角约+45°或-45°
凹槽开口法向具有明确向下分量
```

在两个目标姿态中选择使：

```python
dot(
    groove_opening_normal_world,
    [0.0, 0.0, -1.0],
)
```

更大的候选。

凹槽方向不确定时禁止最终放置，进入重新观察或实体中转流程。

## 6.6 三角形

目标要求：

```text
长边沿房梁轴
名义倾角约+45°或-45°
指定直角边朝上
```

在两个目标姿态中选择使：

```python
dot(
    designated_right_angle_edge_world,
    [0.0, 0.0, 1.0],
)
```

更大的候选。

不能只使用二维框yaw或“尖端方向”代替完整三维边方向。

# 七、Qwen3-VL辅助识别

## 7.1 分工

YOLO负责：

* 候选检测框；
* 分割mask；
* 初始类别；
* 颜色；
* 置信度；
* 图像裁剪范围；
* 低置信度候选池。

Qwen3-VL负责：

* 在受限类别中复核语义类别；
* 判断长方形宽面是否可见；
* 判断凹槽是否存在；
* 判断凹槽开口相对相机方向；
* 判断三角形指定直角边；
* 判断长轴在图像中的方向；
* 输出不确定性和视觉证据。

代码负责：

* 深度点云；
* 三维中心；
* 三维长轴；
* 平面法向；
* 相机坐标到 `base_link`；
* 目标四元数；
* 抓持关系；
* 候选生成；
* 碰撞；
* IK；
* MoveIt；
* 完成判定。

禁止Qwen输出：

```text
机器人坐标
物体三维坐标
roll/pitch/yaw
四元数
关节角
抓取动作
放置动作
轨迹
完成声明
```

## 7.2 单次请求输入

修改当前 `perception_semantic_review.py`。当前只发送整图/overlay，不足以稳定判断小构件细节。

每批请求输入：

1. 一张带候选短ID标注的完整场景图；
2. 每个候选一张带适量上下文的放大裁剪图；
3. 每个候选的YOLO类别；
4. 每个候选的YOLO置信度；
5. 每个候选的bbox和基础几何摘要；
6. 仅允许比较的候选ID；
7. 严格受限JSON输出schema。

不必同时重复发送多张意义相同的全景图。

整图用于对象与ID对应，裁剪图用于观察：

```text
宽面
凹槽
三角形直角边
遮挡
轮廓
```

## 7.3 图像和批次限制

32K上下文下仍要限制请求规模。

配置：

```json
{
  "special_shape_vlm": {
    "maximum_candidates_per_request": 6,
    "hard_maximum_candidates_per_request": 8,
    "crop_long_edge_px": 384,
    "crop_padding_ratio": 0.15
  }
}
```

默认每批最多6个候选。

候选超过6个时拆成独立批次：

```text
批次A：完整标注图 + 候选1～6裁剪
批次B：完整标注图 + 候选7～12裁剪
```

每批都是独立无历史请求。

禁止让第二批依赖第一批assistant消息。

请求前检查估算预算：

```python
estimated_input_tokens + num_predict + reserve <= 32768
```

现有 `_estimate_input_tokens()` 只按文本字符估算，完全没有计算图像token，不能继续把它作为多图请求的完整安全证明。

增加保守图像预算估算或使用固定批次硬限制，并在日志明确记录：

```json
{
  "text_token_estimate": 0,
  "image_count": 0,
  "image_resolution_summary": [],
  "effective_num_ctx": 32768,
  "candidate_batch_size": 0
}
```

## 7.4 严格JSON输出

扩展当前只判断类别的schema。

示例：

```json
{
  "scene_revision": 12,
  "objects": [
    {
      "candidate_id": "obj_3",
      "shape_label": "rectangle",
      "shape_confidence": 0.91,
      "long_axis_image_deg": 32.0,
      "broad_face_visible": true,
      "broad_face_confidence": 0.86,
      "groove_visible": false,
      "groove_opening_direction_camera": "unknown",
      "triangle_right_angle_edge_direction_image": "unknown",
      "occlusion": "none",
      "orientation_confidence": 0.83,
      "evidence": [
        "continuous broad rectangular face",
        "no visible groove"
      ]
    }
  ]
}
```

允许值固定为：

```text
shape_label:
rectangle | concave_rectangle | triangle | square | other | uncertain

groove_opening_direction_camera:
up | down | left | right | toward_camera | away_from_camera | unknown

triangle_right_angle_edge_direction_image:
up | down | left | right | unknown

occlusion:
none | partial | severe
```

拒绝：

* 未输入的候选ID；
* 新建对象；
* 新建类别；
* 非JSON输出；
* 越界置信度；
* 重复candidate_id；
* 自相矛盾字段；
* 输出动作或坐标；
* 对未见结构给出高置信度方向。

## 7.5 YOLO与Qwen融合

每个track保存：

```python
shape_hypotheses
semantic_shape
semantic_shape_confidence
orientation_evidence
orientation_confidence
evidence_scene_revision
evidence_image_ids
```

融合规则：

```text
YOLO与Qwen一致且Qwen置信度足够
→ 接受语义类别

YOLO与Qwen冲突
→ semantic_shape_uncertain

Qwen返回uncertain
→ semantic_shape_uncertain

遮挡严重
→ semantic_shape_uncertain

语义类别与深度尺寸严重冲突
→ semantic_shape_uncertain
```

`semantic_shape_uncertain=True` 时：

* 可以清障；
* 可以移动到观察中转区；
* 不允许执行屋顶或顶层最终放置；
* 必须重新拍摄并再次复核。

单次Qwen结果不能永久覆盖跨帧状态。

# 八、从视觉证据恢复三维语义轴

Qwen输出的是图像语义，不是三维姿态。

使用D435i深度、mask、相机内参和TF完成：

1. 在候选mask内提取有效深度点；
2. 去除桌面点；
3. 去除明显背景点；
4. PCA估计三维长轴；
5. 平面拟合估计主要可见面法向；
6. 根据Qwen宽面、凹槽和直角边语义确定物体局部轴含义；
7. 将轴和法向转换到 `base_link`；
8. 保存带置信度的三维语义证据。

不能仅使用二维bbox长边生成最终旋转轴。

如果点云、Qwen语义、相机内参或TF不足，则：

```text
不生成最终放置候选
→ 生成观察中转候选
→ 放后重新观测
```

# 九、抓取候选修复

## 9.1 保留角度枚举，但增加抓取语义分级

继续扫描抓取角，但给每个候选增加：

```python
grasp_class
edge_alignment_error_deg
closing_extent_m
finger_axial_extent_m
left_axial_overhang_m
right_axial_overhang_m
axial_coverage_ok
grasp_center_offset_m
required_3d_rotation_deg
held_object_rotation_radius_m
joint_motion_cost
```

抓取分类：

```text
0：标准沿边抓取
1：另一组正交沿边抓取
2：小偏角抓取
3：角抓
4：极限中转抓取
```

排序：

```python
selection_priority = (
    grasp_class_rank,
    edge_alignment_error_deg,
    not axial_coverage_ok,
    grasp_center_offset_m,
    held_object_rotation_radius_m,
    required_3d_rotation_deg,
    joint_motion_cost,
    -clearance_margin_m,
)
```

只要存在物理可行的沿边抓取，就不能因为其他角度MoveIt代价略小而选择角抓。

## 9.2 修复长方形物理检查

不能只检查：

```python
opening_ok
contact_length_ok
```

必须分别检查：

```python
opening_ok = (
    closing_extent_m
    <= open_inner_width_m - opening_margin_m
)

axial_coverage_ok = (
    finger_axial_extent_m
    <= effective_fingertip_length_m
       + 2.0 * allowed_axial_overhang_m
)

left_axial_overhang_m <= allowed_axial_overhang_m
right_axial_overhang_m <= allowed_axial_overhang_m
```

修复错误悬出量方向，禁止：

```text
物体轴向长度越长
→ overhang反而等于0
```

以下条件全部满足才允许候选通过：

```text
闭合方向尺寸可夹持
手指轴向覆盖足够
左右悬出量均在阈值内
双指有效接触足够
抓取中心偏差可接受
```

所有阈值进入配置，不得散落魔法数字。

# 十、普通积木放置前yaw转正

普通积木仍然保持：

```text
downward
yaw-only
禁止roll
禁止pitch
```

但必须修复当前“未提供ordinary_release_gripper_yaw时保持抓取yaw直到释放”的默认逻辑。

## 10.1 普通积木正确动作链

```text
沿物理可行角度抓取
→ 垂直抬升
→ 到达远离目标结构的安全yaw调整高度
→ 保持downward
→ 只旋转yaw到目标释放姿态
→ 保持目标yaw运输到目标上方
→ 垂直下降
→ 释放
```

禁止：

```text
任意角度抓取
→ 一直保持抓取角
→ 直接放置
```

## 10.2 抓取角和放置角解耦

保存平面抓持关系：

```python
object_relative_to_gripper_yaw = normalize_yaw(
    observed_object_yaw - grasp_gripper_yaw
)
```

根据目标物体yaw反算释放夹爪yaw：

```python
release_gripper_yaw = normalize_yaw(
    target_object_yaw - object_relative_to_gripper_yaw
)
```

不能直接执行：

```python
release_gripper_yaw = grasp_yaw
```

## 10.3 房梁柱普通方块

搭房子的四个方块支撑必须与房屋结构轴对齐。

目标yaw从以下等价集合中选择：

```python
candidate_object_yaws = [
    house_axis_yaw,
    house_axis_yaw + 90.0,
    house_axis_yaw + 180.0,
    house_axis_yaw - 90.0,
]
```

对于正方形，90°等价。

根据抓持关系反算每个候选对应的释放夹爪yaw，分别执行MoveIt plan-only，选择：

```text
关节总变化最小
wrist_3变化最小
IK分支连续
下降通道安全
```

的表示。

## 10.4 颜色整理普通积木

整理任务的普通积木也应在放置前完成yaw转正。

目标方向按颜色区域或槽位坐标轴对齐：

```text
物体主轴平行于base_link X轴或Y轴
```

不要求极高角度精度，但不能继续完全不约束最终yaw。

增加适度容差，例如配置：

```json
{
  "organize": {
    "placement_yaw_tolerance_deg": 12.0
  }
}
```

正方形使用0°/90°等价姿态，选择机械臂关节变化最小的表示。

普通长方形若不属于房屋特殊构件路径，也应选择使长边与整理槽轴对齐的yaw-only姿态。

## 10.5 yaw调整位置

普通积木yaw调整必须在：

```text
远离房屋结构
远离已完成颜色区域物体
高于当前障碍物最高点
```

的位置完成。

不能在最终下降通道内执行明显yaw旋转。

# 十一、固定GF225抓取TCP的三维旋转

“保持末端不动”必须表示：

```text
GF225实际抓取TCP世界坐标不变
```

不是固定 `tool0` 原点。

三维旋转期间：

```python
fixed_grasp_tcp_position_m = [x, y, z]
```

每个插值点必须满足：

```python
norm(
    waypoint.grasp_tcp_position
    - fixed_grasp_tcp_position_m
) <= configured_tolerance
```

根据TCP目标位姿反算tool0位姿：

```python
T_world_tool0 = (
    T_world_grasp_tcp
    @ inverse(T_tool0_grasp_tcp)
)
```

因为 `tool0 -> GF225 grasp TCP` 存在0.16m偏移，所以：

```text
GF225抓取点不动
积木绕抓取点旋转
tool0位置随姿态发生补偿移动
```

测试中允许tool0位置变化，但不允许TCP位置漂移。

不要重复应用TCP补偿。

# 十二、特殊构件执行轨迹

三类构件统一使用：

```text
1. approach_pose
2. grasp_pose
3. close_gripper
4. vertical_lift_pose
5. safe_orientation_adjustment_approach_pose
6. safe_orientation_adjustment_start_pose
7. fixed_tcp_orientation_waypoints
8. orientation_adjustment_complete_pose
9. final_pre_place_pose
10. vertical_release_descent_pose
11. open_gripper
12. retreat_pose
```

关键约束：

```text
第5～8步远离房屋结构
第8步已获得完整最终放置姿态
第8～10步保持最终三维姿态
第9步以后禁止大角度三维旋转
第10步只允许垂直下降
```

增加：

```python
orientation_trajectory = {
    "mode": "fixed_grasp_tcp_3d_rotation",
    "adjustment_location": "safe_airborne_zone",
    "rotation_axis_world": [...],
    "actual_rotation_angle_deg": ...,
    "nominal_target_tilt_deg": 45.0,
    "start_orientation_xyzw": [...],
    "target_orientation_xyzw": [...],
    "fixed_grasp_tcp_position_m": [...],
    "waypoints": [...],
}
```

使用四元数SLERP插值，最大角度步长约5°。

# 十三、安全空中姿态调整位生成

不能固定使用单个硬编码位置。

根据以下信息生成候选：

```text
工作空间
机器人排除区
当前所有物体
已完成房屋结构
D435i和GF225扫掠
当前抓持物体旋转半径
UR5可达性
```

候选必须满足：

```python
distance_to_house >= configured_minimum
height >= highest_obstacle_z + rotation_radius + safety_margin
```

优先选择：

```text
从当前抓取点运输距离短
远离房屋
远离桌面边界
IK连续
旋转扫掠余量最大
```

的位置。

若没有安全空中姿态调整位，则不执行直接三维放置，转入实体中转或重新观测。

# 十四、三维运动安全检查

对每个旋转插值点检查：

* UR5 IK；
* 关节连续性；
* wrist_2变化；
* wrist_3变化；
* GF225指尖扫掠体；
* GF225掌部扫掠体；
* D435i与安装支架扫掠体；
* 被夹物体三维OBB扫掠体；
* 桌面；
* 房柱；
* 已完成屋顶；
* 其他积木；
* 工作空间边界；
* 机器人自身排除区；
* 姿态调整后的完整运输通道；
* 最终垂直下降通道。

旋转安全高度根据：

```text
抓持点到物体最远顶点的旋转半径
+ 当前最高障碍物高度
+ 安全余量
```

计算。

`+45°`和`-45°`目标候选都进行完整MoveIt plan-only。

# 十五、实体中转区修复

实体中转动作区分：

```text
change_accessibility
change_orientation_observation
regrasp_for_orientation
```

增加字段：

```python
staging_purpose
expected_next_grasp_family
expected_orientation_evidence
requires_fresh_vlm_verification
failed_direct_orientation_reason
```

只把物体平移到中转区且保持原姿态，不算姿态问题已经解决。

实体中转成功条件：

```text
物体进入中转区域
重新检测成功
Qwen语义和朝向证据更新
生成新的稳定沿边抓取
或
生成新的安全空中三维姿态调整候选
```

防止循环：

```text
同一track
同一中转点
同一抓取角
同一姿态证据
同一失败原因
```

不得重复执行。

# 十六、放置后验证

三类构件不能只验证位置、高度和yaw。

## 16.1 rectangle

验证：

```text
长轴与房梁轴误差在容差内
宽面法向具有要求的向上分量
不是厚度面朝上
最终倾角符合目标结构
左右两根上层支撑均有效
质心投影位于支撑区域
```

## 16.2 concave_rectangle

验证：

```text
长轴与房梁轴对齐
凹槽开口法向朝下
最终倾角符合目标结构
左右支撑均有效
屋顶高度符合预期
```

## 16.3 triangle

验证：

```text
长边与房梁轴对齐
指定直角边朝上
底部与屋顶有效接触
质心投影位于屋顶支撑区域
总高度符合预期
```

放置后允许三种成功证据：

```text
原track继续可见且谓词成立
新track重新绑定到该role且谓词成立
物体暂时被末端遮挡但结构几何和支撑关系成立
```

禁止强制原track立即可见。

# 十七、房屋状态与完成判定

角色状态改为：

```python
UNPLACED
REPAIRABLE
COMPLETED_VISIBLE
COMPLETED_OCCLUDED
INVALIDATED
```

进入完成状态后：

* track加入保护集合；
* 不再生成抓取候选；
* 不再生成推动候选；
* 不再作为清障对象；
* 单帧漏检不能退回UNPLACED；
* track ID变化时按结构位置和几何重新绑定。

只有明确反证才能进入 `INVALIDATED`：

```text
目标槽位明确为空
对应层高度消失
支撑关系消失
已完成物体发生大距离位移
结构总高度下降
```

`REPAIRABLE`用于：

```text
高度正确
支撑正确
位置或角度轻微超差
```

该状态只允许局部推正或小幅修正，禁止把物体抓走重新开始。

修改完成判定，不要直接使用：

```python
no_missing = not state.missing_expected_tracks
```

改为：

```python
unresolved_missing_tracks = (
    expected_tracks
    - visible_tracks
    - verified_occluded_tracks
    - geometry_rebound_tracks
    - inferred_hidden_tracks
)
```

最终完成条件包含：

```text
六个role完成
左右柱顶高度接近
屋顶同时连接左右柱
三角顶连接屋顶
总体高度达到预定高度
支撑图连通
没有unresolved_missing_tracks
当前scene revision已确认
```

# 十八、配置

在现有配置体系中加入或合并：

```json
{
  "policy": {
    "model": "qwen3-vl:8b-instruct",
    "temperature": 0.0,
    "think": false,
    "stream": false,
    "num_ctx": 32768,
    "num_predict": 768
  },
  "special_shape_vlm": {
    "enabled": true,
    "minimum_shape_confidence": 0.75,
    "minimum_orientation_confidence": 0.70,
    "crop_padding_ratio": 0.15,
    "crop_long_edge_px": 384,
    "maximum_candidates_per_request": 6,
    "hard_maximum_candidates_per_request": 8
  },
  "house_orientation": {
    "nominal_flip_deg": 45.0,
    "flip_tolerance_deg": 8.0,
    "long_axis_tolerance_deg": 8.0,
    "fixed_tcp_position_tolerance_m": 0.001,
    "orientation_interpolation_step_deg": 5.0,
    "minimum_upward_normal_component": 0.45,
    "minimum_downward_normal_component": 0.45,
    "minimum_airborne_adjustment_house_distance_m": 0.15,
    "airborne_adjustment_safety_margin_m": 0.03
  },
  "grasp": {
    "allowed_axial_overhang_m": 0.003,
    "opening_margin_m": 0.002
  },
  "organize": {
    "placement_yaw_tolerance_deg": 12.0
  }
}
```

参数名按项目现有风格统一，不得重复定义同一参数。

# 十九、测试

新增或修改测试，至少覆盖：

1. 默认有效 `num_ctx` 是32768。
2. 所有Ollama策略请求都发送32768。
3. 代码中不存在仍生效的12288或24576默认路径。
4. `rectangle`、`concave_rectangle`、`triangle` 都进入完整三维姿态路径。
5. 三类构件轨迹不能只有 `yaw_deg`。
6. 三维姿态调整发生在安全空中区域，不发生在房屋目标上方。
7. 第8步以后保持最终姿态，最终下降阶段不再大角度旋转。
8. 房梁方向不是世界X轴时，三类构件长轴仍正确对齐。
9. 长方形选择展示宽面的目标姿态。
10. 凹槽矩形选择开口向下的目标姿态。
11. 三角形选择指定直角边向上的目标姿态。
12. 实际旋转量由当前姿态到目标姿态计算，不是固定增加45°。
13. 固定GF225抓取TCP旋转时，各waypoint位置误差小于阈值。
14. tool0位置可因TCP偏移补偿发生变化。
15. 普通方块始终downward，roll/pitch不变。
16. 普通方块抓取yaw和释放yaw可以不同。
17. 普通方块在安全高度完成yaw转正后才进入最终下降。
18. 房屋支撑方块与房屋轴对齐。
19. 整理方块与颜色槽坐标轴对齐。
20. YOLO与Qwen冲突时不生成最终特殊构件放置。
21. Qwen不确定时生成重新观察或实体中转候选。
22. Qwen不能创建不存在的候选ID。
23. 单次语义请求最多使用配置数量的候选裁剪。
24. 长方形轴向尺寸超过指尖覆盖能力时物理检查失败。
25. 存在沿边抓取时不得选择角抓。
26. 放置后track ID变化仍可恢复role。
27. 单帧漏检不会清除已完成role。
28. `REPAIRABLE`物体不会被抓走。
29. 房屋总高度和支撑图参与最终完成判定。
30. 颜色整理流程回归测试通过。
31. 推动和清障流程回归测试通过。
32. MoveIt plan-only可规划安全空中固定TCP三维旋转。
33. 房屋附近旋转扫掠碰撞时正确拒绝候选。

# 二十、日志和运行产物

每轮保存：

```text
special_shape_overview.jpg
special_shape_crops/
vlm_shape_request.json
vlm_shape_response_raw.txt
vlm_shape_response_validated.json
shape_fusion.json
object_orientation_evidence.json
target_object_pose.json
grasp_tcp_object_transform.json
airborne_adjustment_candidates.json
orientation_candidates.json
orientation_sweep_checks.json
moveit_plan_results.json
ordinary_yaw_adjustment.json
post_place_3d_verification.json
house_completion_state.json
```

日志输出：

```text
effective_num_ctx
YOLO类别
Qwen类别
融合类别
语义置信度
当前物体长轴
房梁轴
目标旋转轴
名义±45°候选
实际所需旋转角
安全空中姿态调整位置
目标物体四元数
目标GF225 TCP四元数
TCP固定位置误差
tool0补偿位移
普通积木抓取yaw
普通积木目标物体yaw
普通积木释放夹爪yaw
三维扫掠拒绝原因
放后语义验证结果
```

# 二十一、实施顺序

按以下顺序修改：

1. 统一Ollama上下文为32768。
2. 增加配置一致性检查和上下文测试。
3. 建立Pose3D和刚体变换工具。
4. 修改action edge和执行链支持完整四元数。
5. 实现安全空中姿态调整位。
6. 实现固定GF225抓取TCP的三维旋转。
7. 实现三类构件目标姿态求解。
8. 修复普通积木放置前yaw转正。
9. 修复抓取物理检查和沿边排序。
10. 扩展Qwen语义复核为标注全景图加候选裁剪图。
11. 修改实体中转和失败fingerprint。
12. 修改放后验证、role持久化和完成判定。
13. 补齐配置、测试、README和日志。
14. 执行全部测试和plan-only集成验证。

# 二十二、验证命令

先记录基线，再运行修改后的测试：

```bash
python3 -m compileall tools robot_scene_pipeline
pytest -q
```

搜索并执行全部stack_demo专项测试。

检查旧上下文值：

```bash
rg -n '"num_ctx"\s*:\s*(12288|24576)|num_ctx.*(12288|24576)' \
  config tools robot_scene_pipeline tests
```

正式代码和配置中不得再有生效的旧默认值。

检查请求：

```bash
find runtime -path '*ollama_calls*' -name ollama_request.json -print
```

确认：

```json
"num_ctx": 32768
```

模型加载后执行：

```bash
ollama ps
```

确认：

```text
CONTEXT 32768
```

在不驱动机械臂情况下完成：

```text
连接D435i
连接TF
连接MoveIt
连接Ollama
夹爪不动作
机械臂不执行
运行build_house
启用Qwen语义复核
启用moveit-plan-only
```

验证每个特殊构件候选输出：

```text
完整orientation_xyzw
安全空中姿态调整位
固定TCP旋转waypoints
持物三维扫掠结果
MoveIt plan-only结果
```

验证普通方块输出：

```text
grasp_yaw
target_object_yaw
release_gripper_yaw
downward姿态检查
safe_height_yaw_adjustment
MoveIt plan-only结果
```

# 二十三、完成标准

修改完成后必须满足：

```text
所有Qwen/Ollama调用有效上下文为32768
rectangle、concave_rectangle、triangle真实执行完整三维姿态规划
三类构件长轴沿实际房梁轴
三维姿态在远离房屋的安全空中区域完成
到达房屋上方前已经获得最终放置姿态
最终下降过程不再执行大角度三维旋转
旋转期间GF225抓取TCP位置保持不变
tool0按TCP偏移正确补偿移动
长方形展示宽面
凹槽矩形凹槽朝下
三角形指定直角边朝上
Qwen只输出受限视觉语义证据
标注全景图和候选裁剪图按批次输入
YOLO/Qwen冲突不会直接执行最终放置
长方形错误抓取通过物理过滤的问题被修复
沿边抓取优先级真实生效
普通积木保持downward且在放置前完成yaw转正
普通积木抓取角与放置角真正解耦
已完成房屋部分不会因漏检再次被抓起
完成判定包含总体高度和支撑拓扑
颜色整理、推动和清障逻辑不回归
```

完成后输出：

1. 修改文件清单；
2. 每个文件的核心修改；
3. 删除的旧错误路径；
4. 上下文从12288改为32768的所有位置；
5. 新增数据结构；
6. Qwen输入输出schema；
7. 特殊构件三维姿态求解；
8. 安全空中姿态调整实现；
9. 固定TCP旋转实现；
10. 普通积木yaw转正实现；
11. 测试结果；
12. `ollama ps`上下文结果；
13. plan-only结果；
14. 尚未经过实机执行验证的风险。

不要只修改提示词。不要把“允许三维旋转”作为字符串标志。必须打通：

```text
YOLO候选
→ Qwen受限视觉语义证据
→ D435i三维语义轴
→ 当前物体姿态
→ 最终目标物体姿态
→ 安全空中姿态调整位
→ GF225目标姿态
→ 固定TCP旋转轨迹
→ 三维扫掠检查
→ 保持最终姿态运输
→ MoveIt plan-only
→ 垂直放置
→ 放后语义和结构验证
```

普通积木必须打通：

```text
可行抓取yaw
→ 保存物体相对夹爪yaw
→ 目标物体yaw
→ 释放夹爪yaw
→ 安全高度downward yaw-only调整
→ 保持目标yaw运输
→ 垂直下降和释放
→ 放后角度与位置验证
```
