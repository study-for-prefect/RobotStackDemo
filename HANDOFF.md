# RobotStackDemo 交接文档

更新时间：2026-07-13（Asia/Shanghai）

## 1. 当前仓库状态

- 工作目录：`/Users/wujl/Code/RobotStackDemo`
- 当前分支：`llm-decision-explore`
- 当前提交：`74fb4b7 explore llm decision`
- `HEAD` 与 `origin/llm-decision-explore` 一致。
- 写本交接文档之前工作树是干净的；`HANDOFF.md` 是本次新增文件。
- 最近一次完整测试：

  ```text
  python3 -m unittest discover -s tests
  Ran 208 tests
  OK
  ```

- 测试时会出现 macOS Python/LibreSSL 的 `urllib3 NotOpenSSLWarning`，当前不是测试失败原因。

## 2. 我们在做什么

我们的最终任务是：**从桌面散落积木中识别并取出任务需要的积木，完成搭房子或按颜色整理；当目标积木没有安全抓取角度或缺少足够操作空间时，先自主规划并执行清障，再重新感知并继续抓取与搭建/整理。** 清障是服务于目标抓取的恢复手段，不是独立任务，也不能破坏已完成结构。

目标是在不重写现有感知、D435i 深度几何、MoveIt、UR5 执行和 GF225 夹爪模块的前提下，完成以下闭环：

```text
用户指令
→ 确定任务家族
→ VLM 充分推理并输出严格 JSON
→ 当前场景对象/角色绑定
→ 语义、几何、碰撞、MoveIt 校验
→ 执行（必须显式开启）
→ 重新感知
→ track 重绑定
→ 下一步决策
```

当前支持的任务家族：

- `stack_blocks`：明确颜色顺序的线性堆叠。
- `build_house`：固定六角色、两列两层、屋顶、三角形结构。
- `organize_blocks`：按颜色整理为 rows、columns 或 regions。

最初的主要故障包括：

1. VLM 连续输出同一个失败动作，五次都进入几何/MoveIt，机械臂始终不动。
2. 检测 `object_id` 跨帧变化导致对象或任务角色绑定错误。
3. GF225 推动下降始终使用 0.112 m 固定矩形，过度拒绝。
4. Qwen3 把推理放在 `message.thinking`，`message.content` 为空；旧代码执行 `json.loads("")`，再把异常伪造成 stop，最终误报对象引用不存在。
5. build_house 和 organize_blocks 共用房屋 Schema/提示词。
6. 房屋 grounded plan 要求模型重复输出固定 assembly steps，但模型输出协议与验证器不一致。
7. 线性堆叠阶段允许模型重复输出缺颜色或错误长度的 `full_stack_order`。

## 3. 已经完成的功能

### 3.1 动作失败防重复

核心文件：

- `robot_scene_pipeline/action_fingerprint.py`
- `tools/workflows/stack_demo/vlm_action.py`
- `tools/workflows/stack_demo/vlm_action_loop.py`
- `tools/workflows/stack_demo/task_workflow.py`

已经实现：

- 基于稳定 `track_id` 的 `ActionFingerprint`。
- 方向量化为 `+X/-X/+Y/-Y/diagonal_1/diagonal_2/other`。
- 距离按 0.005 m 量化，yaw 按 5° 量化。
- `reason`、`confidence`、无意义的小数尾数不影响指纹。
- 每任务步骤维护失败账本、禁止指纹、策略计数。
- 黑名单动作在语义之后、几何/碰撞/MoveIt 之前被拒绝，不重复做昂贵验证。
- 第 3 次可禁止连续失败两次的动作类型，第 4 次要求改变高层策略。
- safe stop 必须有多个唯一失败指纹和多个策略；否则要求 `reobserve`。
- `nudge` 不能实现 `on_top_of` 或“推到另一个物体上方”。
- fallback 只允许从 VLM 自己已经生成的 `alternative_actions` 中选择未尝试动作，代码不会凭空生成完整候选集。

### 3.2 当前帧引用与跨帧跟踪

核心文件：

- `robot_scene_pipeline/object_tracking.py`
- `robot_scene_pipeline/scene_memory.py`

身份规则：

- 检测 ID：仅用于当前检测索引和内部单步执行。
- `object_ref`：`scene_<revision>:obj_<detector_id>`，只在当前 revision 有效。
- `track_id`：跨帧稳定身份，例如 `track_green_01`。

已经实现：

- 形状、颜色、三维中心、尺寸、bbox IoU、预测位移、角色/保护状态的匹配代价。
- 全局一对一最小代价绑定，禁止两个旧 track 绑定到同一个新检测。
- 二义性匹配标记 ambiguous；ambiguous 对象不能直接执行，必须重新观察。
- 过期 object_ref、track/ref 冲突、裸 ID 无 revision 都会被拒绝。
- 日志：`track_assignment.json`、`track_history.json`、`role_binding_history.json`。

### 3.3 GF225 分段几何与受控接触

核心文件：

- `robot_scene_pipeline/tool_swept_volume.py`
- `robot_scene_pipeline/grasp_yaw_search.py`
- `tools/workflows/stack_demo/clearance_execution.py`

推动模型默认分段：

```text
tip             z=0.000–0.025 m   width=0.025 m
upper_fingers   z=0.025–0.070 m   width=0.062 m
gripper_body    z=0.070–0.150 m   width=0.112 m
```

抓取模型使用两根实体手指，中间约 0.049 m 开口不是碰撞实体。

未保护散乱积木只有在侵入量、预计被动位移、接触数量、工作区、倾覆风险和撤离检查均通过时，才允许 `controlled_contact`。桌面、支撑物、已完成结构、protected 对象仍是硬碰撞。发生受控接触后必须重新感知。

### 3.4 统一 Ollama 客户端和 Qwen3 Thinking

核心文件：

- `robot_scene_pipeline/ollama_policy_client.py`
- `robot_scene_pipeline/vlm_action_policy.py`
- `robot_scene_pipeline/vlm_stack_policy.py`
- `robot_scene_pipeline/vlm_task_policy.py`
- `robot_scene_pipeline/llm_scene_reasoner.py`

所有 Ollama `requests.post` 已集中到 `ollama_policy_client.py`。其他策略模块不再直接访问 HTTP 响应或自行对 `message.content` 做 `json.loads`。

统一顺序：

```text
HTTP 状态
→ 响应 JSON
→ message 信封
→ thinking/content
→ done/done_reason
→ JSON 解析
→ JSON Schema
→ 业务语义
```

Qwen3/Qwen2.5 兼容：

- `--vlm-think-mode auto`：Qwen3/Qwen3-VL 发 `think=true`，Qwen2.5-VL 不发该字段。
- 始终兼容 `message.thinking` 缺失或存在。
- thinking 非空、content 合法 JSON：直接使用 content。
- thinking 非空、content 为空或非法且 `done_reason=stop`：进入 Finalization Call。
- Finalization 使用 `think=false`，保留原 system/user、图像和上一轮 thinking，只定稿 JSON，不重新分析图像。
- Ollama 不接受 assistant 消息中的 thinking 字段时，将 thinking 放入单独内部上下文消息，不丢弃推理。
- `done_reason=length` 或明显截断：扩大 `num_predict`，必要时扩大 `num_ctx`，重新 reasoning。
- HTTP、连接、超时、非法信封、thinking/content 同时为空：Backend Retry。
- 后端失败不会生成 stop、object_ref、ActionFingerprint、几何检查或 MoveIt 调用。

默认预算：

```text
stack_order          num_ctx=16384  num_predict=8192
task_contract        num_ctx=16384  num_predict=8192
grounded_task_plan   num_ctx=24576  num_predict=12288
action_proposal      num_ctx=24576  num_predict=12288
action_replan        num_ctx=24576  num_predict=12288
orientation_analysis num_ctx=32768  num_predict=16384
final_json_generation               num_predict=4096
```

新增参数：

- `--vlm-think-mode auto|on|off`
- `--vlm-num-ctx`
- `--vlm-num-predict`
- `--vlm-finalizer-num-predict`
- `--vlm-read-timeout-sec`（默认 1200）
- `--vlm-keep-alive`（默认 `1h`）
- `--vlm-max-backend-retries`（默认 3）
- `--vlm-max-budget-retries`（默认 3）
- `--unload-model-after-task`

模型会在任务开始前预热。同一任务保持 `keep_alive=1h`；只有显式 `--unload-model-after-task` 才会尝试卸载。

调用日志位于：

```text
ollama_calls/<logical-call>/ollama_request.json
ollama_calls/<logical-call>/ollama_response.json
ollama_calls/<logical-call>/ollama_thinking.txt
ollama_calls/<logical-call>/ollama_content.txt
ollama_calls/<logical-call>/ollama_diagnostics.json
model_runtime_diagnostics.json
```

### 3.5 四种计数器严格分离

```text
Backend Attempt：连接、HTTP、超时、非法信封、空消息
Budget Retry：done_reason=length、JSON 截断、预算不足
Order Attempt：四颜色实例绑定语义错误或重复
Action Attempt：合法动作 JSON 后的引用、语义、几何、碰撞、MoveIt 重规划
```

Backend 和 Budget 不进入 Action/Order 失败账本。Backend 全部失败会报告 `VLM_BACKEND_FAILED`；预算耗尽和 finalizer 失败分别报告 `TOKEN_BUDGET_EXHAUSTED`、`FINALIZATION_FAILED`。

### 3.6 任务路由与独立 Schema

核心文件：

- `robot_scene_pipeline/task_routing.py`
- `robot_scene_pipeline/task_schemas.py`
- `robot_scene_pipeline/vlm_task_policy.py`

确定性路由只确定任务家族，不替代 VLM 规划：

```text
整理 / 按颜色 / 分类 / 分组 / 归类 → organize_blocks
搭房子 / 建房子 / 房屋              → build_house
堆叠 / 叠放 / 依次向上              → stack_blocks
```

合同 Schema 已拆开：

- `BUILD_HOUSE_CONTRACT_SCHEMA` 只接受 `build_house`。
- `ORGANIZE_BLOCKS_CONTRACT_SCHEMA` 只接受 `organize_blocks`。

整理 task_contract 输入不含图像、检测对象、bbox、点云、PCA、房屋本体或六角色。任务类型与路由不一致返回 `task_type_instruction_mismatch`。

### 3.7 房屋精简 grounded plan

核心文件：

- `robot_scene_pipeline/house_grounded_adapter.py`
- `robot_scene_pipeline/house_task_definition.py`
- `robot_scene_pipeline/task_semantic_validation.py`

VLM 的房屋 grounded 输出协议是 `grounded_house_plan_v1`，只输出：

- `role_bindings`（role_id、object_ref、track_id、confidence）
- `orientation_observations`
- `reason`
- `confidence`

模型不再重复输出 label、中心、尺寸、bbox、固定 assembly steps 或完成状态。代码根据 object_ref/track_id 回填事实，并注入 `canonical_house_assembly_steps()`。

固定顺序为左右下层、左右上层、屋顶、三角形；prerequisites 使用角色 ID。模型输出 `assembly_status=completed` 或自定义步骤会被 Schema 拒绝。

形状归一化已经支持：

```text
concave
concave rectangle
concave_rectangle
→ concave_rectangle
```

roof 允许 `concave_rectangle` 或 `rectangle`，并优先 concave。

### 3.8 四层堆叠实例绑定与 OrderFingerprint

核心文件：

- `robot_scene_pipeline/stack_binding.py`
- `tools/workflows/stack_demo/scene.py`

明确的“红绿蓝黄依次向上堆叠”不再让模型自由输出任意长度 `full_stack_order`。新协议 `stack_binding_v2` 让模型为固定颜色槽位选择实例：

```json
{
  "selected_by_color": {
    "red": {"object_ref": "...", "track_id": "..."},
    "green": {"object_ref": "...", "track_id": "..."},
    "blue": {"object_ref": "...", "track_id": "..."},
    "yellow": {"object_ref": "...", "track_id": "..."}
  }
}
```

验证四槽完整、revision 正确、四个 track 唯一、实际颜色正确。OrderFingerprint 是四个颜色对应的 track 组合。

- 同一个错误绑定第二次出现：`duplicate_failed_order`，不重复完整验证。
- 每种颜色只有一个合法实例：`stack_binding_deterministic_unique_fallback`，无需模型重复确认唯一事实。
- 同色有多个实例：仍由 VLM 自主选择。
- 多个合法组合且模型未选择：`stack_binding_selection_failed`，绝不随机绑定。

### 3.9 Workspace 启动检查

动作调用前必须存在 `table_bounds` 或 `workspace_bounds`。缺失时立即返回：

```text
WORKSPACE_CONFIGURATION_MISSING
```

这不是 VLM 推理失败；模型不得生成或猜测工作区边界。

## 4. 六个运行目录的离线回归结论

测试文件：`tests/test_new_runtime_regressions.py`

### `runtime/linear_stack_qwen3_30b_20260712_184333`

- 已知正确顺序恢复为 `[red id=3, green id=1, blue id=2, yellow id=0]`。
- 每色唯一，新代码直接形成 deterministic unique binding。
- 原动作阶段 `thinking` 非空但 `content=""`；新客户端会 finalization，不再伪造成 stop 或 `selected_object_reference_not_found`。

### `runtime/build_house_qwen3_30b_20260712_184615`

- 原五次 task_contract 都是空 content，并报 `Expecting value`。
- 新流程会得到合法 JSON、明确 Backend/Budget/Finalization 失败之一，不会把异常计入五次语义重规划。
- 该记录场景本身没有明确 triangle 候选；合同问题修复后，后续可能明确报告资源不足。

### `runtime/organize_blocks_qwen3_30b_20260712_184818`

- 新路由固定为 organize_blocks。
- task_contract/grounded prompt 测试确认不包含房屋角色、本体或 `house_semantics`。

### `runtime/linear_stack_20260712_202038`

- 红、绿各一个候选；蓝、黄有多个候选。
- 因存在多个合法组合，代码不会 deterministic 随机选择，必须由 VLM 输出完整四槽绑定。
- 原三 ID、缺黄色结果现在会被拒绝并通过 OrderFingerprint 防重复。

### `runtime/build_house_20260712_202457`

- 检测标签 `concave` 已归一为 `concave_rectangle`，可绑定 roof。
- assembly steps 由代码注入，不再因模型固定步骤格式错误重试。

### `runtime/organize_blocks_20260712_202701`

- 原日志错误生成 build_house 合同。
- 新路由和 `ORGANIZE_BLOCKS_CONTRACT_SCHEMA` 会在格式与语义两层拒绝该结果。

## 5. 当前卡在哪里

代码和离线单元测试已经完成，当前卡点在外部运行条件，不是已知单元测试失败：

1. 六个旧运行目录的 scene state 都没有 `table_bounds`/`workspace_bounds`。
2. 当前 Mac 环境不是机器人 PC，没有 Ubuntu 22.04、ROS 2 Humble、真实 MoveIt planning scene、UR5 驱动和 GF225 实机。
3. 旧 Qwen3 运行没有保存原始 `message.thinking`，只能从空 content 和旧失败状态诊断，无法恢复当时的 thinking 文本。
4. 尚未连接在线 Qwen3/Qwen2.5 Ollama 服务重新完整跑三个任务。
5. 因此尚未证明三个任务都能在真实机器人环境稳定进入“首个合法动作的 MoveIt plan-only”阶段。

## 6. 下一步计划（严格按顺序）

### 第一步：配置真实 workspace

- 从现有标定/机器人工作区配置中提供可信 `table_bounds` 或 `workspace_bounds`。
- 不要从检测物体包围盒临时推断工作区，也不要让 VLM 输出边界。
- 先确认六个方向和单位均为 `base_link`、米。

### 第二步：在线 Ollama dry-run

分别运行：

1. Qwen3-VL 30B：stack、build_house、organize_blocks。
2. Qwen2.5-VL 7B：相同三任务作为回归基准。

检查：

- `ollama_response.json` 是否同时保存 thinking/content。
- 空 content 是否进入 finalizer。
- `done_reason=length` 是否只触发 Budget Retry。
- Backend/Budget 是否没有增加 Order/Action Attempt。
- organize prompt 是否完全没有房屋字段。
- 房屋 VLM 是否只输出精简 role bindings。
- stack 多实例时是否输出完整四槽绑定。

### 第三步：离线动作链和 dry-run

- 不加 `--execute`。
- 确认三个任务都能形成合法首个动作。
- 确认 workspace、object_ref、track_id、ActionFingerprint、碰撞和 controlled contact 日志完整。

### 第四步：机器人 PC 上 MoveIt plan-only

- 仍然不驱动真实 UR5。
- 确认每个任务至少一个首动作通过真实 TF、IK、planning scene 和轨迹 plan-only。
- 任何一个任务不稳定，都回到日志修复，不得直接试真实动作。

### 第五步：真实硬件（需要用户再次明确授权）

- 只有三个任务都稳定通过首动作 MoveIt plan-only 后再考虑。
- 真实执行必须显式 `--execute`，清障还必须显式 `--execute-push-clearing`。
- 先低风险单步，再闭环任务。

## 7. 绝对不要踩的坑

### 调用层

1. **不要在任何策略模块新增 `requests.post`。** 所有 Ollama 调用必须经过 `ollama_policy_client.py`。
2. **不要直接 `json.loads(message.content)`。** 必须走统一信封、done、预算、Schema 状态机。
3. **不要丢弃 `message.thinking`。** 也不要从 thinking 用正则提取 object ID、动作或位姿直接执行。
4. **不要把空 content、HTTP 异常、超时、非法 JSON 伪造成 stop。** 后端失败必须保持 `decision=None`。
5. **不要把 Backend/Budget Retry 计入 Action 或 Order Attempt。**
6. **不要为了响应速度重新把 `num_predict` 截断到 512/1024/1536。** 本阶段明确优先充分推理。
7. **不要使用 `num_predict=-1`。** 防止错误提示导致无限生成。
8. **不要根据显存从 23 GB 降到 20–21 GB 就判断模型已卸载。** 看 Ollama 状态和 `load_duration`。

### 任务语义

9. **不要让 organize_blocks 使用房屋 Schema 或房屋 system prompt。**
10. **不要把确定性任务路由扩大成代码替代 VLM 规划。** 路由只确定任务家族。
11. **不要让模型生成固定房屋 assembly steps、placed/completed 或 assembly_status。**
12. **不要把 `concave` 当作 unknown。** 必须先于 rectangle 归一为 `concave_rectangle`。
13. **不要继续用旧的任意长度 `full_stack_order` 作为在线四层协议。**
14. **多实例 stack 不得随机挑一个。** 只有组合唯一时才允许 deterministic fallback。

### 身份与动作

15. **不要把 detector object_id 当作任务级永久身份。** object_ref 是帧内，track_id 才跨帧。
16. **不要按数组顺序或仅按颜色重绑 track。** 必须一对一匹配。
17. **ambiguous track 不得执行，必须 reobserve。**
18. **不要删除或绕过 ActionFingerprint 黑名单。** 重复失败动作不能再次进入几何/MoveIt。
19. **不要允许 nudge 表示竖直堆叠。** nudge 只表示桌面平面清障。
20. **不要因为相同动作重复五次就判定场景无解。** safe stop 需要多个唯一动作和策略。

### 机器人安全

21. **不要让 VLM 猜 workspace。** 缺失就是 `WORKSPACE_CONFIGURATION_MISSING`。
22. **不要恢复单个 0.112 m 矩形作为全部 GF225 推动碰撞体。**
23. **不要把 protected、support、table 或已完成结构接触降级为 controlled contact。**
24. **不要默认执行机器人。** dry-run 和 plan-only 必须先行。
25. **没有用户新的明确授权，不得驱动真实 UR5。**

## 8. 新会话开始时建议先做的检查

```bash
cd /Users/wujl/Code/RobotStackDemo
git branch --show-current
git status --short
python3 -m unittest discover -s tests
rg -n "requests\.post" robot_scene_pipeline tools --glob '*.py'
rg -n "fail_safe_stop|vlm_json_or_call_failed" robot_scene_pipeline tools --glob '*.py'
```

预期：

- 分支为 `llm-decision-explore`。
- 除交接文档或用户后续改动外，工作树应清晰可解释。
- 208 项测试通过。
- `requests.post` 只出现在 `ollama_policy_client.py`。
- `fail_safe_stop` 只能代表合法 JSON 后的安全终止，不能出现在调用异常转换路径。

## 9. 重要测试入口

- `tests/test_ollama_policy_client.py`：thinking、finalizer、length、HTTP、超时、空消息、think 降级。
- `tests/test_policy_routing_and_grounding.py`：任务路由、独立 Schema、organize prompt 隔离、concave、房屋精简协议。
- `tests/test_stack_binding_protocol.py`：四槽、track 唯一、实际颜色、OrderFingerprint、唯一 fallback、多实例不随机。
- `tests/test_action_identity_and_contact.py`：ActionFingerprint、track、引用、分段夹爪、controlled contact、后端失败不进入动作验证。
- `tests/test_new_runtime_regressions.py`：六个新旧运行目录离线回归。
- `tests/test_vlm_replanning_loop.py`：stop/reobserve、premature stop、workspace、动作重规划。

## 10. 最重要的一句话

下一阶段不是继续重构，而是：**围绕“从散落积木中取出所需积木并完成搭房子/按颜色整理，抓取角度或空间不足时先清障”这一最终任务，先提供可信 workspace，在在线 Ollama 上跑三个任务的 dry-run，再到机器人 PC 做 MoveIt plan-only；任何异常都根据统一调用诊断和四类计数器做局部修复，绝不绕过安全链路直接执行真实 UR5。**
