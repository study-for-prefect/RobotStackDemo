# Stack Demo Workflow

`tools/workflows/stack_demo_pipeline.py` remains the single command-line
entry point. This package separates the workflow by responsibility:

| Module | Responsibility |
| --- | --- |
| `arguments.py` | Command-line options and defaults |
| `commands.py` | External process commands and observation capture |
| `scene.py` | Scene lookup, target reacquisition, memory matching, stack estimation |
| `pick.py` | Pick plans, motion command construction, dry-run scene simulation |
| `placement.py` | Place-on-stack geometry and safety validation |
| `obstruction_frontier.py` | Obstruction graph, frontier candidates, utility/easiness/risk scoring |
| `push_clearing.py` | Push-plan construction and locked-structure annotations |
| `push_flow.py` | Multi-step push execution, dry-run reporting, re-observation |
| `push_selection.py` | LLM/geometry selection among already-safe push candidates |
| `target_recovery.py` | Missing-target recovery and high-obstacle clearance relations |
| `app.py` | Top-level cycle orchestration and final success/failure output |
| `constants.py` | Shared project paths |

Object identity in this workflow is snapshot-local:

- `id` is the detector instance id inside the current observation. It is used
  for pick/place/push planning, but it is not stable across observations.
- `label` is the detector class name, such as `square green`.
- `label_id` / `class_id` are YOLO class ids. Objects of the same class share
  these values; they are not unique instance ids.

After every scoped observation, protected stack objects are rebound from the
current detector objects by label and geometry template. Do not use an old
snapshot id as proof that the same physical object still has that id.

Run the workflow through the existing entry:

```bash
python3 tools/workflows/stack_demo_pipeline.py --help
```

Hardware execution remains opt-in through `--execute`. The refactor does not
introduce another executable path or change pick/place behavior.

## Persistent Perception Server

Start perception once before running stack demo:

```bash
python3 -m robot_scene_pipeline.perception_server \
  --detector-weight models/yolo/weights/best.pt \
  --base-frame base_link \
  --camera-frame camera_color_optical_frame
```

The server keeps subscribing to `/camera/camera/color/image_raw` and
`/camera/camera/aligned_depth_to_color/image_raw`, loads YOLO once, and returns
the latest detections, depth geometry, tabletop geometry, and base-frame
coordinates on request. `stack_demo_pipeline.py` requests this server by
default through `--perception-server-url http://127.0.0.1:8765`; it no longer
starts `robot_scene_pipeline.snapshot_pipeline` as a subprocess unless
`--allow-snapshot-subprocess-fallback` is explicitly provided.

## Pick Yaw And Close Snapshot

Pick yaw selection is axis-first. The adaptive yaw search still checks obstacle
clearance, but it now prefers a feasible yaw aligned with the detected block
principal axis or its 90-degree equivalent before choosing a larger off-axis
clearance angle. This avoids cases where a square block detected near `0 deg`
is grasped at `20-30 deg` only because that angle has slightly more clearance.

The selected yaw is written as a signed 180-degree-equivalent angle, so a yaw
such as `176 deg` is reported as `-4 deg`. The analysis also records
`selected_grasp_axis_delta_deg`; values near zero mean the final grasp yaw is
aligned with the block axis.

The close target snapshot after moving above the block is disabled by default.
The stack workflow now picks from the locked first observation unless this flag
is explicitly provided:

```bash
python3 tools/workflows/stack_demo_pipeline.py \
  ... \
  --enable-second-pick-snapshot
```

When the flag is omitted, `pick_second_xy_correction.json` records
`correction_applied: false` and `fallback:
use_locked_first_observation_without_second_snapshot`. This is the preferred
default for stable tabletop stacking because it avoids stopping above the target
for another RGB-D capture.

When `--enable-second-pick-snapshot` is enabled, the approach pose used for
that snapshot is `target_position_m.z + --second-snapshot-hover-above-object-m`
and the default hover distance is `0.10 m`. The correction is no longer
"believe the second object coordinate"; it measures the current TCP-to-target
XY error from TF plus the second observation, then applies only that bounded
delta to the locked first pick plan. The report is still written to
`pick_second_xy_correction.json`.

## Locked Place Yaw And Calibration

Placement yaw is locked before the pick. With the default
`--place-yaw-strategy base`, the final held-object close observation is only a
verification snapshot; it cannot rewrite `target_position_m` or
`chosen_place_yaw_deg`. If that snapshot reports a different base id, label,
size, center, Z, or yaw, the workflow writes
`place_final_held_observation_rejected_use_locked.json` and still executes the
locked `place_on_top_plan_locked_before_pick.json`.

Use one calibration file for systematic offsets:

```bash
python3 tools/workflows/stack_demo_pipeline.py \
  ... \
  --calibration-json runtime/stack_calibration.json
```

The file may contain:

```json
{
  "affine_xy": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
  "grasp_base_bias_m": [0.0, 0.0, 0.0],
  "place_base_bias_m": [0.0, 0.0, 0.0],
  "tcp_offset_tool_m": [0.0, 0.0, 0.15],
  "max_correction_m": {"xy_m": 0.02, "z_m": 0.015}
}
```

Pick/place plans record the applied correction source, before/after target
point, base bias, XY norm, Z magnitude, and limits. `tcp_offset_tool_m` is used
only by robot motion to convert TCP goals to `tool0`; it is not mixed into
camera-frame XY compensation.

## 抓取遮挡与 Frontier 清障

抓取遮挡不再是单一 boolean。每次 pick 前会先对目标运行连续 yaw 搜索：

- 搜索范围是 `[0, 180)`，因为平行夹爪 180 度等价。
- 候选 yaw 来自目标主轴、障碍物方向、当前腕部 yaw 偏置和均匀 fallback 采样。
- 可行候选优先贴近目标主轴；只有主轴附近不可行时才偏离主轴去换取避障空间。
- 先用粗粒度 `yaw_step_deg` 找候选，再用 `local_refine_step_deg` 细化可行区间。
- 默认夹爪外宽 `0.112 m`，内宽 `0.048 m`。

`geometry_relations_before_pick.json` 里会写入 `target_grasp_analysis`，包括
`selected_grasp_yaw_deg`、`feasible_yaw_intervals_deg`、
`selected_grasp_axis_delta_deg`、`blocked_yaw_intervals_deg`、`all_grasps_blocked` 和
`blocking_objects_by_interval`。

决策顺序：

1. 如果目标被上方物体压住，返回 `remove_top_object`，不靠旋转夹爪解决。
2. 否则先选最佳抓取 yaw。
3. 如果存在可行 yaw，直接 `pick`，并把 `selected_grasp_yaw_deg` 写入 pick plan。
4. 只有目标没有安全抓取 yaw 时，才构建 `obstruction_graph`。
5. graph 从直接阻挡目标抓取的障碍开始；若障碍本身不可操作，继续展开阻挡该障碍的二级/三级障碍。
6. 未标注 `role/state` 的当前检测物体按松散可移动物体评估；`base`、
   `structure`、`locked`、`placed` 或 `pushable=false` 的物体仍然不可推。
7. 如果 frontier 中没有任何安全动作，才输出 `no_feasible_clearance_action`。

Frontier clearing 的候选动作统一评分：

- `pick_away`: 障碍物自身有可抓 yaw 时优先生成，安全放置点必须在桌面边界内且避开目标/受保护结构。
- `nudge`: 障碍物不可抓或 pick_away 不安全时，才生成小距离拨动候选，默认距离 `0.025 m`。
- `utility`: 是否直接挡住目标、挡住多个对象、移除后是否增加目标可抓 yaw。
- `easiness`: 障碍物自身可抓 yaw 数量、几何评分、操作路径复杂度代理。
- `risk`: 间接层级、碰撞/扫掠/未来放置风险。风险是硬过滤之外的排序惩罚，不能绕过安全检查。

最终按 `utility + easiness - risk` 排序。清障目标不是清空桌面，而是制造当前目标的抓取空间；如果某动作不能增加目标可抓 yaw，也不能释放关键通道，它会被降权。

Hardware execution remains separately opt-in:

```bash
python3 tools/workflows/stack_demo_pipeline.py \
  ... \
  --execute \
  --execute-push-clearing
```

Without both flags, selected clearance actions are recorded only. Real nudge
actions are preflighted through MoveIt and refused when they would damage the
locked base, placed stack, target grasp path, table bounds, or future place
regions. Before a real nudge, the gripper is closed and used as a rigid paddle;
the next ready/observation stage opens it again. Real `pick_away` actions build
an obstacle pick plan plus a safe-place plan, and both are sent through the
existing MoveIt pick/place preview before motion.

Each evaluated action receives a stable `candidate_id`. When LLM selection is
enabled, the LLM receives only safe candidates that already passed code-side
filters. The LLM may choose one `candidate_id`; it cannot introduce a new
direction, obstacle, or action. If the LLM is unavailable or returns an
unknown/unsafe id, the workflow falls back to the highest code score.

LLM push selection is enabled by default and can be disabled explicitly:

```bash
python3 tools/workflows/stack_demo_pipeline.py \
  ... \
  --disable-llm-push-selection
```

After each executed clearance action, the workflow performs a scoped
observation, rebinds current ids, rebuilds scene memory, reconstructs the
obstruction graph, and reruns candidate generation. It never executes multiple
clearance actions from the same stale scene. If the target remains blocked, it
can automatically replan more than once:

```bash
python3 tools/workflows/stack_demo_pipeline.py \
  ... \
  --max-automatic-push-clearing-attempts 4 \
  --obstruction-graph-max-depth 3 \
  --clearance-nudge-distance-m 0.025
```

Each step writes its own reports:

```text
cycle_*/obstruction_graph.json
cycle_*/obstacle_frontier_candidates.json
cycle_*/all_clearance_action_candidates.json
cycle_*/safe_clearance_candidates.json
cycle_*/selected_clearance_action.json
cycle_*/clearance_verification.json
cycle_*/clearance_step_XX_llm_selection.json
cycle_*/clearance_step_XX_result.json
cycle_*/multi_step_clearance_summary.json
```

If the target is not detected before pick, the workflow first attempts scoped
multi-view recovery. If recovery still fails, it keeps the locked first
observation template as the target region and only considers visible nearby
loose objects that are high enough to plausibly hide or block that target. These
missing-target clearing candidates still go through the same safety evaluator
before any dry-run or real push is selected.

Useful missing-target tuning parameters:

```bash
python3 tools/workflows/stack_demo_pipeline.py \
  ... \
  --missing-target-clearance-radius-m 0.10 \
  --high-block-min-top-z-delta-m 0.01
```

If no automatic push is feasible during live execution, the workflow asks the
operator to clear an obstacle, then captures one new observation and updates
scene memory. Operator confirmation is not treated as proof that the target is
safe to pick: if the target is still blocked by loose movable objects, the
workflow returns to automatic push planning; if a critical/protected object is
missing or still blocks every safe option, it refuses to continue.
