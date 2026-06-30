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
| `push_clearing.py` | Push-plan construction and locked-structure annotations |
| `push_flow.py` | Direction evaluation, push execution, manual clearing, re-observation |
| `app.py` | Top-level cycle orchestration and final success/failure output |
| `constants.py` | Shared project paths |

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

## 抓取遮挡与推开策略

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
4. 只有所有 yaw 都被松散可移动物体挡住时，才进入 `push_clearing`。
5. 如果阻挡物包含 base、locked 或 placed structure，则不 push，返回 `replan_required`。

Geometry-based push clearing is separately opt-in:

```bash
python3 tools/workflows/stack_demo_pipeline.py \
  ... \
  --execute \
  --execute-push-clearing
```

Without both flags, push plans are recorded only. A real push is preflighted
through MoveIt and refused when its translated obstacle AABB intersects the
locked base or existing stack.

The planner evaluates away, opposite, perpendicular, and base-axis directions.
Each result records collisions, table-bound status, and score. If no direction
is feasible during live execution, the workflow asks the operator to clear the
obstacle, then captures one new observation, updates scene memory, and verifies
that the target is no longer blocked.
