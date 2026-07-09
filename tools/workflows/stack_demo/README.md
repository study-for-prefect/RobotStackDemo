# Stack Demo Workflow

`tools/workflows/stack_demo_pipeline.py` remains the single command-line entry.
This workflow is VLM-first: the model decides the initial stack structure and
each high-level action from the snapshot image plus compact detection JSON.

## Responsibility Split

| Module | Responsibility |
| --- | --- |
| `arguments.py` | Command-line options and defaults |
| `commands.py` | External process commands and observation capture |
| `scene.py` | Initial VLM stack decision, scene lookup, target reacquisition, stack estimation |
| `push_flow.py` | VLM action loop before each pick |
| `vlm_action.py` | VLM action-intent workflow glue and MoveIt preflight handoff |
| `robot_scene_pipeline/vlm_stack_policy.py` | Initial stack-order prompt, call, and validation |
| `robot_scene_pipeline/vlm_action_policy.py` | Action-intent prompt, parsing, and code-side safety validation |
| `clearance_execution.py` | Validated nudge and pick-away execution helpers |
| `pick.py` | Pick plans, motion command construction, dry-run scene simulation |
| `placement.py` | Place-on-stack geometry and safety validation |
| `app.py` | Top-level cycle orchestration and final success/failure output |

## VLM Decisions

Initial stack decision output:

```json
{
  "base_object_id": 1,
  "full_stack_order": [1, 2, 3],
  "stack_order": [2, 3],
  "structure_plan": {},
  "reason": "...",
  "confidence": 0.8
}
```

Per-step action output:

```json
{
  "action_type": "pick|nudge|pick_away|reobserve|stop",
  "object_id": 2,
  "target_object_id": 1,
  "push_direction_base": [1.0, 0.0, 0.0],
  "push_distance_m": 0.025,
  "safe_place_center_base_m": [0.20, -0.10, 0.02],
  "reason": "...",
  "confidence": 0.8
}
```

The VLM input uses the original snapshot, optional numbered overlay, bbox,
base-link object centers, dimensions, object state, protected ids, current
target, and scene memory. It does not include camera intrinsics or raw robot
control commands.

## Code Safety Gates

Code validates VLM intent before any motion:

- object ids must exist in the current observation;
- base, locked, placed, protected, or `pushable=false` objects cannot be moved;
- `nudge` direction must be a base-link unit XY vector and distance must be
  `0.01..0.05 m`;
- `nudge` end and swept path must avoid protected structure;
- `pick_away` must have a VLM-proposed `safe_place_center_base_m` that avoids
  visible objects, protected structure, future stack regions, and table bounds;
- hardware clearing actions must pass existing MoveIt preflight before motion.

Invalid JSON, unknown ids, unsafe intent, or MoveIt failure stops fail-safe.
The workflow does not fall back to geometry scores or generated candidates.

## Logs

Initial stage:

```text
initial_order_vlm/vlm_stack_decision_input.json
initial_order_vlm/vlm_stack_decision_raw.json
initial_order_vlm/vlm_stack_decision_validated.json
```

Each action step:

```text
cycle_*/vlm_action_decision_input.json
cycle_*/vlm_action_decision_raw.json
cycle_*/vlm_action_decision_validated.json
cycle_*/vlm_action_safety_report.json
cycle_*/selected_action.json
cycle_*/clearance_verification.json
cycle_*/clearance_step_XX_result.json
```

Hardware execution remains opt-in:

```bash
python3 tools/workflows/stack_demo_pipeline.py \
  ... \
  --execute \
  --execute-push-clearing
```
