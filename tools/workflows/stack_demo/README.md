# Stack Demo Workflow

`tools/workflows/stack_demo_pipeline.py` remains the single command-line entry.
This workflow uses autonomous VLM action decisions: the model diagnoses the
scene and chooses the action, object, direction, distance, and expected benefit
from snapshot images, objective geometry, and task state.

## Responsibility Split

| Module | Responsibility |
| --- | --- |
| `arguments.py` | Command-line options and defaults |
| `commands.py` | External process commands and observation capture |
| `scene.py` | Initial VLM stack decision, scene lookup, target reacquisition, stack estimation |
| `push_flow.py` | Autonomous VLM action loop and post-decision dispatch |
| `vlm_action.py` | VLM action-intent workflow glue and MoveIt preflight handoff |
| `pick_preflight.py` | Plan-only MoveIt validation for a VLM-selected normal pick |
| `robot_scene_pipeline/vlm_stack_policy.py` | Initial stack-order prompt, call, and validation |
| `robot_scene_pipeline/vlm_action_policy.py` | Autonomous action prompt, parsing, and objective input construction |
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
  "object_bindings": [
    {"object_id": 1, "observed_label": "square red", "geometry_center_base_m": [0.30, 0.18, 0.02]},
    {"object_id": 2, "observed_label": "square green", "geometry_center_base_m": [0.36, 0.06, 0.02]},
    {"object_id": 3, "observed_label": "square blue", "geometry_center_base_m": [0.40, 0.09, 0.02]}
  ],
  "structure_plan": {},
  "reason": "...",
  "confidence": 0.8
}
```

Per-step action output:

```json
{
  "scene_problem": "...",
  "action_type": "pick|nudge|pick_away|reobserve|stop",
  "object_id": 2,
  "object_label": "square yellow",
  "object_center_base_m": [0.30, 0.10, 0.02],
  "target_object_id": 1,
  "target_object_label": "square green",
  "target_object_center_base_m": [0.30, 0.05, 0.02],
  "push_direction_base": [1.0, 0.0, 0.0],
  "push_distance_m": 0.025,
  "safe_place_center_base_m": [0.20, -0.10, 0.02],
  "predicted_scene_benefit": "...",
  "risk_assessment": "...",
  "reason": "...",
  "confidence": 0.8
}
```

The VLM input uses the original snapshot, optional numbered overlay, bbox,
base-link object centers, dimensions, object state, task goal, protected ids,
and scene memory. It does not include camera intrinsics, grasp feasibility,
blocking-object conclusions, generated action candidates, candidate scores,
recommended directions, or raw robot control commands.

`task_goal.current_plan_focus` is advisory context from the earlier VLM stack
plan. Code does not require a pick to use that object. `target_object_id` is the
task object the VLM predicts will benefit from the action; code checks that it
exists but does not replace it with a code-selected target.
Executable decisions must echo the exact detector label and base-link center
for both ids. This grounding check rejects an id that points to a different
color/instance than the VLM claims. Initial stack decisions use the same rule
through `object_bindings`.

## Code Safety Gates

Code validates VLM intent before any motion:

- object ids must exist in the current observation;
- object ids must be unique inside the current snapshot before VLM action
  validation; if duplicate ids are found before action planning, the workflow
  writes `scene_state_unique_object_ids.json` and gives VLM the reassigned ids;
- same-label objects must be selected by numbered id, bbox, and `base_link`
  center, not by color/label alone;
- base, locked, placed, protected, or `pushable=false` objects cannot be moved;
- `nudge` direction must be a base-link unit XY vector and distance must be
  `0.01..0.05 m`;
- `nudge` end and swept path must avoid protected structure;
- nudge contact is generated on the side opposite the VLM direction; code
  aligns the closed tool to the push direction plus calibrated yaw offset and
  checks the complete tool swept volume before MoveIt;
- `pick_away` must have a VLM-proposed `safe_place_center_base_m` that avoids
  visible objects, protected structure, future stack regions, and table bounds;
- hardware clearing actions must pass existing MoveIt preflight before motion.
- a normal VLM-selected `pick` must pass a MoveIt plan-only preflight before
  real execution;

Invalid JSON, unknown ids, unsafe intent, or
MoveIt failure stops fail-safe.
The workflow does not fall back to geometry scores or generated candidates.
Initial stack output is not repaired or overridden by a color-rule parser.

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

`--max-vlm-action-attempts` limits repeated `reobserve`/clearance decisions in
one pick cycle and stops fail-safe when the limit is reached.
Use `--push-tool-yaw-offset-deg` for the calibrated closed-gripper pushing-face
axis and `--push-tool-finger-length-m` for conservative swept-volume checks.
