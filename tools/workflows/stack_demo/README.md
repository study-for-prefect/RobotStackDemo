# Stack Demo Workflow

`tools/workflows/stack_demo_pipeline.py` remains the single command-line entry.
Its default semantic workflow supports `build_house` and `organize_blocks`.
`--legacy-linear-stack` retains the previous `stack_blocks` compatibility path.

The semantic workflow separates immutable task meaning from current detections:

```text
instruction -> task_contract -> grounded_task_plan(scene_revision)
-> task_goal_progress -> one VLM action -> validation/MoveIt -> reobserve
```

`task_contract` never contains a detection `object_id`. `grounded_task_plan`
contains temporary current-scene bindings, and each action must echo its
`scene_revision`; ID changes and ordinary motion cause reassociation/replanning,
not task failure.

`build_house` is fixed to `two_column_two_level_roof_triangle`: four distinct
square supports form two two-level columns, a concave rectangle (preferred)
or rectangle bridges the upper supports, and a triangle is centered on the
roof. The six legal role names are fixed; the old three-block house is invalid.
`organize_blocks`
uses the configured grouping rule, non-overlapping workspace regions, full
oriented footprints, boundary spacing, and separate rows/columns/grid
predicates. Required groups cannot complete while empty. Completion is
computed from a new observation, never from VLM text.

Every executable action carries `selected_object_id` and never VLM-emitted
`object_id`. A house `pick_place`/`pick_reorient_place` also carries `role_id`; an organize action
carries `group_id` and `target_region_id`. Code validates task semantics,
grounding, workspace and the dynamically rebuilt protected structure. One
adapter then copies the selected id to the legacy field immediately before
the existing physical validator or execution handoff.

After every observation, all distinct legal house role combinations are
scored. Current satisfied roles become protected ids and footprint regions;
changed detection ids replace old protection automatically, while a lost
predicate removes protection and requests repair. Large movement and
same-class reassignment are scene events rather than automatic task failure.

The VLM receives the original RGB image, numbered overlay, candidate RGB/depth
crops, bbox, dimensions, point-cloud height features, PCA axes, contour angles,
and an explicit `house_frame`. Image-up is never treated as a fixed base-link
direction. VLM output is limited to semantic orientation observations; code
fuses them with depth/contour evidence and computes target quaternions.

A wrong-face concave roof or triangle uses `pick_reorient_place`. The planner
lifts first, derives the relative quaternion from current pose, target pose and
object-to-tool grasp transform, then SLERPs at safe height. Pure yaw is rejected
for a required flip. The angle is not fixed to 45 degrees. Every waypoint is
MoveIt plan-only checked before execution, with joint-delta limits applied to
the sequential motions. Placement is followed by a fresh RGB-D observation and
orientation fusion. Triangle apex validation analyzes all three inner angles;
the right-angle vertex is not assumed to be the apex.

Ollama `format`, prompt `output_schema`, and local validators share the schemas
in `robot_scene_pipeline/task_schemas.py`.

Real execution is rejected before robot initialization whenever an offline,
mock, or recorded perception source is enabled. Dry-run remains supported.

Task logs include `task_contract_{input,raw,validated}.json`,
`grounded_task_plan_{input,raw,validated}.json`,
`task_goal_progress_revision_XX.json`, `role_assignment_candidates.json`,
`selected_role_assignment.json`, `dynamic_protection.json`, and
`task_action_semantic_validation.json`.

## Responsibility Split

| Module | Responsibility |
| --- | --- |
| `arguments.py` | Command-line options and defaults |
| `task_workflow.py` | Default task-contract orchestration and reobservation loop |
| `task_execution.py` | VLM `pick_place` plan-only validation and execution handoff |
| `robot_scene_pipeline/vlm_task_policy.py` | Task contract, temporary binding, and action VLM inputs |
| `robot_scene_pipeline/task_semantic_validation.py` | Contract and current-scene binding validation |
| `robot_scene_pipeline/task_goal_evaluator.py` | House/organization geometry predicate progress |
| `robot_scene_pipeline/task_geometry.py` | Shared oriented-footprint predicates |
| `robot_scene_pipeline/task_dynamic_protection.py` | Per-revision protected roles, ids, regions, and relations |
| `robot_scene_pipeline/task_action_adapter.py` | The only selected-id to legacy-id compatibility handoff |
| `robot_scene_pipeline/house_task_definition.py` | Canonical six-role house ontology and assembly dependencies |
| `robot_scene_pipeline/task_schemas.py` | Shared Ollama/prompt/local JSON Schemas |
| `robot_scene_pipeline/orientation_assets.py` | Full-resolution roof/triangle RGB and depth crops |
| `robot_scene_pipeline/orientation_fusion.py` | House frame, concavity and triangle-apex evidence fusion |
| `robot_scene_pipeline/reorientation_planner.py` | Safe-height quaternion SLERP reorientation planning |
| `execution_safety.py` | Shared offline-source real-execution guard |
| `commands.py` | External process commands and observation capture |
| `scene.py` | Initial VLM stack decision, scene lookup, target reacquisition, stack estimation |
| `push_flow.py` | Autonomous VLM action loop and post-decision dispatch |
| `vlm_action.py` | VLM action-intent workflow glue and MoveIt preflight handoff |
| `vlm_action_loop.py` | Same-scene VLM rejection feedback, duplicate detection, and bounded replanning |
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
  "full_stack_order": [1, 2, 3],
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

`full_stack_order` is the only authoritative order emitted by the VLM. Code
derives `base_object_id = full_stack_order[0]` and
`stack_order = full_stack_order[1:]`; conflicting legacy copies are ignored.
Missing, repeated, unknown, or instruction-inconsistent labels produce
structured feedback and another VLM request up to `--max-vlm-stack-attempts`.

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
  "contact_side": "-x",
  "direction_base": [1.0, 0.0, 0.0],
  "distance_m": 0.025,
  "gripper_yaw_rad": 0.0,
  "safe_place_center_base_m": [0.20, -0.10, 0.02],
  "predicted_scene_benefit": "...",
  "risk_assessment": "...",
  "reason": "...",
  "confidence": 0.8
}
```

The VLM input uses the original snapshot, optional depth visualization and numbered overlay, bbox,
base-link object centers, dimensions, object state, task goal, protected ids,
workspace/frame conventions, scene revision, failure history, and scene memory. It does not include camera intrinsics, grasp feasibility,
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
- `nudge` contact side must oppose its base-link unit XY direction, distance
  must be `0.01..0.05 m`, and gripper yaw must be supplied by the VLM;
- `nudge` end and swept path must avoid protected structure;
- the GF225 precheck uses yaw-oriented segmented OBBs: 25 mm tip below 25 mm,
  62 mm upper fingers from 25–70 mm, and 112 mm body from 70–150 mm; these
  installed heights are calibration defaults and must be measured on hardware;
- open-gripper grasp checks use two solid fingers and a non-solid 49 mm gap;
- table/support/protected contact is strict; small loose-object contact may pass
  only within intrusion, displacement, object-count, workspace, topple, and
  withdrawal limits, and always requires reobservation;
- pushed-object contact with an ordinary movable object is recorded as a
  recoverable contact and does not by itself reject the proposal;
- `pick_away` must have a VLM-proposed `safe_place_center_base_m` that avoids
  visible objects, protected structure, future stack regions, and table bounds;
- hardware clearing actions must pass existing MoveIt preflight before motion.
- a normal VLM-selected `pick` must pass a MoveIt plan-only preflight before
  real execution;

Invalid JSON, unknown ids, unsafe intent, tool collision, or MoveIt failure is
returned to the VLM as structured JSON. The VLM must change at least one
action field. Actions are normalized to track-based `ActionFingerprint` values
before geometry: direction, distance (5 mm), and yaw (5 degrees) are bucketed,
while prose and confidence are ignored. Failed fingerprints are hard-blacklisted.
Replanning escalates from changing the physical action, to forbidding twice-failed
action types, to requiring a new strategy. Safe-stop requires multiple unique
failed fingerprints and strategies; otherwise the control result is `reobserve`.
A fresh RGB-D observation increments the revision, invalidates old frame-local
references, and rebinds stable tracks one-to-one.
The workflow never invents geometry candidates. If the VLM supplied optional
`alternative_actions`, anti-loop fallback may select the highest-confidence
untried candidate that still passes reference and basic semantic checks.
Initial stack output is not repaired or overridden by a color-rule parser.

## Logs

Initial stage:

```text
initial_order_vlm/vlm_stack_decision_input.json
initial_order_vlm/vlm_stack_decision_raw.json
initial_order_vlm/vlm_stack_decision_validated.json
initial_order_vlm/vlm_stack_attempt_XX_{input,output,validation}.json
initial_order_vlm/vlm_stack_decision_history.json
```

Each action step:

```text
cycle_*/vlm_action_decision_input.json
cycle_*/vlm_action_decision_raw.json
cycle_*/vlm_action_decision_validated.json
cycle_*/vlm_action_safety_report.json
cycle_*/vlm_action_attempt_XX_{input,output,validation}.json
cycle_*/autonomous_action_history.json
cycle_*/action_fingerprint.json
cycle_*/failure_ledger.json
cycle_*/replanning_context.json
cycle_*/track_assignment.json
cycle_*/gripper_collision_profile.json
cycle_*/controlled_contact_evaluation.json
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

## Qwen3 reasoning and policy protocols

All Ollama chat requests are issued by `ollama_policy_client.py`. Qwen3 reasoning
preserves `message.thinking`; empty or malformed final content is converted by a
separate no-thinking finalization call. Backend retries and token-budget retries
finish before `Order Attempt` or `Action Attempt` begins. A failed transport never
creates a stop action, object reference, fingerprint, geometry check, or MoveIt call.

Task contracts are routed before the VLM call and use separate house and organization
schemas. Organization prompts contain no house definition. House grounded output is
the concise `grounded_house_plan_v1`; code resolves role refs, restores observed facts,
and injects canonical assembly steps. Four-color linear stacks use `stack_binding_v2`
and an independent OrderFingerprint blacklist.

Ollama diagnostics are stored below `ollama_calls/<logical-call>/`:

```text
ollama_request.json
ollama_response.json
ollama_thinking.txt
ollama_content.txt
ollama_diagnostics.json
```

`model_runtime_diagnostics.json` records load duration, token counts, context budget,
generation budget, keep-alive, and possible repeated model loading. Real execution
remains forbidden when workspace bounds are absent or until dry-run and MoveIt
plan-only have reached the first legal action for every task family.

`--max-vlm-action-attempts` limits same-scene rejected action proposals;
`--max-vlm-stack-attempts` limits initial semantic-order proposals. Both stop
fail-safe at their limits. `--push-tool-finger-length-m`,
`--push-tool-depth-m`, and `--push-tool-fingertip-thickness-m` describe the
conservative GF225 OBB; MoveIt remains the final collision/IK authority.
