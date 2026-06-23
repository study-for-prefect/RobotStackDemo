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
