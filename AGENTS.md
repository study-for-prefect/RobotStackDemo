# AGENTS.md

## Environment
- User Mac: macOS, Python 3.9.6
- Robot PC: Ubuntu 22.04, ROS 2 Humble, RTX 3090
- Robot: UR5
- Camera: Intel RealSense D435i
- Gripper: ViTai GF225 / DH gripper depending on project context

## Code Rules

### Main Entry

Keep one clear main entry for each workflow.

Example:

```text
tools/stack_demo_pipeline.py
```

Main entry is responsible only for:

* reading arguments
* loading config
* calling modules in order
* handling top-level errors
* writing final output

Do not put detailed perception, geometry, memory, reasoning, or execution logic directly in the main entry.

---

## File Size

Keep files small.

Target:

* normal module: under 300 lines
* complex module: under 500 lines
* main entry script: under 700 lines

If a file keeps growing, split by responsibility.

---

## Module Responsibility

Each module should do one thing.

Good:

```text
detector_runtime.py      # detection
depth_geometry.py        # depth to 3D geometry
geometry_relations.py    # spatial relations
scene_memory.py          # memory update
llm_scene_reasoner.py    # task reasoning
moveit_plan_preview.py   # robot execution
```

Bad:

```text
pipeline.py              # detection + geometry + planning + execution
utils.py                 # unrelated helper functions
```

---

## Layer Boundaries

Do not mix layers.

Perception code must not move the robot.

Geometry code must not call the camera.

Memory code must not call MoveIt.

Reasoning code must not execute robot commands.

Execution code must not run detection or LLM reasoning.

---

## Function Rules

Keep functions short.

Target:

* normal function: under 50 lines
* complex function: under 80 lines

Split when a function mixes:

* argument parsing
* validation
* computation
* file I/O
* robot control

---

## Naming

Use explicit names.

Good:

```python
target_object_id
geometry_center_m
blocking_objects
offset_base_m
```

Bad:

```python
x
tmp
data
res
```

Short names are allowed only in simple math code.

---

## Type Hints

All new public functions must use type hints.

```python
def compute_distance_m(a: Point3D, b: Point3D) -> float:
    ...
```

---

## Safety

Any code path that may move hardware must support dry-run.

Robot execution must never be the default.

Execution must require explicit user intent.

---

## Patch Policy

Prefer minimal patches.

Do not rewrite whole files.

Do not reformat unrelated code.

Do not rename unrelated files.

Add new files only when responsibility separation requires it.

Explain every new file added.

让用户最终可以有读懂代码的能力
