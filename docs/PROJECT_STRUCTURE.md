# Project Structure

RobotStackDemo follows a layered structure. Public commands stay stable while
large workflows are implemented in small responsibility-focused packages.

## Core library

`robot_scene_pipeline/` contains reusable logic that does not define robot
workflow entry points:

- perception capture and detector runtime
- RGB-D and tabletop geometry
- TF transforms
- scene memory and stack-state estimation
- geometry relations and XY correction
- symbolic scene reasoning

## Public command entry points

The following files are intentionally thin and keep the existing command paths:

- `tools/workflows/stack_demo_pipeline.py`
- `tools/workflows/two_stage_visual_pick.py`
- `tools/robot/moveit_plan_preview.py`
- `tools/monitoring/realtime_yolo_monitor.py`
- `tools/diagnostics/perception_stability_probe.py`
- `tools/calibration/hover_tool_offset_calibration.py`
- `tools/calibration/xy_bias_diagnosis.py`

Each entry imports one internal package and calls its `main()` function.

## Internal workflow packages

```text
tools/
├── calibration/
│   ├── hover_calibration/       # hover target, TF, pose, command, orchestration
│   └── xy_bias/                 # sample collection and model analysis
├── diagnostics/
│   └── perception_stability/    # capture, sampling, statistics, orchestration
├── monitoring/
│   └── realtime_monitor/        # camera, TF, display, realtime loop
├── robot/
│   └── moveit_preview/          # arguments, poses, steps, trajectories, execution, TF node
└── workflows/
    ├── stack_demo/              # closed-loop stack workflow
    └── two_stage_pick/          # two-snapshot visual pick workflow
```

## Dependency direction

```text
entry scripts
    ↓
workflow/tool packages
    ↓
robot_scene_pipeline
```

Core geometry, memory, and reasoning modules must not import robot workflow
entry scripts. Hardware movement remains opt-in through explicit `--execute`
flags.

## Other directories

- `config/`: runtime configuration
- `docs/`: operator and architecture documentation
- `models/`: model artifacts and training reports
- `tests/`: hardware-independent tests
- `training/`: dataset conversion and model training commands
- `runtime/`: ignored generated run output
- `docs/archive/`: historical implementation patches kept only for reference
