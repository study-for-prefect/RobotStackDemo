# Hover and XY Calibration

This procedure separates four error sources:

1. Robot execution error: actual `tool0` minus commanded `tool0`.
2. Base-fixed visual/hand-eye error: error direction stays fixed in `base_link`.
3. Tool-local TCP error: error direction rotates with grasp yaw.
4. Workspace affine error: error changes with object position.

Measured error must use this convention:

```text
error_x = observed gripper center X - physical object center X
error_y = observed gripper center Y - physical object center Y
```

## Hover-only Check

The standalone hover command is now grouped with the other calibration tools:

```bash
python3 tools/calibration/hover_tool_offset_calibration.py \
  --label green \
  --hover-height 0.08
```

Add `--execute --yes` only after checking the detected target and robot safety
state. This workflow never descends, closes the gripper, picks, or places.

Enter measurements in millimeters. For example, if the gripper center is
3 mm toward positive base X and 2.5 mm toward positive base Y:

```text
3.0 2.5
```

## Phase 1: Yaw Test

Keep one object at the same position. Do not change offsets during collection.

```bash
python3 tools/calibration/xy_bias_diagnosis.py collect \
  --phase yaw \
  --label green \
  --output-dir runtime/xy_bias_diagnosis/yaw_green \
  --yaws 0 90 180 -90 \
  --repeats 3 \
  --hover-height 0.08 \
  --known-object-height-m 0.0235 \
  --tool-offset-base -0.015 0 0 \
  --pre-rotate-wrist-yaw-sign negative \
  --execute
```

The report separates:

- `base_error_m`
- `yaw_local_error_m`
- suggested incremental base and tool-local offset changes

## Phase 2: Workspace Test

Keep yaw at zero. Move the same object to each prompted workspace location.

```bash
python3 tools/calibration/xy_bias_diagnosis.py collect \
  --phase workspace \
  --label green \
  --output-dir runtime/xy_bias_diagnosis/workspace_green \
  --positions center left right front back \
  --workspace-yaw 0 \
  --repeats 3 \
  --hover-height 0.08 \
  --known-object-height-m 0.0235 \
  --tool-offset-base -0.015 0 0 \
  --execute
```

The report outputs a suggested absolute `affine_xy` when at least three
distinct positions are available. It is fitted from:

```text
measured_error_xy = error_affine_xy @ [visual_x, visual_y, 1]
corrected_visual_xy = visual_xy - predicted_error_xy
```

Do not combine this matrix with another active `affine_xy` without composing
the transforms explicitly.

## Offline Analysis

```bash
python3 tools/calibration/xy_bias_diagnosis.py analyze \
  runtime/xy_bias_diagnosis/yaw_green/samples.json
```

## Validation

Do not apply fitted values immediately to production grasping.

1. Apply the suggested changes to a separate validation command.
2. Test held-out yaws such as `45` and `-45` degrees.
3. Test workspace positions not used during fitting.
4. Require mean absolute XY error below the chosen tolerance.
5. Keep robot execution error separate from visual/TCP correction.

The generated corrections are incremental relative to the offsets recorded in
`samples.json`. For the yaw test, the report also prints the resulting updated
`--tool-offset-base` and `--tool-offset-yaw-local` values.

Each sample records `pointcloud_source`, `geometry_estimation_method`, and
whether depth geometry was observable. This makes
`mask_known_height_fallback` visible in the final report instead of mixing it
silently with TCP or hand-eye error.

The hover workflow stages explicit yaw by changing `wrist_3_joint` through the
short branch and verifies the reached yaw before translating over the object.
The configured UR mapping is `--pre-rotate-wrist-yaw-sign negative`. Do not
raise the joint-delta safety threshold to bypass a rejected IK branch.

Collection can be interrupted safely. Re-running the same command with the
same output directory skips trials already present in `samples.json`. The
phase, label, and collection offsets must match; otherwise use a new output
directory.
