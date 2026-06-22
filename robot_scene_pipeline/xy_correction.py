"""Apply each configured XY correction exactly once."""

import json
import math


def load_xy_correction(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pair(value, name):
    value = [0.0, 0.0] if value is None else value
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("{} must contain exactly two values.".format(name))
    pair = [float(value[0]), float(value[1])]
    if not all(math.isfinite(item) for item in pair):
        raise ValueError("{} must contain finite values.".format(name))
    return pair


def corrected_xy(xy, correction=None, kind="grasp"):
    """Apply optional affine calibration followed by one kind-specific bias."""
    correction = correction or {}
    source = _pair(xy, "xy")
    affine = correction.get("affine_xy")
    if affine is not None:
        if not isinstance(affine, (list, tuple)) or len(affine) != 2 or any(len(row) != 3 for row in affine):
            raise ValueError("affine_xy must be a 2x3 matrix.")
        output = [
            float(affine[row][0]) * source[0]
            + float(affine[row][1]) * source[1]
            + float(affine[row][2])
            for row in range(2)
        ]
    else:
        output = list(source)
    bias = _pair(correction.get("{}_xy_bias_m".format(kind)), "{}_xy_bias_m".format(kind))
    return [output[0] + bias[0], output[1] + bias[1]]


def apply_step_xy_correction(step, geometry_xy, correction=None, kind="grasp"):
    marker = "{}_xy_correction_applied".format(kind)
    if step.get(marker):
        raise ValueError("{} XY correction was already applied.".format(kind))
    target_xy = corrected_xy(geometry_xy, correction=correction, kind=kind)
    for key in ("target_position_m", "approach_position_m"):
        if step.get(key):
            step[key][0] = round(target_xy[0], 5)
            step[key][1] = round(target_xy[1], 5)
    step[marker] = True
    step["{}_xy_bias_m".format(kind)] = [
        round(target_xy[0] - float(geometry_xy[0]), 6),
        round(target_xy[1] - float(geometry_xy[1]), 6),
    ]
    return step
