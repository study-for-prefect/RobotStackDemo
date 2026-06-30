"""Apply calibrated base-frame corrections exactly once."""

import json
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


DEFAULT_MAX_XY_CORRECTION_M = 0.02
DEFAULT_MAX_Z_CORRECTION_M = 0.015


def load_xy_correction(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Calibration JSON must be an object.")
    payload["_source_path"] = path
    return payload


def _pair(value: Optional[Sequence[float]], name: str) -> List[float]:
    value = [0.0, 0.0] if value is None else value
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("{} must contain exactly two values.".format(name))
    pair = [float(value[0]), float(value[1])]
    if not all(math.isfinite(item) for item in pair):
        raise ValueError("{} must contain finite values.".format(name))
    return pair


def _triple(value: Optional[Sequence[float]], name: str) -> List[float]:
    value = [0.0, 0.0, 0.0] if value is None else value
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("{} must contain exactly three values.".format(name))
    triple = [float(value[0]), float(value[1]), float(value[2])]
    if not all(math.isfinite(item) for item in triple):
        raise ValueError("{} must contain finite values.".format(name))
    return triple


def _affine_corrected_xy(xy: Sequence[float], correction: Mapping[str, Any]) -> List[float]:
    source = _pair(xy, "xy")
    affine = (correction or {}).get("affine_xy")
    if affine is None:
        return list(source)
    if (
        not isinstance(affine, (list, tuple))
        or len(affine) != 2
        or any(not isinstance(row, (list, tuple)) or len(row) != 3 for row in affine)
    ):
        raise ValueError("affine_xy must be a 2x3 matrix.")
    return [
        float(affine[row][0]) * source[0]
        + float(affine[row][1]) * source[1]
        + float(affine[row][2])
        for row in range(2)
    ]


def _kind_base_bias(correction: Mapping[str, Any], kind: str) -> List[float]:
    correction = correction or {}
    base_key = "{}_base_bias_m".format(kind)
    legacy_key = "{}_xy_bias_m".format(kind)
    if correction.get(base_key) is not None:
        return _triple(correction.get(base_key), base_key)
    legacy_bias = _pair(correction.get(legacy_key), legacy_key)
    return [legacy_bias[0], legacy_bias[1], 0.0]


def _correction_limits(correction: Mapping[str, Any]) -> Tuple[float, float]:
    value = (correction or {}).get("max_correction_m")
    if value is None:
        return DEFAULT_MAX_XY_CORRECTION_M, DEFAULT_MAX_Z_CORRECTION_M
    if isinstance(value, dict):
        xy = float(value.get("xy_m", value.get("xy", DEFAULT_MAX_XY_CORRECTION_M)))
        z = float(value.get("z_m", value.get("z", DEFAULT_MAX_Z_CORRECTION_M)))
    else:
        xy = float(value)
        z = float(value)
    if not math.isfinite(xy) or not math.isfinite(z) or xy < 0.0 or z < 0.0:
        raise ValueError("max_correction_m must contain finite non-negative limits.")
    return xy, z


def _check_correction_limit(
    delta: Sequence[float],
    correction: Mapping[str, Any],
    kind: str,
) -> Tuple[float, float, float, float]:
    max_xy, max_z = _correction_limits(correction)
    xy_norm = math.hypot(float(delta[0]), float(delta[1]))
    z_abs = abs(float(delta[2]))
    if xy_norm > max_xy:
        raise ValueError(
            "{} correction XY norm {:.4f} m exceeds {:.4f} m.".format(kind, xy_norm, max_xy)
        )
    if z_abs > max_z:
        raise ValueError(
            "{} correction Z {:.4f} m exceeds {:.4f} m.".format(kind, z_abs, max_z)
        )
    return xy_norm, z_abs, max_xy, max_z


def corrected_xy(
    xy: Sequence[float],
    correction: Optional[Dict[str, Any]] = None,
    kind: str = "grasp",
) -> List[float]:
    """Apply optional affine calibration followed by one kind-specific bias."""
    correction = correction or {}
    output = _affine_corrected_xy(xy, correction)
    bias = _pair(correction.get("{}_xy_bias_m".format(kind)), "{}_xy_bias_m".format(kind))
    return [output[0] + bias[0], output[1] + bias[1]]


def corrected_xyz(
    xyz: Sequence[float],
    correction: Optional[Dict[str, Any]] = None,
    kind: str = "grasp",
) -> Dict[str, Any]:
    """Apply optional XY affine plus one kind-specific base-frame XYZ bias."""
    correction = correction or {}
    source = _triple(xyz, "xyz")
    affine_xy = _affine_corrected_xy(source[:2], correction)
    base_bias = _kind_base_bias(correction, kind)
    output = [
        affine_xy[0] + base_bias[0],
        affine_xy[1] + base_bias[1],
        source[2] + base_bias[2],
    ]
    delta = [output[index] - source[index] for index in range(3)]
    xy_norm, z_abs, max_xy, max_z = _check_correction_limit(delta, correction, kind)
    return {
        "source_xyz_m": source,
        "corrected_xyz_m": output,
        "delta_m": delta,
        "base_bias_m": base_bias,
        "xy_norm_m": xy_norm,
        "z_abs_m": z_abs,
        "max_xy_correction_m": max_xy,
        "max_z_correction_m": max_z,
        "affine_xy_applied": correction.get("affine_xy") is not None,
    }


def apply_step_xy_correction(
    step: Dict[str, Any],
    geometry_xy: Sequence[float],
    correction: Optional[Dict[str, Any]] = None,
    kind: str = "grasp",
) -> Dict[str, Any]:
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


def apply_step_xyz_correction(
    step: Dict[str, Any],
    source_xyz: Sequence[float],
    correction: Optional[Dict[str, Any]] = None,
    kind: str = "grasp",
) -> Dict[str, Any]:
    marker = "{}_correction_applied".format(kind)
    if step.get(marker):
        raise ValueError("{} correction was already applied.".format(kind))
    result = corrected_xyz(source_xyz, correction=correction, kind=kind)
    corrected = result["corrected_xyz_m"]
    delta = result["delta_m"]
    for key in ("target_position_m", "release_position_m", "approach_position_m"):
        if step.get(key):
            step[key][0] = round(float(step[key][0]) + delta[0], 5)
            step[key][1] = round(float(step[key][1]) + delta[1], 5)
            step[key][2] = round(float(step[key][2]) + delta[2], 5)
    if step.get("tcp_place_xy_base_m"):
        step["tcp_place_xy_base_m"][0] = round(float(step["tcp_place_xy_base_m"][0]) + delta[0], 5)
        step["tcp_place_xy_base_m"][1] = round(float(step["tcp_place_xy_base_m"][1]) + delta[1], 5)
    if step.get("release_z_base_m") is not None:
        step["release_z_base_m"] = float(step["release_z_base_m"]) + delta[2]
    if step.get("pre_place_z_base_m") is not None:
        step["pre_place_z_base_m"] = float(step["pre_place_z_base_m"]) + delta[2]
    step[marker] = True
    step["{}_source_xyz_m".format(kind)] = [round(value, 6) for value in result["source_xyz_m"]]
    step["{}_corrected_xyz_m".format(kind)] = [round(value, 6) for value in corrected]
    step["{}_base_bias_m".format(kind)] = [round(value, 6) for value in result["base_bias_m"]]
    step["{}_correction_delta_m".format(kind)] = [round(value, 6) for value in delta]
    step["{}_correction_norm_xy_m".format(kind)] = round(result["xy_norm_m"], 6)
    step["{}_correction_abs_z_m".format(kind)] = round(result["z_abs_m"], 6)
    step["{}_max_xy_correction_m".format(kind)] = result["max_xy_correction_m"]
    step["{}_max_z_correction_m".format(kind)] = result["max_z_correction_m"]
    step["{}_affine_xy_applied".format(kind)] = bool(result["affine_xy_applied"])
    step["{}_calibration_source".format(kind)] = (correction or {}).get("_source_path", "")
    return step
