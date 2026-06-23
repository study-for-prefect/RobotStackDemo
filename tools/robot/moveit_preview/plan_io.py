"""Execution-plan and calibration file loading."""

import json
import os

from pymoveit2.robots import ur

def load_plan(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tcp_offset(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    offset = payload.get("tcp_offset_tool_m")
    if not isinstance(offset, list) or len(offset) != 3:
        raise RuntimeError("TCP calibration JSON must contain tcp_offset_tool_m with 3 values.")
    if payload.get("quality_pass") is not True:
        raise RuntimeError("TCP calibration quality_pass must be true: {}".format(path))
    return [float(value) for value in offset]


def load_joint_pose(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    joint_names = payload.get("joint_names")
    joint_positions = payload.get("joint_positions") or payload.get("positions")
    if not isinstance(joint_names, list) or not isinstance(joint_positions, list):
        raise RuntimeError("Joint pose JSON must contain joint_names and joint_positions.")
    if len(joint_names) != len(joint_positions):
        raise RuntimeError("Joint pose JSON joint_names and joint_positions lengths differ.")
    required = ur.joint_names()
    position_by_name = {str(name): float(position) for name, position in zip(joint_names, joint_positions)}
    missing = [name for name in required if name not in position_by_name]
    if missing:
        raise RuntimeError("Joint pose JSON missing required joints: {}".format(", ".join(missing)))
    return {
        "name": payload.get("name") or os.path.basename(path),
        "joint_names": required,
        "joint_positions": [position_by_name[name] for name in required],
    }
