"""TF JSON parsing and camera/base coordinate transforms."""

import json
import time
from pathlib import Path

import numpy as np

def quat_xyzw_to_rot(q):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        raise ValueError("zero quaternion")
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def as_vec3(v):
    if isinstance(v, dict):
        return np.array([float(v["x"]), float(v["y"]), float(v["z"])], dtype=np.float64)
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=np.float64)
    raise ValueError(f"cannot parse vec3: {v}")


def as_quat_xyzw(v):
    if isinstance(v, dict):
        return np.array([float(v["x"]), float(v["y"]), float(v["z"]), float(v["w"])], dtype=np.float64)
    if isinstance(v, (list, tuple)) and len(v) == 4:
        return np.array([float(v[0]), float(v[1]), float(v[2]), float(v[3])], dtype=np.float64)
    raise ValueError(f"cannot parse quat xyzw: {v}")


def matrix_from_translation_quat(t, q_xyzw):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_rot(q_xyzw)
    T[:3, 3] = as_vec3(t)
    return T


def is_4x4_matrix(v):
    try:
        a = np.asarray(v, dtype=np.float64)
        return a.shape == (4, 4)
    except Exception:
        return False


def find_transform_matrix(obj):
    """
    Accept several common tf_lookup_json formats:
    - {"translation": {"x":..}, "rotation": {"x":..,"y":..,"z":..,"w":..}}
    - {"transform": {"translation": ..., "rotation": ...}}
    - {"matrix": [[...]*4]*4}
    - nested equivalents.
    """
    if isinstance(obj, dict):
        for key in (
            "matrix",
            "transform_matrix",
            "T",
            "T_base_camera",
            "base_T_camera",
            "tf_matrix",
            "homogeneous_matrix",
        ):
            if key in obj and is_4x4_matrix(obj[key]):
                return np.asarray(obj[key], dtype=np.float64)

        node = obj.get("transform", obj)
        if isinstance(node, dict):
            trans = None
            rot = None
            for k in ("translation", "trans", "position", "xyz"):
                if k in node:
                    trans = node[k]
                    break
            for k in ("rotation", "quaternion", "quat", "q", "quaternion_xyzw", "rotation_xyzw", "q_xyzw"):
                if k in node:
                    rot = node[k]
                    break
            if trans is not None and rot is not None:
                return matrix_from_translation_quat(trans, as_quat_xyzw(rot))

            # Some tools flatten fields.
            flat_t_keys = ("x", "y", "z")
            flat_q_keys = ("qx", "qy", "qz", "qw")
            if all(k in node for k in flat_t_keys) and all(k in node for k in flat_q_keys):
                t = [node["x"], node["y"], node["z"]]
                q = [node["qx"], node["qy"], node["qz"], node["qw"]]
                return matrix_from_translation_quat(t, q)

        for value in obj.values():
            try:
                return find_transform_matrix(value)
            except Exception:
                pass

    if is_4x4_matrix(obj):
        return np.asarray(obj, dtype=np.float64)

    raise ValueError("no transform matrix found in json")


class TfJsonCache:
    def __init__(self, path, reload_s, base_frame=None, camera_frame=None):
        self.path = Path(path) if path else None
        self.reload_s = float(reload_s)
        self.base_frame = str(base_frame or "").lstrip("/")
        self.camera_frame = str(camera_frame or "").lstrip("/")
        self.last_check = 0.0
        self.last_mtime = None
        self.T = None
        self.error = "not loaded"

    def _frame_matches(self, actual, requested):
        actual = str(actual or "").lstrip("/")
        requested = str(requested or "").lstrip("/")
        return bool(actual and requested and (actual == requested or actual.endswith("/" + requested)))

    def _validate_frames(self, data):
        parent_frame = data.get("parent_frame") if isinstance(data, dict) else None
        child_frame = data.get("child_frame") if isinstance(data, dict) else None
        if self.base_frame and parent_frame and not self._frame_matches(parent_frame, self.base_frame):
            raise ValueError(
                "parent_frame mismatch: expected {}, got {}".format(self.base_frame, parent_frame)
            )
        if self.camera_frame and child_frame and not self._frame_matches(child_frame, self.camera_frame):
            raise ValueError(
                "child_frame mismatch: expected {}, got {}".format(self.camera_frame, child_frame)
            )

    def update(self, force=False):
        if self.path is None:
            self.error = "disabled"
            return self.T

        now = time.time()
        if not force and now - self.last_check < self.reload_s:
            return self.T
        self.last_check = now

        try:
            if not self.path.exists():
                self.error = f"missing {self.path}"
                return self.T

            mtime = self.path.stat().st_mtime
            if not force and self.last_mtime == mtime and self.T is not None:
                self.error = None
                return self.T

            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            self._validate_frames(data)
            self.T = find_transform_matrix(data)
            self.last_mtime = mtime
            self.error = None
            print(f"[TF] loaded {self.path}", flush=True)
        except Exception as exc:
            self.error = str(exc)

        return self.T


def sample_depth_m(depth_frame, cx, cy, radius, min_depth_m, max_depth_m):
    if depth_frame is None:
        return 0.0

    width = depth_frame.get_width()
    height = depth_frame.get_height()
    xs = range(max(0, cx - radius), min(width - 1, cx + radius) + 1)
    ys = range(max(0, cy - radius), min(height - 1, cy + radius) + 1)

    vals = []
    for y in ys:
        for x in xs:
            d = float(depth_frame.get_distance(int(x), int(y)))
            if min_depth_m <= d <= max_depth_m:
                vals.append(d)

    if not vals:
        return 0.0
    return float(np.median(np.asarray(vals, dtype=np.float32)))


def optical_to_camera_link(p_optical):
    # ROS optical frame: x right, y down, z forward
    # ROS camera_link convention: x forward, y left, z up
    x, y, z = p_optical
    return np.array([z, -x, -y], dtype=np.float64)


def transform_point(T_dst_src, p_src):
    p = np.ones(4, dtype=np.float64)
    p[:3] = p_src
    q = T_dst_src @ p
    return q[:3]
