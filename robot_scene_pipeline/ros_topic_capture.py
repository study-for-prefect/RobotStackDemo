"""ROS 2 RGB-D topic capture helpers.

The RealSense driver should be launched outside this project, for example with
``ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true``.  This
module only subscribes to the already-published color, aligned-depth, and
camera-info topics.
"""

import glob
import importlib
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import cv2
import numpy as np


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    frame_id: str = ""
    model: str = "plumb_bob"
    coeffs: tuple = ()


@dataclass
class RosRgbdFrame:
    frame_bgr: np.ndarray
    depth_frame: object
    intrinsics: CameraIntrinsics
    profile: dict
    color_seq: int


class TopicDepthFrame:
    """Depth image wrapper matching the small subset of rs.depth_frame we use."""

    def __init__(self, depth_m: Any) -> None:
        self.depth_m = np.asarray(depth_m, dtype=np.float32)

    def get_width(self) -> int:
        return int(self.depth_m.shape[1])

    def get_height(self) -> int:
        return int(self.depth_m.shape[0])

    def get_data(self) -> np.ndarray:
        return self.depth_m

    def get_distance(self, x: float, y: float) -> float:
        x = max(0, min(self.get_width() - 1, int(round(x))))
        y = max(0, min(self.get_height() - 1, int(round(y))))
        value = float(self.depth_m[y, x])
        return value if np.isfinite(value) and value > 0.0 else 0.0


def add_ros_topic_args(parser: Any) -> None:
    parser.add_argument(
        "--camera-source",
        choices=("ros-topic", "realsense"),
        default="ros-topic",
        help="Use ROS image topics by default. Use 'realsense' only for legacy direct pyrealsense2 capture.",
    )
    parser.add_argument("--color-topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/camera/color/camera_info")
    parser.add_argument(
        "--ros-depth-scale-m",
        type=float,
        default=0.001,
        help="Meters per raw unit for 16UC1 depth images. 32FC1 depth is treated as meters.",
    )


def _stamp_to_float(header):
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return time.time()
    return float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) * 1e-9


def _reshape_image(msg, dtype, channels):
    itemsize = np.dtype(dtype).itemsize
    row_elems = int(msg.step) // itemsize
    data = np.frombuffer(msg.data, dtype=dtype)
    if bool(getattr(msg, "is_bigendian", False)) and itemsize > 1:
        data = data.byteswap()
    rows = data.reshape(int(msg.height), row_elems)
    width = int(msg.width)
    if channels == 1:
        return rows[:, :width].copy()
    return rows[:, : width * channels].reshape(int(msg.height), width, channels).copy()


def decode_color_image(msg: Any) -> np.ndarray:
    encoding = str(msg.encoding).lower()
    if encoding in ("bgr8", "rgb8", "bgra8", "rgba8", "8uc3", "8uc4"):
        channels = 4 if encoding in ("bgra8", "rgba8", "8uc4") else 3
        image = _reshape_image(msg, np.uint8, channels)
        if encoding == "rgb8":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if encoding == "rgba8":
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if encoding in ("bgra8", "8uc4"):
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image
    if encoding in ("mono8", "8uc1"):
        return cv2.cvtColor(_reshape_image(msg, np.uint8, 1), cv2.COLOR_GRAY2BGR)
    raise ValueError("Unsupported color image encoding: {}".format(msg.encoding))


def decode_depth_image_m(msg: Any, depth_scale_m: float) -> np.ndarray:
    encoding = str(msg.encoding).lower()
    if encoding in ("16uc1", "mono16"):
        raw = _reshape_image(msg, np.uint16, 1)
        depth_m = raw.astype(np.float32) * float(depth_scale_m)
        depth_m[raw <= 0] = 0.0
        return depth_m
    if encoding == "32fc1":
        depth_m = _reshape_image(msg, np.float32, 1)
        depth_m[~np.isfinite(depth_m)] = 0.0
        depth_m[depth_m < 0.0] = 0.0
        return depth_m.astype(np.float32)
    raise ValueError("Unsupported depth image encoding: {}".format(msg.encoding))


def intrinsics_from_camera_info(msg: Any) -> CameraIntrinsics:
    k = list(msg.k)
    return CameraIntrinsics(
        width=int(msg.width),
        height=int(msg.height),
        fx=float(k[0]),
        fy=float(k[4]),
        ppx=float(k[2]),
        ppy=float(k[5]),
        frame_id=str(getattr(getattr(msg, "header", None), "frame_id", "") or ""),
        model=str(getattr(msg, "distortion_model", "plumb_bob")),
        coeffs=tuple(float(value) for value in getattr(msg, "d", [])),
    )


def _add_ros_python_paths() -> None:
    version = "python{}.{}".format(sys.version_info.major, sys.version_info.minor)
    distros = []
    if os.environ.get("ROS_DISTRO"):
        distros.append(os.environ["ROS_DISTRO"])
    distros.extend(name for name in ("humble", "iron", "jazzy") if name not in distros)

    candidates = []
    for distro in distros:
        prefix = os.path.join("/opt/ros", distro)
        candidates.extend(
            [
                os.path.join(prefix, "local", "lib", version, "dist-packages"),
                os.path.join(prefix, "lib", version, "dist-packages"),
                os.path.join(prefix, "lib", version, "site-packages"),
            ]
        )
    candidates.extend(glob.glob("/opt/ros/*/local/lib/{}/dist-packages".format(version)))
    candidates.extend(glob.glob("/opt/ros/*/lib/{}/dist-packages".format(version)))
    candidates.extend(glob.glob("/opt/ros/*/lib/{}/site-packages".format(version)))
    for path in candidates:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.append(path)


def _import_ros_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as first_error:
        if first_error.name != module_name:
            raise
    _add_ros_python_paths()
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as second_error:
        if second_error.name != module_name:
            raise
        raise RuntimeError(
            "ROS 2 Python module '{}' is not importable in this Python environment.\n"
            "If you run YOLO from conda on Ubuntu, start the shell like this first:\n"
            "  source /opt/ros/humble/setup.bash\n"
            "  conda activate yolo\n"
            "  python tools/monitoring/realtime_yolo_monitor.py\n"
            "Or run with explicit PYTHONPATH for ROS Humble Python packages. "
            "Current executable: {}".format(module_name, sys.executable)
        ) from second_error


def _ensure_rclpy_initialized():
    rclpy = _import_ros_module("rclpy")

    if not rclpy.ok():
        rclpy.init(args=None)
        return True
    return False


class RosRgbdSubscriber:
    """Subscribe to color, aligned depth, and camera-info topics."""

    def __init__(
        self,
        color_topic: str,
        depth_topic: str,
        camera_info_topic: str,
        depth_scale_m: float = 0.001,
        node_name: str = "robot_stack_rgbd_subscriber",
        shutdown_on_close: Optional[bool] = None,
    ) -> None:
        rclpy = _import_ros_module("rclpy")
        qos_profile_sensor_data = _import_ros_module("rclpy.qos").qos_profile_sensor_data
        sensor_msgs = _import_ros_module("sensor_msgs.msg")
        CameraInfo = sensor_msgs.CameraInfo
        Image = sensor_msgs.Image

        owns_rclpy = _ensure_rclpy_initialized()
        self._rclpy = rclpy
        self._shutdown_on_close = owns_rclpy if shutdown_on_close is None else bool(shutdown_on_close)
        self.node = rclpy.create_node(node_name)
        self.color_topic = str(color_topic)
        self.depth_topic = str(depth_topic)
        self.camera_info_topic = str(camera_info_topic)
        self.depth_scale_m = float(depth_scale_m)
        self.color_bgr = None
        self.depth_frame = None
        self.intrinsics = None
        self.color_seq = 0
        self.color_stamp = None
        self.depth_stamp = None
        self.color_frame_id = ""
        self.depth_frame_id = ""
        self.camera_info_frame_id = ""
        self.last_error = None
        self._lock = threading.Lock()

        self.node.create_subscription(Image, self.color_topic, self._on_color, qos_profile_sensor_data)
        self.node.create_subscription(Image, self.depth_topic, self._on_depth, qos_profile_sensor_data)
        self.node.create_subscription(CameraInfo, self.camera_info_topic, self._on_camera_info, qos_profile_sensor_data)

    def close(self) -> None:
        self.node.destroy_node()
        if self._shutdown_on_close and self._rclpy.ok():
            self._rclpy.shutdown()

    def _on_color(self, msg):
        try:
            image = decode_color_image(msg)
            with self._lock:
                self.color_bgr = image
                self.color_stamp = _stamp_to_float(msg.header)
                self.color_frame_id = str(getattr(msg.header, "frame_id", "") or "")
                self.color_seq += 1
                self.last_error = None
        except Exception as exc:
            with self._lock:
                self.last_error = "color decode failed: {}".format(exc)

    def _on_depth(self, msg):
        try:
            depth_frame = TopicDepthFrame(decode_depth_image_m(msg, self.depth_scale_m))
            with self._lock:
                self.depth_frame = depth_frame
                self.depth_stamp = _stamp_to_float(msg.header)
                self.depth_frame_id = str(getattr(msg.header, "frame_id", "") or "")
                self.last_error = None
        except Exception as exc:
            with self._lock:
                self.last_error = "depth decode failed: {}".format(exc)

    def _on_camera_info(self, msg):
        try:
            intrinsics = intrinsics_from_camera_info(msg)
            with self._lock:
                self.intrinsics = intrinsics
                self.camera_info_frame_id = intrinsics.frame_id
                self.last_error = None
        except Exception as exc:
            with self._lock:
                self.last_error = "camera_info decode failed: {}".format(exc)

    def _ready(self, require_depth):
        with self._lock:
            if self.color_bgr is None or self.intrinsics is None:
                return False
            return (not require_depth) or self.depth_frame is not None

    def ready(self, require_depth: bool = True) -> bool:
        return self._ready(require_depth)

    def wait_for_frame(
        self,
        timeout_sec: float,
        last_color_seq: Optional[int] = None,
        require_depth: bool = True,
    ) -> RosRgbdFrame:
        deadline = time.time() + float(timeout_sec)
        while time.time() < deadline:
            remaining = max(0.0, deadline - time.time())
            self._rclpy.spin_once(self.node, timeout_sec=min(0.05, remaining))
            if not self._ready(require_depth):
                continue
            if last_color_seq is not None and self.color_seq <= int(last_color_seq):
                continue
            return self.latest_frame()
        detail = self.last_error or "waiting for color/depth/camera_info"
        raise TimeoutError(
            "Timed out after {:.2f}s subscribing to {}, {}, {} ({})".format(
                float(timeout_sec),
                self.color_topic,
                self.depth_topic,
                self.camera_info_topic,
                detail,
            )
        )

    def latest_frame(self) -> RosRgbdFrame:
        with self._lock:
            depth_frame = None
            if self.depth_frame is not None:
                depth_frame = TopicDepthFrame(self.depth_frame.depth_m.copy())
            frame_bgr = self.color_bgr.copy()
            intrinsics = self.intrinsics
            color_seq = int(self.color_seq)
            color_stamp = self.color_stamp
            depth_stamp = self.depth_stamp
            color_frame_id = self.color_frame_id
            depth_frame_id = self.depth_frame_id
            camera_info_frame_id = self.camera_info_frame_id
        profile = {
            "source": "ros-topic",
            "color_topic": self.color_topic,
            "depth_topic": self.depth_topic,
            "camera_info_topic": self.camera_info_topic,
            "color_width": int(frame_bgr.shape[1]),
            "color_height": int(frame_bgr.shape[0]),
            "depth_width": None if depth_frame is None else depth_frame.get_width(),
            "depth_height": None if depth_frame is None else depth_frame.get_height(),
            "depth_available": depth_frame is not None,
            "depth_scale_m_per_unit": self.depth_scale_m,
            "color_stamp": color_stamp,
            "depth_stamp": depth_stamp,
            "color_frame_id": color_frame_id,
            "depth_frame_id": depth_frame_id,
            "camera_info_frame_id": camera_info_frame_id,
            "coordinate_frame": camera_info_frame_id or depth_frame_id or color_frame_id,
        }
        return RosRgbdFrame(
            frame_bgr=frame_bgr,
            depth_frame=depth_frame,
            intrinsics=intrinsics,
            profile=profile,
            color_seq=color_seq,
        )


def capture_rgbd_from_ros_topics(args: Any) -> Tuple[np.ndarray, TopicDepthFrame, CameraIntrinsics, dict]:
    from .realsense_capture import sharpness_score

    timeout_sec = float(getattr(args, "frame_timeout_ms", 5000)) / 1000.0
    subscriber = RosRgbdSubscriber(
        getattr(args, "color_topic", "/camera/camera/color/image_raw"),
        getattr(args, "depth_topic", "/camera/camera/aligned_depth_to_color/image_raw"),
        getattr(args, "camera_info_topic", "/camera/camera/color/camera_info"),
        depth_scale_m=getattr(args, "ros_depth_scale_m", 0.001),
        node_name="robot_scene_snapshot_topic_capture",
    )
    try:
        warmup_frames = max(0, int(getattr(args, "warmup_frames", 0)))
        stable_frames = max(1, int(getattr(args, "stable_frames", 1)))
        total = warmup_frames + stable_frames
        last_seq = None
        best_frame_bgr = None
        best_score = -1.0
        depth_samples_m = []
        intrinsics = None
        profile = None

        for index in range(total):
            frame = subscriber.wait_for_frame(timeout_sec, last_color_seq=last_seq, require_depth=True)
            last_seq = frame.color_seq
            if index < warmup_frames:
                continue
            score = sharpness_score(frame.frame_bgr)
            if score > best_score:
                best_score = score
                best_frame_bgr = frame.frame_bgr.copy()
            depth_samples_m.append(frame.depth_frame.depth_m.copy())
            intrinsics = frame.intrinsics
            profile = dict(frame.profile)

        if best_frame_bgr is None or not depth_samples_m:
            raise RuntimeError("No ROS RGB-D frame received.")

        stack = np.stack(depth_samples_m, axis=0)
        valid_count = np.count_nonzero(np.isfinite(stack) & (stack > 0.0), axis=0)
        median_depth_m = np.zeros(stack.shape[1:], dtype=np.float32)
        valid_pixels = valid_count > 0
        with np.errstate(invalid="ignore"):
            masked = np.where(stack > 0.0, stack, np.nan)
            median_depth_m[valid_pixels] = np.nanmedian(masked[:, valid_pixels], axis=0).astype(np.float32)
        median_depth_m[~np.isfinite(median_depth_m)] = 0.0
        depth_frame = TopicDepthFrame(median_depth_m)
        profile.update(
            {
                "stable_frames": stable_frames,
                "best_color_sharpness": round(float(best_score), 3),
                "aligned_depth_width": depth_frame.get_width(),
                "aligned_depth_height": depth_frame.get_height(),
            }
        )
        return best_frame_bgr, depth_frame, intrinsics, profile
    finally:
        subscriber.close()
