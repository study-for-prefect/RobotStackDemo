import time

import cv2
import numpy as np


class ArrayDepthFrame:
    """Small wrapper so median-filtered depth can be consumed like rs.depth_frame."""

    def __init__(self, depth_m):
        self.depth_m = np.asarray(depth_m, dtype=np.float32)

    def get_width(self):
        return int(self.depth_m.shape[1])

    def get_height(self):
        return int(self.depth_m.shape[0])

    def get_distance(self, x, y):
        x = max(0, min(self.get_width() - 1, int(round(x))))
        y = max(0, min(self.get_height() - 1, int(round(y))))
        value = float(self.depth_m[y, x])
        return value if np.isfinite(value) and value > 0.0 else 0.0


def add_realsense_args(parser):
    from .ros_topic_capture import add_ros_topic_args

    add_ros_topic_args(parser)
    parser.add_argument("--image-in", default="", help="Use an existing color image instead of capturing from RealSense.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--depth-width", type=int, default=1280)
    parser.add_argument("--depth-height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--auto-profile", action="store_true", default=True)
    parser.add_argument("--no-auto-profile", action="store_false", dest="auto_profile")
    parser.add_argument(
        "--allow-low-fps-fallback",
        action="store_true",
        default=False,
        help="Allow 15fps fallback profiles. Default keeps D435i USB3.x 30fps profiles only.",
    )
    parser.add_argument("--warmup-frames", type=int, default=45)
    parser.add_argument("--stable-frames", type=int, default=7, help="Capture N stable frames and median-fuse depth.")
    parser.add_argument("--frame-timeout-ms", type=int, default=5000)
    parser.add_argument("--rs-reset", action="store_true")
    parser.add_argument("--rs-exposure", type=float, default=-1.0)
    parser.add_argument("--rs-gain", type=float, default=-1.0)
    parser.add_argument(
        "--rs-depth-preset",
        choices=("none", "default", "high_accuracy", "high_density", "medium_density"),
        default="high_accuracy",
    )
    parser.add_argument("--rs-emitter-enabled", type=int, default=1)
    parser.add_argument(
        "--rs-laser-power",
        type=float,
        default=-1.0,
        help="Depth laser power. Negative means use the sensor maximum when supported.",
    )
    parser.add_argument("--rs-depth-postprocess", action="store_true", default=True)
    parser.add_argument("--no-rs-depth-postprocess", action="store_false", dest="rs_depth_postprocess")
    parser.add_argument("--require-usb3", action="store_true", dest="require_usb3", default=True)
    parser.add_argument("--allow-usb2", action="store_false", dest="require_usb3")


def _safe_camera_info(rs, dev, field):
    try:
        if dev.supports(field):
            return dev.get_info(field)
    except Exception:
        pass
    return ""


def _option_range(sensor, option):
    try:
        if sensor.supports(option):
            return sensor.get_option_range(option)
    except Exception:
        pass
    return None


def _set_option(sensor, option, value, name):
    try:
        if not sensor.supports(option):
            return False
        option_range = sensor.get_option_range(option)
        clipped = max(float(option_range.min), min(float(option_range.max), float(value)))
        sensor.set_option(option, clipped)
        print("RealSense option {}={}".format(name, clipped), flush=True)
        return True
    except Exception as exc:
        print("RealSense option {} failed: {}".format(name, exc), flush=True)
        return False


def configure_color_sensor(rs, profile, exposure, gain):
    for sensor in profile.get_device().query_sensors():
        name = sensor.get_info(rs.camera_info.name)
        if "rgb" not in name.lower():
            continue
        if exposure >= 0 and sensor.supports(rs.option.enable_auto_exposure):
            sensor.set_option(rs.option.enable_auto_exposure, 0)
        elif sensor.supports(rs.option.enable_auto_exposure):
            sensor.set_option(rs.option.enable_auto_exposure, 1)
        if exposure >= 0 and sensor.supports(rs.option.exposure):
            sensor.set_option(rs.option.exposure, float(exposure))
        if gain >= 0 and sensor.supports(rs.option.gain):
            sensor.set_option(rs.option.gain, float(gain))
        if sensor.supports(rs.option.enable_auto_white_balance):
            sensor.set_option(rs.option.enable_auto_white_balance, 1)


_DEPTH_PRESET_VALUES = {
    "default": 1,
    "high_accuracy": 3,
    "high_density": 4,
    "medium_density": 5,
}


def configure_depth_sensor(rs, profile, args):
    for sensor in profile.get_device().query_sensors():
        name = sensor.get_info(rs.camera_info.name)
        lower = name.lower()
        if "stereo" not in lower and "depth" not in lower:
            continue
        print("Configuring RealSense depth sensor: {}".format(name), flush=True)
        preset = getattr(args, "rs_depth_preset", "high_accuracy")
        if preset != "none" and sensor.supports(rs.option.visual_preset):
            value = _DEPTH_PRESET_VALUES.get(preset)
            if value is not None:
                _set_option(sensor, rs.option.visual_preset, value, "visual_preset({})".format(preset))
        if sensor.supports(rs.option.enable_auto_exposure):
            _set_option(sensor, rs.option.enable_auto_exposure, 1, "depth_auto_exposure")
        if sensor.supports(rs.option.emitter_enabled):
            _set_option(sensor, rs.option.emitter_enabled, int(getattr(args, "rs_emitter_enabled", 1)), "emitter_enabled")
        if sensor.supports(rs.option.laser_power):
            laser_power = float(getattr(args, "rs_laser_power", -1.0))
            if laser_power < 0.0:
                option_range = _option_range(sensor, rs.option.laser_power)
                laser_power = option_range.max if option_range is not None else laser_power
            if laser_power >= 0.0:
                _set_option(sensor, rs.option.laser_power, laser_power, "laser_power")


def depth_scale_from_profile(rs, profile):
    for sensor in profile.get_device().query_sensors():
        try:
            if sensor.supports(rs.option.depth_units):
                return float(sensor.get_depth_scale())
        except Exception:
            pass
    return 0.001


def make_depth_filters(rs):
    # No decimation: keep native pixel resolution for XY precision.
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()
    return [spatial, temporal, hole_filling]


def apply_depth_filters(depth_frame, filters):
    filtered = depth_frame
    for post_filter in filters:
        filtered = post_filter.process(filtered)
    return filtered.as_depth_frame()


def hardware_reset(rs):
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("No RealSense device found.")
    serial = devices[0].get_info(rs.camera_info.serial_number)
    print("Hardware-resetting RealSense serial={}...".format(serial), flush=True)
    devices[0].hardware_reset()
    time.sleep(8.0)


def list_realsense_devices(rs):
    ctx = rs.context()
    devices = ctx.query_devices()
    print("RealSense devices: {}".format(len(devices)), flush=True)
    for idx, dev in enumerate(devices):
        fields = {}
        for info in [
            rs.camera_info.name,
            rs.camera_info.serial_number,
            rs.camera_info.firmware_version,
            rs.camera_info.usb_type_descriptor,
            rs.camera_info.physical_port,
        ]:
            if dev.supports(info):
                fields[str(info)] = dev.get_info(info)
        print("device {}: {}".format(idx, fields), flush=True)


def active_usb_type(rs, profile):
    dev = profile.get_device()
    value = _safe_camera_info(rs, dev, rs.camera_info.usb_type_descriptor)
    return str(value or "unknown")


def stream_profiles(args):
    requested = (args.width, args.height, args.depth_width, args.depth_height, args.fps)
    candidates = [
        requested,
        (1280, 720, 1280, 720, 30),
        (1280, 720, 848, 480, 30),
        (960, 540, 848, 480, 30),
        (640, 480, 848, 480, 30),
        (640, 480, 640, 480, 30),
        (640, 480, 640, 360, 30),
    ]
    if getattr(args, "allow_low_fps_fallback", False):
        candidates.extend(
            [
                (640, 480, 640, 480, 15),
                (640, 480, 640, 360, 15),
                (640, 480, 480, 270, 15),
                (424, 240, 480, 270, 15),
            ]
        )
    output = []
    seen = set()
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
        if not args.auto_profile:
            break
    return output


def sharpness_score(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def read_aligned_rgbd(rs, pipeline, align, timeout_ms, warmup_frames, stable_frames, depth_scale, depth_filters):
    sample_count = max(1, int(stable_frames))
    best_frame_bgr = None
    best_score = -1.0
    depth_samples_m = []

    total = max(0, int(warmup_frames)) + sample_count
    for idx in range(total):
        frames = pipeline.wait_for_frames(timeout_ms)
        aligned = align.process(frames)
        color = aligned.get_color_frame()
        depth = aligned.get_depth_frame()
        if not color or not depth:
            continue
        if depth_filters:
            depth = apply_depth_filters(depth, depth_filters)
            if depth is None:
                continue
        if idx < warmup_frames:
            continue

        frame_bgr = np.asanyarray(color.get_data()).copy()
        score = sharpness_score(frame_bgr)
        if score > best_score:
            best_score = score
            best_frame_bgr = frame_bgr

        depth_z16 = np.asanyarray(depth.get_data()).astype(np.float32)
        depth_m = depth_z16 * float(depth_scale)
        depth_m[depth_z16 <= 0.0] = np.nan
        depth_samples_m.append(depth_m)

    if best_frame_bgr is None or not depth_samples_m:
        raise RuntimeError("No aligned RGB-D frame received.")

    stack = np.stack(depth_samples_m, axis=0)
    valid_count = np.count_nonzero(np.isfinite(stack), axis=0)
    median_depth_m = np.zeros(stack.shape[1:], dtype=np.float32)
    valid_pixels = valid_count > 0
    with np.errstate(invalid="ignore"):
        median_depth_m[valid_pixels] = np.nanmedian(stack[:, valid_pixels], axis=0).astype(np.float32)
    median_depth_m[valid_count <= 0] = 0.0
    median_depth_m[~np.isfinite(median_depth_m)] = 0.0
    depth_frame = ArrayDepthFrame(median_depth_m)
    return best_frame_bgr, depth_frame, best_score, sample_count


def capture_color_only(rs, args):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    print(
        "Starting RealSense color-only fallback color={}x{}@{}".format(
            args.width, args.height, args.fps
        ),
        flush=True,
    )
    profile = pipeline.start(config)
    configure_color_sensor(rs, profile, args.rs_exposure, args.rs_gain)
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()
    usb_type = active_usb_type(rs, profile)
    if getattr(args, "require_usb3", False) and not usb_type.startswith("3"):
        pipeline.stop()
        raise RuntimeError("RealSense color-only fallback is not running on USB3.x: {}".format(usb_type))
    try:
        frame_bgr = None
        best_score = -1.0
        total = max(0, args.warmup_frames) + max(1, int(getattr(args, "stable_frames", 1)))
        for idx in range(total):
            frames = pipeline.wait_for_frames(args.frame_timeout_ms)
            color = frames.get_color_frame()
            if not color:
                continue
            if idx < args.warmup_frames:
                continue
            candidate = np.asanyarray(color.get_data()).copy()
            score = sharpness_score(candidate)
            if score > best_score:
                best_score = score
                frame_bgr = candidate
        if frame_bgr is None:
            raise RuntimeError("No color frame received.")
        used_profile = {
            "color_width": args.width,
            "color_height": args.height,
            "depth_width": None,
            "depth_height": None,
            "fps": args.fps,
            "usb_type_descriptor": usb_type,
            "depth_available": False,
            "stable_frames": max(1, int(getattr(args, "stable_frames", 1))),
            "best_color_sharpness": round(float(best_score), 3),
        }
        return frame_bgr, None, intrinsics, used_profile
    finally:
        pipeline.stop()


def capture_rgbd_with_profile(rs, args, width, height, depth_width, depth_height, fps):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, fps)
    align = rs.align(rs.stream.color)

    print(
        "Starting RealSense color={}x{}@{} depth={}x{}@{}".format(
            width, height, fps, depth_width, depth_height, fps
        ),
        flush=True,
    )
    profile = pipeline.start(config)
    usb_type = active_usb_type(rs, profile)
    print("Active RealSense USB descriptor: {}".format(usb_type), flush=True)
    if getattr(args, "require_usb3", False) and not usb_type.startswith("3"):
        pipeline.stop()
        raise RuntimeError("RealSense is not running on USB3.x: {}".format(usb_type))

    configure_color_sensor(rs, profile, args.rs_exposure, args.rs_gain)
    configure_depth_sensor(rs, profile, args)
    depth_scale = depth_scale_from_profile(rs, profile)
    depth_filters = make_depth_filters(rs) if getattr(args, "rs_depth_postprocess", True) else []

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()

    try:
        frame_bgr, depth_frame, sharpness, sample_count = read_aligned_rgbd(
            rs,
            pipeline,
            align,
            args.frame_timeout_ms,
            args.warmup_frames,
            getattr(args, "stable_frames", 1),
            depth_scale,
            depth_filters,
        )
        used_profile = {
            "color_width": width,
            "color_height": height,
            "depth_width": depth_width,
            "depth_height": depth_height,
            "fps": fps,
            "usb_type_descriptor": usb_type,
            "depth_available": True,
            "depth_scale_m_per_unit": depth_scale,
            "depth_postprocess": bool(depth_filters),
            "stable_frames": sample_count,
            "best_color_sharpness": round(float(sharpness), 3),
            "depth_profile_requested": [depth_width, depth_height, fps],
            "aligned_depth_width": depth_frame.get_width(),
            "aligned_depth_height": depth_frame.get_height(),
        }
        return frame_bgr, depth_frame, intrinsics, used_profile
    finally:
        pipeline.stop()


def capture_rgbd(args):
    import pyrealsense2 as rs

    list_realsense_devices(rs)
    if args.rs_reset:
        hardware_reset(rs)

    errors = []
    for width, height, depth_width, depth_height, fps in stream_profiles(args):
        if (width, height, depth_width, depth_height, fps) != (args.width, args.height, args.depth_width, args.depth_height, args.fps):
            print(
                "Trying fallback profile color={}x{}@{} depth={}x{}@{}".format(
                    width, height, fps, depth_width, depth_height, fps
                ),
                flush=True,
            )
        try:
            return capture_rgbd_with_profile(rs, args, width, height, depth_width, depth_height, fps)
        except RuntimeError as exc:
            error = "profile color={}x{}@{} depth={}x{}@{} failed: {}".format(
                width, height, fps, depth_width, depth_height, fps, exc
            )
            print(error, flush=True)
            errors.append(error)
            time.sleep(1.0)

    try:
        return capture_color_only(rs, args)
    except RuntimeError as exc:
        errors.append("color-only fallback failed: {}".format(exc))
    raise RuntimeError("All RealSense profiles failed:\n{}".format("\n".join(errors)))
