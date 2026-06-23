"""RealSense stream selection and sensor configuration."""

def option_range(sensor, option):
    try:
        if sensor.supports(option):
            return sensor.get_option_range(option)
    except Exception:
        pass
    return None


def set_option(sensor, option, value, name):
    try:
        if not sensor.supports(option):
            return False
        r = sensor.get_option_range(option)
        value = max(float(r.min), min(float(r.max), float(value)))
        sensor.set_option(option, value)
        print(f"[RS] {name}={value}", flush=True)
        return True
    except Exception as exc:
        print(f"[RS] set {name} failed: {exc}", flush=True)
        return False


def configure_sensors(rs, profile, args):
    preset_values = {
        "default": 1,
        "high_accuracy": 3,
        "high_density": 4,
        "medium_density": 5,
    }
    for sensor in profile.get_device().query_sensors():
        name = sensor.get_info(rs.camera_info.name)
        lower = name.lower()
        if "rgb" in lower:
            if args.rs_exposure >= 0 and sensor.supports(rs.option.enable_auto_exposure):
                set_option(sensor, rs.option.enable_auto_exposure, 0, "rgb_auto_exposure")
                set_option(sensor, rs.option.exposure, args.rs_exposure, "rgb_exposure")
            elif sensor.supports(rs.option.enable_auto_exposure):
                set_option(sensor, rs.option.enable_auto_exposure, 1, "rgb_auto_exposure")
            if args.rs_gain >= 0:
                set_option(sensor, rs.option.gain, args.rs_gain, "rgb_gain")
            if sensor.supports(rs.option.enable_auto_white_balance):
                set_option(sensor, rs.option.enable_auto_white_balance, 1, "rgb_auto_white_balance")
        if "stereo" in lower or "depth" in lower:
            if args.rs_depth_preset != "none" and sensor.supports(rs.option.visual_preset):
                set_option(sensor, rs.option.visual_preset, preset_values[args.rs_depth_preset], f"visual_preset({args.rs_depth_preset})")
            if sensor.supports(rs.option.enable_auto_exposure):
                set_option(sensor, rs.option.enable_auto_exposure, 1, "depth_auto_exposure")
            if sensor.supports(rs.option.emitter_enabled):
                set_option(sensor, rs.option.emitter_enabled, args.rs_emitter_enabled, "emitter_enabled")
            if sensor.supports(rs.option.laser_power):
                laser_power = args.rs_laser_power
                if laser_power < 0:
                    r = option_range(sensor, rs.option.laser_power)
                    laser_power = r.max if r is not None else laser_power
                if laser_power >= 0:
                    set_option(sensor, rs.option.laser_power, laser_power, "laser_power")


def get_usb_type(rs, profile):
    dev = profile.get_device()
    try:
        if dev.supports(rs.camera_info.usb_type_descriptor):
            return str(dev.get_info(rs.camera_info.usb_type_descriptor))
    except Exception:
        pass
    return "unknown"


def stream_profiles(args):
    requested = (args.width, args.height, args.depth_width, args.depth_height, args.fps)
    candidates = [
        requested,
        (1280, 720, 1280, 720, 30),
        (1280, 720, 848, 480, 30),
        (960, 540, 848, 480, 30),
        (640, 480, 848, 480, 30),
        (640, 480, 640, 480, 30),
    ]
    if args.allow_low_fps_fallback:
        candidates.extend(
            [
                (640, 480, 640, 480, 15),
                (640, 480, 640, 360, 15),
            ]
        )
    seen = set()
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        yield item


def start_camera(rs, args):
    errors = []
    for width, height, depth_width, depth_height, fps in stream_profiles(args):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, fps)
        try:
            print(f"[RS] starting color={width}x{height}@{fps} depth={depth_width}x{depth_height}@{fps}", flush=True)
            profile = pipeline.start(config)
            usb_type = get_usb_type(rs, profile)
            print(f"[RS] USB descriptor: {usb_type}", flush=True)
            if args.require_usb3 and not usb_type.startswith("3"):
                pipeline.stop()
                raise RuntimeError(f"RealSense is not USB3.x: {usb_type}")
            configure_sensors(rs, profile, args)
            return pipeline, profile, (width, height, depth_width, depth_height, fps), usb_type
        except RuntimeError as exc:
            try:
                pipeline.stop()
            except Exception:
                pass
            msg = f"profile color={width}x{height}@{fps} depth={depth_width}x{depth_height}@{fps} failed: {exc}"
            print("[RS] " + msg, flush=True)
            errors.append(msg)
    raise RuntimeError("All RealSense profiles failed:\n" + "\n".join(errors))
