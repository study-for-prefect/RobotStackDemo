#!/usr/bin/env python3
import argparse
import os
import time
import cv2
import numpy as np
import pyrealsense2 as rs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="/home/wxm/code/RobotStackDemo/datasets/blocks_rgb")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--mode", choices=["manual"], default="manual")
    p.add_argument("--interval-s", type=float, default=0.5)
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--show", action="store_true")
    p.add_argument("--warmup-frames", type=int, default=45)
    p.add_argument("--rs-exposure", type=float, default=-1.0)
    p.add_argument("--rs-gain", type=float, default=-1.0)
    p.add_argument("--jpeg-quality", type=int, default=100)
    p.add_argument("--format", choices=["jpg", "png"], default="jpg")
    p.add_argument("--require-usb3", action="store_true", dest="require_usb3", default=True)
    p.add_argument("--allow-usb2", action="store_false", dest="require_usb3")
    p.add_argument("--allow-low-fps-fallback", action="store_true", default=False)
    return p.parse_args()


def next_index(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    nums = []

    for name in os.listdir(output_dir):
        lower = name.lower()
        if lower.endswith(".jpg") or lower.endswith(".png"):
            stem = os.path.splitext(name)[0]
            if stem.isdigit():
                nums.append(int(stem))

    return max(nums) + 1 if nums else 1


def get_safe_path(output_dir, idx, image_ext):
    while True:
        stem = f"{idx:06d}"
        image_path = os.path.join(output_dir, stem + "." + image_ext)

        if not os.path.exists(image_path):
            return idx, stem, image_path

        idx += 1


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


def configure_color_sensor(profile, args):
    for sensor in profile.get_device().query_sensors():
        name = sensor.get_info(rs.camera_info.name)
        if "rgb" not in name.lower():
            continue
        if args.rs_exposure >= 0 and sensor.supports(rs.option.enable_auto_exposure):
            set_option(sensor, rs.option.enable_auto_exposure, 0, "rgb_auto_exposure")
            set_option(sensor, rs.option.exposure, args.rs_exposure, "rgb_exposure")
        elif sensor.supports(rs.option.enable_auto_exposure):
            set_option(sensor, rs.option.enable_auto_exposure, 1, "rgb_auto_exposure")
        if args.rs_gain >= 0:
            set_option(sensor, rs.option.gain, args.rs_gain, "rgb_gain")
        if sensor.supports(rs.option.enable_auto_white_balance):
            set_option(sensor, rs.option.enable_auto_white_balance, 1, "rgb_auto_white_balance")


def get_usb_type(profile):
    dev = profile.get_device()
    try:
        if dev.supports(rs.camera_info.usb_type_descriptor):
            return str(dev.get_info(rs.camera_info.usb_type_descriptor))
    except Exception:
        pass
    return "unknown"


def start_camera(args):
    candidates = [
        (args.width, args.height, args.fps),
        (1280, 720, 30),
        (960, 540, 30),
        (640, 480, 30),
    ]
    if args.allow_low_fps_fallback:
        candidates.append((640, 480, 15))
    seen = set()
    errors = []
    for width, height, fps in candidates:
        item = (width, height, fps)
        if item in seen:
            continue
        seen.add(item)
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        try:
            print(f"[RS] starting color={width}x{height}@{fps}", flush=True)
            profile = pipeline.start(config)
            usb_type = get_usb_type(profile)
            print(f"[RS] USB descriptor: {usb_type}", flush=True)
            if args.require_usb3 and not usb_type.startswith("3"):
                pipeline.stop()
                raise RuntimeError(f"RealSense is not USB3.x: {usb_type}")
            configure_color_sensor(profile, args)
            return pipeline, profile, (width, height, fps), usb_type
        except RuntimeError as exc:
            try:
                pipeline.stop()
            except Exception:
                pass
            msg = f"profile color={width}x{height}@{fps} failed: {exc}"
            print("[RS] " + msg, flush=True)
            errors.append(msg)
    raise RuntimeError("All RealSense color profiles failed:\n" + "\n".join(errors))


def save_image(output_dir, idx, image, image_ext, jpeg_quality):
    idx, stem, image_path = get_safe_path(output_dir, idx, image_ext)

    params = []
    if image_ext == "jpg":
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    elif image_ext == "png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 0]
    ok = cv2.imwrite(image_path, image, params)
    if not ok:
        raise RuntimeError(f"Failed to save image: {image_path}")

    print(f"saved {image_path}", flush=True)

    return idx + 1


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    pipeline, profile, used_profile, usb_type = start_camera(args)
    used_width, used_height, used_fps = used_profile
    idx = next_index(args.output_dir)
    saved_count = 0
    last_save_time = 0.0

    print(f"output_dir: {args.output_dir}", flush=True)
    print(f"mode: {args.mode}", flush=True)
    print(f"profile: {used_width}x{used_height}@{used_fps} USB:{usb_type}", flush=True)

    if args.mode == "manual":
        print("press s to save, q to quit", flush=True)
    else:
        print("continuous saving, press q to quit", flush=True)

    try:
        for _ in range(max(0, int(args.warmup_frames))):
            pipeline.wait_for_frames()

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            image = np.asanyarray(color_frame.get_data())

            vis = image.copy()
            cv2.putText(
                vis,
                f"{idx:06d} saved:{saved_count} {used_width}x{used_height}@{used_fps}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("D435i RGB collector", vis)
            key = cv2.waitKey(1) & 0xFF

            now = time.time()

            if True:
                if key == ord("s"):
                    idx = save_image(args.output_dir, idx, image, args.format, args.jpeg_quality)
                    saved_count += 1

            if key == ord("q"):
                break

            if args.max_images > 0 and saved_count >= args.max_images:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
