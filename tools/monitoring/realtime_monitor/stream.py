"""Realtime RGB-D stream adapters."""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from robot_scene_pipeline.ros_topic_capture import RosRgbdSubscriber

from .camera import start_camera


@dataclass
class RealtimeFrame:
    frame_bgr: np.ndarray
    depth_frame: object
    intrinsics: object
    profile: dict


class RealSenseRealtimeStream:
    def __init__(self, args: Any) -> None:
        import pyrealsense2 as rs

        self.rs = rs
        self.pipeline, self.profile, used, usb_type = start_camera(rs, args)
        self.color_width, self.color_height, self.depth_width, self.depth_height, self.fps = used
        self.usb_type = usb_type
        self.align = rs.align(rs.stream.color)
        for _ in range(max(0, int(args.warmup_frames))):
            self.pipeline.wait_for_frames(args.frame_timeout_ms)

    def read(self, args: Any) -> Optional[RealtimeFrame]:
        frames = self.pipeline.wait_for_frames(args.frame_timeout_ms)
        frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame:
            return None
        intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
        profile = {
            "source": "realsense",
            "color_width": self.color_width,
            "color_height": self.color_height,
            "depth_width": self.depth_width,
            "depth_height": self.depth_height,
            "fps": self.fps,
            "usb_type_descriptor": self.usb_type,
        }
        return RealtimeFrame(np.asanyarray(color_frame.get_data()), depth_frame, intrinsics, profile)

    def close(self) -> None:
        self.pipeline.stop()


class RosTopicRealtimeStream:
    def __init__(self, args: Any) -> None:
        self.subscriber = RosRgbdSubscriber(
            args.color_topic,
            args.depth_topic,
            args.camera_info_topic,
            depth_scale_m=args.ros_depth_scale_m,
            node_name="robot_stack_realtime_topic_monitor",
        )
        self.last_color_seq = None
        for _ in range(max(0, int(args.warmup_frames))):
            frame = self.subscriber.wait_for_frame(
                float(args.frame_timeout_ms) / 1000.0,
                last_color_seq=self.last_color_seq,
                require_depth=True,
            )
            self.last_color_seq = frame.color_seq

    def read(self, args: Any) -> RealtimeFrame:
        frame = self.subscriber.wait_for_frame(
            float(args.frame_timeout_ms) / 1000.0,
            last_color_seq=self.last_color_seq,
            require_depth=True,
        )
        self.last_color_seq = frame.color_seq
        return RealtimeFrame(frame.frame_bgr, frame.depth_frame, frame.intrinsics, frame.profile)

    def close(self) -> None:
        self.subscriber.close()


def start_realtime_stream(args: Any) -> Any:
    if getattr(args, "camera_source", "ros-topic") == "realsense":
        return RealSenseRealtimeStream(args)
    return RosTopicRealtimeStream(args)
