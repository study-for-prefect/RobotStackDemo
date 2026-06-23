"""Detection filtering and visualization helpers."""

import cv2
import numpy as np

def box_intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def auto_ignore_zone(width, height):
    return [int(width * 0.86), int(height * 0.84), int(width), int(height)]


def finite_xyz(p):
    if p is None:
        return None
    a = np.asarray(p, dtype=np.float64).reshape(-1)
    if a.shape[0] < 3 or not np.all(np.isfinite(a[:3])):
        return None
    return a[:3]


def known_height_m(args):
    h = float(getattr(args, "known_block_height_m", 0.0) or 0.0)
    return h if h > 0 else None


def object_height_for_display(det, args):
    known = known_height_m(args)
    dims = det.get("dimensions_m")
    measured = None
    if isinstance(dims, (list, tuple)) and len(dims) >= 3:
        try:
            measured = float(dims[2])
        except Exception:
            measured = None
    if measured is not None and np.isfinite(measured) and measured > 0:
        return max(measured, known) if known is not None else measured
    return known


def top_z_from_geometry(det, args):
    center = finite_xyz(det.get("geometry_center_m"))
    height = object_height_for_display(det, args)
    if center is None or height is None:
        return None
    return float(max(center[2] + height / 2.0, height))


def top_z_fallback_from_sample(det, args):
    p_base = finite_xyz(det.get("p_base"))
    height = known_height_m(args)
    if p_base is None or height is None:
        return None
    # The bbox center is only a surface sample. Treat it as an approximate
    # object-center fallback so the displayed top Z is not below a 25 mm block.
    return float(max(p_base[2] + height / 2.0, height))


def draw_lines(img, lines, x, y, color, scale=0.52, thickness=2, line_h=20):
    yy = y
    for line in lines:
        cv2.putText(
            img,
            line,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        yy += line_h
