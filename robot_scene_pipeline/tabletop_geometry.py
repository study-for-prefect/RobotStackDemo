"""Estimate a tabletop plane and object footprint geometry from aligned depth."""

import math

import numpy as np

from .instance_pointcloud import (
    build_object_pointcloud,
    deproject_depth_mask,
    dilate_mask,
    is_realsense_intrinsics,
    object_mask,
    resize_mask_to_image,
    transform_points,
)


def add_tabletop_args(parser):
    parser.add_argument(
        "--estimate-tabletop",
        action="store_true",
        help="Estimate table plane plus per-object point-cloud dimensions and yaw.",
    )
    parser.add_argument("--plane-point-stride", type=int, default=4)
    parser.add_argument("--plane-distance-threshold-m", type=float, default=0.006)
    parser.add_argument("--plane-ransac-iterations", type=int, default=400)
    parser.add_argument("--plane-min-inliers", type=int, default=300)
    parser.add_argument("--plane-max-depth-m", type=float, default=2.0)
    parser.add_argument("--plane-min-up-alignment", type=float, default=0.70)
    parser.add_argument("--object-point-stride", type=int, default=2)
    parser.add_argument("--object-min-height-m", type=float, default=0.006)
    parser.add_argument("--object-max-height-m", type=float, default=0.50)
    parser.add_argument("--object-min-points", type=int, default=40)
    parser.add_argument("--object-mask-erode-px", type=int, default=2)
    parser.add_argument("--object-mask-dilate-fallback-px", type=int, default=4)
    parser.add_argument("--object-outlier-percentile", type=float, default=2.0)
    parser.add_argument(
        "--object-height-bin-m",
        type=float,
        default=0.003,
        help="Height bin size used to find the dominant top-surface cluster; not an object-height clamp.",
    )
    parser.add_argument(
        "--object-top-coverage-ratio",
        type=float,
        default=0.18,
        help="Minimum UV footprint coverage for accepting a height bin as an object top surface.",
    )
    parser.add_argument(
        "--object-support-ring-expand-px",
        type=int,
        default=24,
        help="Pixel expansion around an object mask used to estimate the local support surface, e.g. a book top.",
    )
    parser.add_argument(
        "--object-support-exclude-dilate-px",
        type=int,
        default=6,
        help="Mask dilation excluded from local support-surface estimation.",
    )
    parser.add_argument(
        "--object-support-bin-m",
        type=float,
        default=0.003,
        help="Z bin size for local support-surface estimation.",
    )
    parser.add_argument(
        "--object-support-min-points",
        type=int,
        default=80,
        help="Minimum surrounding depth points required for local support-surface estimation.",
    )
    parser.add_argument(
        "--known-object-height-m",
        type=float,
        default=0.0,
        help="Optional object-height prior used only when mask depth contains no points above the table. Default disables guessing.",
    )
    parser.add_argument("--yaw-min-aspect-ratio", type=float, default=1.20)


def deproject_depth_roi(depth_frame, intrinsics, roi=None, stride=1, max_depth_m=2.0):
    width = int(depth_frame.get_width())
    height = int(depth_frame.get_height())
    if roi is None:
        x1, y1, x2, y2 = 0, 0, width - 1, height - 1
    else:
        x1, y1, x2, y2 = roi
        x1 = max(0, min(width - 1, int(round(x1))))
        y1 = max(0, min(height - 1, int(round(y1))))
        x2 = max(0, min(width - 1, int(round(x2))))
        y2 = max(0, min(height - 1, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 3), dtype=float)

    fx = float(intrinsics.fx)
    fy = float(intrinsics.fy)
    ppx = float(intrinsics.ppx)
    ppy = float(intrinsics.ppy)
    try:
        import pyrealsense2 as rs
    except ImportError:
        rs = None
    use_rs_deproject = is_realsense_intrinsics(intrinsics, rs)
    points = []
    for y in range(y1, y2 + 1, max(1, int(stride))):
        for x in range(x1, x2 + 1, max(1, int(stride))):
            depth = float(depth_frame.get_distance(x, y))
            if depth <= 0 or depth > float(max_depth_m):
                continue
            if use_rs_deproject:
                points.append(rs.rs2_deproject_pixel_to_point(intrinsics, [float(x), float(y)], depth))
            else:
                points.append([(x - ppx) * depth / fx, (y - ppy) * depth / fy, depth])
    if not points:
        return np.empty((0, 3), dtype=float)
    return np.asarray(points, dtype=float)


def transform_points(points, matrix=None, point_mode="optical-to-camera-link"):
    points = np.asarray(points, dtype=float)
    if not len(points):
        return points.reshape((-1, 3))
    if matrix is None:
        return points.copy()
    if point_mode == "optical-to-camera-link":
        points = np.column_stack([points[:, 2], -points[:, 0], -points[:, 1]])
    elif point_mode != "direct":
        raise ValueError("Unsupported point mode: {}".format(point_mode))
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=float)])
    return homogeneous.dot(np.asarray(matrix, dtype=float).T)[:, :3]


def orient_normal(normal, orientation_hint):
    normal = np.asarray(normal, dtype=float)
    hint = np.asarray(orientation_hint, dtype=float) if orientation_hint is not None else None
    if hint is not None and np.dot(normal, hint) < 0:
        normal = -normal
    return normal


def fit_plane_ransac(
    points,
    distance_threshold_m=0.006,
    iterations=400,
    min_inliers=300,
    up_hint=None,
    min_up_alignment=0.0,
    orientation_hint=None,
    random_seed=7,
):
    points = np.asarray(points, dtype=float)
    if len(points) < max(3, int(min_inliers)):
        raise ValueError("Not enough points for table plane: {}.".format(len(points)))
    rng = np.random.RandomState(random_seed)
    best_mask = None
    best_count = 0
    best_rms = float("inf")
    up = None
    if up_hint is not None:
        up = np.asarray(up_hint, dtype=float)
        up /= np.linalg.norm(up)

    for _ in range(max(1, int(iterations))):
        sample = points[rng.choice(len(points), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        if up is not None and abs(float(np.dot(normal, up))) < float(min_up_alignment):
            continue
        d = -float(np.dot(normal, sample[0]))
        distances = np.abs(points.dot(normal) + d)
        mask = distances <= float(distance_threshold_m)
        count = int(np.count_nonzero(mask))
        if count < int(min_inliers):
            continue
        rms = float(np.sqrt(np.mean(distances[mask] ** 2)))
        if count > best_count or (count == best_count and rms < best_rms):
            best_mask = mask
            best_count = count
            best_rms = rms

    if best_mask is None:
        raise ValueError(
            "No table plane reached {} inliers. Check hand-eye TF, workspace ROI, and threshold.".format(min_inliers)
        )

    inlier_points = points[best_mask]
    origin = np.mean(inlier_points, axis=0)
    _, _, vh = np.linalg.svd(inlier_points - origin, full_matrices=False)
    normal = vh[-1]
    hint = up if up is not None else orientation_hint
    normal = orient_normal(normal, hint)
    d = -float(np.dot(normal, origin))
    distances = np.abs(points.dot(normal) + d)
    final_mask = distances <= float(distance_threshold_m)
    final_distances = distances[final_mask]
    return {
        "normal": normal,
        "d": d,
        "origin": origin,
        "inlier_mask": final_mask,
        "inlier_count": int(np.count_nonzero(final_mask)),
        "point_count": int(len(points)),
        "rms_error_m": float(np.sqrt(np.mean(final_distances ** 2))),
        "max_error_m": float(np.max(final_distances)),
    }


def make_plane_basis(normal, x_hint):
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    candidates = [
        np.asarray(x_hint, dtype=float),
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([0.0, 0.0, 1.0], dtype=float),
    ]
    x_axis = None
    for candidate in candidates:
        projected = candidate - np.dot(candidate, normal) * normal
        norm = np.linalg.norm(projected)
        if norm > 1e-6:
            x_axis = projected / norm
            break
    if x_axis is None:
        raise ValueError("Cannot construct table plane basis.")
    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return x_axis, y_axis


def normalize_axis_yaw_rad(angle):
    angle = float(angle) % math.pi
    if angle >= math.pi / 2.0:
        angle -= math.pi
    return angle


def convex_hull_2d(points):
    points = sorted(set((float(x), float(y)) for x, y in points))
    if len(points) <= 1:
        return np.asarray(points, dtype=float)

    def cross(origin, a, b):
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def min_area_rect_footprint(uv):
    hull = convex_hull_2d(uv)
    if len(hull) < 3:
        return None
    best = None
    for index in range(len(hull)):
        edge = hull[(index + 1) % len(hull)] - hull[index]
        edge_norm = np.linalg.norm(edge)
        if edge_norm < 1e-9:
            continue
        u_axis = edge / edge_norm
        v_axis = np.array([-u_axis[1], u_axis[0]], dtype=float)
        u_values = uv.dot(u_axis)
        v_values = uv.dot(v_axis)
        u_extent = float(np.max(u_values) - np.min(u_values))
        v_extent = float(np.max(v_values) - np.min(v_values))
        area = u_extent * v_extent
        yaw_rad = math.atan2(u_axis[1], u_axis[0])
        length, width = u_extent, v_extent
        if width > length:
            length, width = width, length
            yaw_rad += math.pi / 2.0
        candidate = {
            "area": area,
            "length": length,
            "width": width,
            "yaw_rad": normalize_axis_yaw_rad(yaw_rad),
        }
        if best is None or candidate["area"] < best["area"]:
            best = candidate
    return best


def robust_extent(values, percentiles=(2.0, 98.0)):
    low, high = np.percentile(values, percentiles)
    return float(max(0.0, high - low)), float(low), float(high)


def estimate_top_surface_height(
    heights,
    long_values,
    short_values,
    footprint_area,
    *,
    bin_size_m=0.003,
    min_points=40,
    min_coverage_ratio=0.18,
):
    """Estimate cuboid top height from a dense height layer, not from max points.

    A few high points from gripper fingers or mask/depth edges can dominate a
    high percentile.  This estimator searches height bins for a layer whose XY
    footprint covers enough of the detected object.  It does not clamp to an
    expected block height; it derives the top from the observed point cloud.
    """
    heights = np.asarray(heights, dtype=float)
    long_values = np.asarray(long_values, dtype=float)
    short_values = np.asarray(short_values, dtype=float)
    finite = (
        np.isfinite(heights)
        & np.isfinite(long_values)
        & np.isfinite(short_values)
    )
    heights = heights[finite]
    long_values = long_values[finite]
    short_values = short_values[finite]
    if len(heights) < max(3, int(min_points)):
        fallback = float(np.percentile(heights, 98.0)) if len(heights) else 0.0
        return {
            "height_m": fallback,
            "method": "percentile_98_insufficient_points",
            "candidate_count": 0,
            "selected": None,
        }

    bin_size = max(0.001, float(bin_size_m))
    min_height = float(np.min(heights))
    max_height = float(np.max(heights))
    if max_height <= min_height:
        return {
            "height_m": max_height,
            "method": "single_height_layer",
            "candidate_count": 1,
            "selected": {
                "count": int(len(heights)),
                "coverage_ratio": 1.0,
                "height_m": max_height,
            },
        }

    edges = np.arange(
        min_height,
        max_height + 2.0 * bin_size,
        bin_size,
        dtype=float,
    )
    if len(edges) < 2:
        edges = np.asarray([min_height, max_height + bin_size], dtype=float)

    min_count = max(8, int(0.05 * len(heights)), int(0.25 * int(min_points)))
    min_area = max(1e-8, float(footprint_area) * float(min_coverage_ratio))
    candidates = []
    for index in range(len(edges) - 1):
        low = edges[index]
        high = edges[index + 1]
        if index == len(edges) - 2:
            mask = (heights >= low) & (heights <= high)
        else:
            mask = (heights >= low) & (heights < high)
        count = int(np.count_nonzero(mask))
        if count < min_count:
            continue
        layer_long = long_values[mask]
        layer_short = short_values[mask]
        long_extent, _, _ = robust_extent(layer_long, percentiles=(5.0, 95.0))
        short_extent, _, _ = robust_extent(layer_short, percentiles=(5.0, 95.0))
        area = long_extent * short_extent
        coverage = area / max(float(footprint_area), 1e-8)
        if area < min_area:
            continue
        layer_heights = heights[mask]
        candidates.append(
            {
                "bin_low_m": float(low),
                "bin_high_m": float(high),
                "count": count,
                "coverage_ratio": float(coverage),
                "height_m": float(np.percentile(layer_heights, 90.0)),
                "median_height_m": float(np.median(layer_heights)),
            }
        )

    if candidates:
        # Highest layer with enough footprint support is the observed top
        # surface.  Sparse high layers are ignored because they do not cover the
        # object footprint.
        selected = max(candidates, key=lambda item: item["height_m"])
        return {
            "height_m": selected["height_m"],
            "method": "top_surface_height_cluster",
            "candidate_count": len(candidates),
            "selected": selected,
        }

    fallback = float(np.percentile(heights, 95.0))
    return {
        "height_m": fallback,
        "method": "percentile_95_no_surface_cluster",
        "candidate_count": 0,
        "selected": None,
    }


def expanded_bbox_mask(bbox, width, height, expand_px):
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return np.zeros((height, width), dtype=bool)
    x1, y1, x2, y2 = bbox
    expand = int(expand_px)
    x1 = max(0, min(width - 1, int(round(x1)) - expand))
    y1 = max(0, min(height - 1, int(round(y1)) - expand))
    x2 = max(0, min(width - 1, int(round(x2)) + expand))
    y2 = max(0, min(height - 1, int(round(y2)) + expand))
    mask = np.zeros((height, width), dtype=bool)
    if x2 > x1 and y2 > y1:
        mask[y1:y2 + 1, x1:x2 + 1] = True
    return mask


def dominant_z_layer_below(points_base, top_z_base, bin_size_m=0.003, min_points=80):
    points = np.asarray(points_base, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3 or not len(points):
        return None
    z = points[:, 2]
    z = z[np.isfinite(z)]
    if not len(z):
        return None
    top_z = float(top_z_base)
    # Local support must be below the observed top surface. This is geometric
    # separation, not a clamp to a known object height.
    z = z[z < top_z - 0.004]
    if len(z) < int(min_points):
        z = points[:, 2]
        z = z[np.isfinite(z) & (z < top_z)]
    if len(z) < int(min_points):
        return None
    bin_size = max(0.001, float(bin_size_m))
    edges = np.arange(float(np.min(z)), float(np.max(z)) + 2.0 * bin_size, bin_size)
    if len(edges) < 2:
        return None
    candidates = []
    for index in range(len(edges) - 1):
        low = edges[index]
        high = edges[index + 1]
        if index == len(edges) - 2:
            mask = (z >= low) & (z <= high)
        else:
            mask = (z >= low) & (z < high)
        count = int(np.count_nonzero(mask))
        if count < int(min_points):
            continue
        values = z[mask]
        candidates.append(
            {
                "bin_low_m": float(low),
                "bin_high_m": float(high),
                "count": count,
                "support_z_base_m": float(np.median(values)),
            }
        )
    if not candidates:
        return None
    # For a book/table support around a block, choose the highest dense layer
    # below the block top. Lower layers are usually the desk or background.
    return max(candidates, key=lambda item: item["support_z_base_m"])


def estimate_local_support_surface(
    depth_frame,
    intrinsics,
    det,
    top_z_base,
    *,
    transform_matrix=None,
    point_mode="optical-to-camera-link",
    max_depth_m=2.0,
    stride=2,
    expand_px=24,
    exclude_dilate_px=6,
    bin_size_m=0.003,
    min_points=80,
):
    width = int(depth_frame.get_width())
    height = int(depth_frame.get_height())
    support_mask = expanded_bbox_mask(det.get("bbox"), width, height, expand_px)
    raw_mask = object_mask(det)
    if raw_mask is not None:
        resized = resize_mask_to_image(raw_mask, width, height)
        if resized is not None:
            support_mask &= ~dilate_mask(resized, pixels=exclude_dilate_px)
    else:
        support_mask &= ~expanded_bbox_mask(det.get("bbox"), width, height, 0)
    if not np.any(support_mask):
        return None
    points_optical = deproject_depth_mask(
        depth_frame,
        intrinsics,
        support_mask,
        stride=stride,
        max_depth_m=max_depth_m,
    )
    points_base = transform_points(
        points_optical,
        matrix=transform_matrix,
        point_mode=point_mode,
    )
    selected = dominant_z_layer_below(
        points_base,
        top_z_base,
        bin_size_m=bin_size_m,
        min_points=min_points,
    )
    if selected is None:
        return None
    selected["raw_point_count"] = int(len(points_optical))
    return selected


def estimate_object_footprint(
    points,
    plane,
    min_height_m=0.006,
    max_height_m=0.50,
    min_points=40,
    yaw_min_aspect_ratio=1.20,
    use_min_area_rect=False,
    height_bin_m=0.003,
    top_coverage_ratio=0.18,
):
    points = np.asarray(points, dtype=float)
    normal = plane["normal"]
    origin = plane["origin"]
    x_axis = plane["x_axis"]
    y_axis = plane["y_axis"]
    heights = (points - origin).dot(normal)
    mask = (heights >= float(min_height_m)) & (heights <= float(max_height_m))
    object_points = points[mask]
    object_heights = heights[mask]
    if len(object_points) < int(min_points):
        return None

    relative = object_points - origin
    uv = np.column_stack([relative.dot(x_axis), relative.dot(y_axis)])
    uv_center = np.median(uv, axis=0)
    centered = uv - uv_center
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    long_axis_uv = eigenvectors[:, order[0]]
    if long_axis_uv[0] < 0:
        long_axis_uv = -long_axis_uv
    short_axis_uv = np.array([-long_axis_uv[1], long_axis_uv[0]], dtype=float)
    long_values = centered.dot(long_axis_uv)
    short_values = centered.dot(short_axis_uv)
    length, long_low, long_high = robust_extent(long_values)
    width, short_low, short_high = robust_extent(short_values)
    if width > length:
        length, width = width, length
        long_axis_uv = short_axis_uv
    aspect_ratio = length / max(width, 1e-9)
    yaw_rad = normalize_axis_yaw_rad(math.atan2(long_axis_uv[1], long_axis_uv[0]))
    yaw_source = "pca_long_axis"
    min_rect = min_area_rect_footprint(uv)
    if use_min_area_rect and min_rect is not None:
        length = min_rect["length"]
        width = min_rect["width"]
        aspect_ratio = length / max(width, 1e-9)
        yaw_rad = min_rect["yaw_rad"]
        yaw_source = "min_area_rect"
    footprint_area = max(length * width, 1e-8)
    height_estimate = estimate_top_surface_height(
        object_heights,
        long_values,
        short_values,
        footprint_area,
        bin_size_m=height_bin_m,
        min_points=min_points,
        min_coverage_ratio=top_coverage_ratio,
    )
    height = float(height_estimate["height_m"])
    selected_surface = height_estimate.get("selected")
    top_surface_points = None
    if selected_surface is not None:
        low = float(selected_surface["bin_low_m"])
        high = float(selected_surface["bin_high_m"])
        surface_mask = (object_heights >= low) & (object_heights <= high)
        if np.count_nonzero(surface_mask):
            top_surface_points = object_points[surface_mask]
    center_plane = origin + uv_center[0] * x_axis + uv_center[1] * y_axis
    center_object = center_plane + (height / 2.0) * normal
    if top_surface_points is not None and len(top_surface_points):
        top_surface_center = np.median(top_surface_points, axis=0)
        # Placement is commanded in base_link Z, so record the observed absolute
        # vertical top coordinate directly from the top-surface point cluster.
        top_z_base = float(np.median(top_surface_points[:, 2]))
        center_object = top_surface_center - (height / 2.0) * normal
    else:
        top_surface_center = center_plane + height * normal
        top_z_base = float(top_surface_center[2])
    return {
        "point_count": int(len(object_points)),
        "dimensions_m": [length, width, height],
        "aspect_ratio": float(aspect_ratio),
        "yaw_rad": yaw_rad,
        "yaw_deg": math.degrees(yaw_rad),
        "yaw_valid": bool(aspect_ratio >= float(yaw_min_aspect_ratio)),
        "yaw_source": yaw_source,
        "center_on_plane_m": center_plane,
        "center_object_m": center_object,
        "top_surface_center_m": top_surface_center,
        "top_z_base_m": top_z_base,
        "top_z_source": (
            "observed_top_surface_points"
            if top_surface_points is not None and len(top_surface_points)
            else "plane_plus_estimated_height"
        ),
        "height_percentiles_m": np.percentile(object_heights, [2.0, 50.0, 98.0]).tolist(),
        "height_estimation_method": height_estimate["method"],
        "height_surface_cluster": height_estimate["selected"],
        "height_surface_candidate_count": height_estimate["candidate_count"],
        "eigenvalues": eigenvalues.tolist(),
    }


def workspace_roi(detections):
    candidates = [
        det for det in detections if det.get("label") == "workspace" or det.get("is_workspace")
    ]
    if not candidates:
        return None
    workspace = max(candidates, key=lambda item: item.get("confidence", 0.0))
    return workspace.get("bbox")


def public_plane_payload(plane, frame):
    return {
        "frame": frame,
        "normal": plane["normal"].tolist(),
        "d": float(plane["d"]),
        "origin_m": plane["origin"].tolist(),
        "x_axis": plane["x_axis"].tolist(),
        "y_axis": plane["y_axis"].tolist(),
        "inlier_count": plane["inlier_count"],
        "point_count": plane["point_count"],
        "inlier_ratio": float(plane["inlier_count"]) / max(1, plane["point_count"]),
        "rms_error_m": plane["rms_error_m"],
        "max_error_m": plane["max_error_m"],
    }


def attach_tabletop_geometry(detections, depth_frame, intrinsics, args, transform_matrix=None):
    for det in detections:
        det["pointcloud_geometry_valid"] = False
        det["pointcloud_source"] = None
        det["geometry_estimation_method"] = None
        det["depth_geometry_observable"] = False
        det["dimensions_m"] = None
        det["table_yaw_deg"] = None
        det["table_yaw_valid"] = False

    if depth_frame is None or intrinsics is None:
        raise ValueError("Tabletop estimation requires aligned depth and color intrinsics.")

    plane_points_optical = deproject_depth_roi(
        depth_frame,
        intrinsics,
        roi=workspace_roi(detections),
        stride=args.plane_point_stride,
        max_depth_m=args.plane_max_depth_m,
    )
    plane_points = transform_points(
        plane_points_optical,
        matrix=transform_matrix,
        point_mode=getattr(args, "tf_point_mode", "optical-to-camera-link"),
    )
    if not len(plane_points):
        raise ValueError("No valid depth points were available for table plane estimation.")
    in_base = transform_matrix is not None
    frame = args.base_frame if in_base else "camera_optical_frame"
    up_hint = np.array([0.0, 0.0, 1.0], dtype=float) if in_base else None
    orientation_hint = up_hint if in_base else -np.mean(plane_points, axis=0)
    plane = fit_plane_ransac(
        plane_points,
        distance_threshold_m=args.plane_distance_threshold_m,
        iterations=args.plane_ransac_iterations,
        min_inliers=args.plane_min_inliers,
        up_hint=up_hint,
        min_up_alignment=args.plane_min_up_alignment if in_base else 0.0,
        orientation_hint=orientation_hint,
    )
    x_hint = np.array([1.0, 0.0, 0.0], dtype=float)
    plane["x_axis"], plane["y_axis"] = make_plane_basis(plane["normal"], x_hint)

    for det in detections:
        if det.get("label") == "workspace" or det.get("is_workspace"):
            continue
        pointcloud = build_object_pointcloud(
            depth_frame,
            intrinsics,
            det,
            stride=args.object_point_stride,
            max_depth_m=args.plane_max_depth_m,
            mask_erode_px=getattr(args, "object_mask_erode_px", 2),
            transform_matrix=transform_matrix,
            point_mode=getattr(args, "tf_point_mode", "optical-to-camera-link"),
            plane=plane,
            min_height_m=args.object_min_height_m,
            max_height_m=args.object_max_height_m,
            outlier_percentile=getattr(args, "object_outlier_percentile", 2.0),
            min_object_points=args.object_min_points,
            mask_dilate_fallback_px=getattr(args, "object_mask_dilate_fallback_px", 4),
            known_object_height_m=getattr(args, "known_object_height_m", 0.0),
        )
        object_points = pointcloud["object_points_base"]
        det["pointcloud_source"] = pointcloud["source"]
        det["pointcloud_raw_point_count"] = pointcloud["raw_point_count"]
        det["pointcloud_table_filtered_count"] = pointcloud["table_filtered_point_count"]
        det["pointcloud_outlier_filtered_count"] = pointcloud["outlier_filtered_point_count"]
        det["_object_points_base"] = pointcloud["object_points_base"]
        geometry = estimate_object_footprint(
            object_points,
            plane,
            min_height_m=args.object_min_height_m,
            max_height_m=args.object_max_height_m,
            min_points=args.object_min_points,
            yaw_min_aspect_ratio=args.yaw_min_aspect_ratio,
            use_min_area_rect=any(
                value in str(det.get("label") or "").lower()
                for value in ("square", "rectangle")
            ),
            height_bin_m=getattr(args, "object_height_bin_m", 0.003),
            top_coverage_ratio=getattr(args, "object_top_coverage_ratio", 0.18),
        )
        if geometry is None:
            continue
        det["pointcloud_geometry_valid"] = True
        det["geometry_estimation_method"] = (
            "known_height_prior"
            if pointcloud["source"] == "mask_known_height_fallback"
            else "depth_pointcloud"
        )
        det["depth_geometry_observable"] = pointcloud["source"] != "mask_known_height_fallback"
        det["pointcloud_point_count"] = geometry["point_count"]
        if in_base and geometry.get("top_z_base_m") is not None:
            support = estimate_local_support_surface(
                depth_frame,
                intrinsics,
                det,
                geometry["top_z_base_m"],
                transform_matrix=transform_matrix,
                point_mode=getattr(args, "tf_point_mode", "optical-to-camera-link"),
                max_depth_m=args.plane_max_depth_m,
                stride=args.object_point_stride,
                expand_px=getattr(args, "object_support_ring_expand_px", 24),
                exclude_dilate_px=getattr(args, "object_support_exclude_dilate_px", 6),
                bin_size_m=getattr(args, "object_support_bin_m", 0.003),
                min_points=getattr(args, "object_support_min_points", 80),
            )
            if support is not None:
                local_height = float(geometry["top_z_base_m"]) - float(support["support_z_base_m"])
                if math.isfinite(local_height) and local_height > 0.0:
                    geometry["dimensions_m"][2] = local_height
                    center = np.asarray(geometry["center_object_m"], dtype=float).copy()
                    center[2] = float(support["support_z_base_m"]) + local_height / 2.0
                    geometry["center_object_m"] = center
                    support_center = np.asarray(geometry["center_on_plane_m"], dtype=float).copy()
                    support_center[2] = float(support["support_z_base_m"])
                    geometry["center_on_plane_m"] = support_center
                    geometry["local_support_surface"] = support
                    geometry["height_estimation_method"] = "{}+local_support_ring".format(
                        geometry["height_estimation_method"]
                    )
        det["dimensions_m"] = geometry["dimensions_m"]
        det["footprint_aspect_ratio"] = geometry["aspect_ratio"]
        det["table_yaw_deg"] = geometry["yaw_deg"]
        det["table_yaw_rad"] = geometry["yaw_rad"]
        det["table_yaw_valid"] = geometry["yaw_valid"]
        det["table_yaw_source"] = geometry["yaw_source"]
        det["geometry_frame"] = frame
        det["geometry_center_m"] = geometry["center_object_m"].tolist()
        det["center_on_table_m"] = geometry["center_on_plane_m"].tolist()
        det["top_surface_center_m"] = geometry["top_surface_center_m"].tolist()
        det["top_z_base_m"] = geometry["top_z_base_m"] if in_base else None
        det["top_z_source"] = geometry["top_z_source"] if in_base else None
        det["local_support_surface"] = geometry.get("local_support_surface")
        det["height_percentiles_m"] = geometry["height_percentiles_m"]
        det["height_estimation_method"] = geometry["height_estimation_method"]
        det["height_surface_cluster"] = geometry["height_surface_cluster"]
        det["height_surface_candidate_count"] = geometry["height_surface_candidate_count"]

    return detections, public_plane_payload(plane, frame)
