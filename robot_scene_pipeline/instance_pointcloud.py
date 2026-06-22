"""Build per-instance point clouds from detector masks and aligned depth."""

import numpy as np


def resize_mask_to_image(mask, width, height, threshold=0.5):
    if mask is None:
        return None
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.size == 0:
        return None
    if mask.shape != (height, width):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Object mask resize requires cv2.") from exc
        mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
    return mask > float(threshold)


def erode_mask(mask_bool, pixels=2):
    mask_bool = np.asarray(mask_bool, dtype=bool)
    pixels = int(pixels)
    if pixels <= 0 or mask_bool.size == 0:
        return mask_bool
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Object mask erosion requires cv2.") from exc
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask_bool.astype(np.uint8), kernel, iterations=pixels)
    return eroded.astype(bool)


def dilate_mask(mask_bool, pixels=2):
    mask_bool = np.asarray(mask_bool, dtype=bool)
    pixels = int(pixels)
    if pixels <= 0 or mask_bool.size == 0:
        return mask_bool
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Object mask dilation requires cv2.") from exc
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=pixels)
    return dilated.astype(bool)


def mask_bounds(mask_bool):
    ys, xs = np.nonzero(mask_bool)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def deproject_depth_mask(depth_frame, intrinsics, mask_bool, stride=1, max_depth_m=2.0):
    width = int(depth_frame.get_width())
    height = int(depth_frame.get_height())
    mask_bool = resize_mask_to_image(mask_bool, width, height)
    if mask_bool is None:
        return np.empty((0, 3), dtype=float)
    bounds = mask_bounds(mask_bool)
    if bounds is None:
        return np.empty((0, 3), dtype=float)

    fx = float(intrinsics.fx)
    fy = float(intrinsics.fy)
    ppx = float(intrinsics.ppx)
    ppy = float(intrinsics.ppy)
    try:
        import pyrealsense2 as rs
    except ImportError:
        rs = None

    x1, y1, x2, y2 = bounds
    step = max(1, int(stride))
    points = []
    for y in range(y1, y2 + 1, step):
        row = mask_bool[y]
        for x in range(x1, x2 + 1, step):
            if not row[x]:
                continue
            depth = float(depth_frame.get_distance(x, y))
            if depth <= 0 or depth > float(max_depth_m):
                continue
            if rs is not None and hasattr(intrinsics, "model"):
                points.append(rs.rs2_deproject_pixel_to_point(intrinsics, [float(x), float(y)], depth))
            else:
                points.append([(x - ppx) * depth / fx, (y - ppy) * depth / fy, depth])
    if not points:
        return np.empty((0, 3), dtype=float)
    return np.asarray(points, dtype=float)


def deproject_depth_bbox(depth_frame, intrinsics, bbox, stride=1, max_depth_m=2.0):
    width = int(depth_frame.get_width())
    height = int(depth_frame.get_height())
    x1, y1, x2, y2 = bbox
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

    step = max(1, int(stride))
    points = []
    for y in range(y1, y2 + 1, step):
        for x in range(x1, x2 + 1, step):
            depth = float(depth_frame.get_distance(x, y))
            if depth <= 0 or depth > float(max_depth_m):
                continue
            if rs is not None and hasattr(intrinsics, "model"):
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


def object_mask(det):
    for key in ("_mask_bool", "_mask", "mask", "segmentation_mask"):
        mask = det.get(key)
        if mask is not None:
            return mask
    return None


def project_mask_to_plane_height(
    intrinsics,
    mask_bool,
    plane,
    height_m,
    *,
    stride=1,
    transform_matrix=None,
    point_mode="optical-to-camera-link",
):
    mask_bool = np.asarray(mask_bool, dtype=bool)
    bounds = mask_bounds(mask_bool)
    if bounds is None:
        return np.empty((0, 3), dtype=float)

    fx = float(intrinsics.fx)
    fy = float(intrinsics.fy)
    ppx = float(intrinsics.ppx)
    ppy = float(intrinsics.ppy)
    if transform_matrix is None:
        rotation = np.eye(3, dtype=float)
        camera_origin = np.zeros(3, dtype=float)
    else:
        matrix = np.asarray(transform_matrix, dtype=float)
        rotation = matrix[:3, :3]
        camera_origin = matrix[:3, 3]

    normal = np.asarray(plane["normal"], dtype=float)
    plane_origin = np.asarray(plane["origin"], dtype=float)
    numerator = float(height_m) - float(np.dot(camera_origin - plane_origin, normal))
    x1, y1, x2, y2 = bounds
    step = max(1, int(stride))
    points = []
    for y in range(y1, y2 + 1, step):
        row = mask_bool[y]
        for x in range(x1, x2 + 1, step):
            if not row[x]:
                continue
            ray_optical = np.array([(x - ppx) / fx, (y - ppy) / fy, 1.0], dtype=float)
            if point_mode == "optical-to-camera-link":
                ray_source = np.array(
                    [ray_optical[2], -ray_optical[0], -ray_optical[1]],
                    dtype=float,
                )
            elif point_mode == "direct":
                ray_source = ray_optical
            else:
                raise ValueError("Unsupported point mode: {}".format(point_mode))
            ray = rotation.dot(ray_source)
            denominator = float(np.dot(ray, normal))
            if abs(denominator) < 1e-9:
                continue
            distance = numerator / denominator
            if distance <= 0:
                continue
            points.append(camera_origin + distance * ray)
    if not points:
        return np.empty((0, 3), dtype=float)
    return np.asarray(points, dtype=float)


def remove_table_points(points, plane, min_height_m=0.006, max_height_m=0.50):
    points = np.asarray(points, dtype=float)
    if plane is None or not len(points):
        return points.reshape((-1, 3)), np.empty((0,), dtype=float)
    normal = np.asarray(plane["normal"], dtype=float)
    origin = np.asarray(plane["origin"], dtype=float)
    heights = (points - origin).dot(normal)
    mask = (heights >= float(min_height_m)) & (heights <= float(max_height_m))
    return points[mask], heights[mask]


def remove_outliers(points, percentile=2.0):
    points = np.asarray(points, dtype=float)
    if not len(points):
        return points.reshape((-1, 3))
    percentile = float(percentile)
    if percentile <= 0:
        return points
    percentile = min(20.0, percentile)
    low = np.percentile(points, percentile, axis=0)
    high = np.percentile(points, 100.0 - percentile, axis=0)
    mask = np.all((points >= low) & (points <= high), axis=1)
    filtered = points[mask]
    return filtered if len(filtered) else points


def build_object_pointcloud(
    depth_frame,
    intrinsics,
    det,
    *,
    stride=1,
    max_depth_m=2.0,
    mask_erode_px=2,
    transform_matrix=None,
    point_mode="optical-to-camera-link",
    plane=None,
    min_height_m=0.006,
    max_height_m=0.50,
    outlier_percentile=2.0,
    min_object_points=40,
    mask_dilate_fallback_px=4,
    known_object_height_m=0.0,
):
    width = int(depth_frame.get_width())
    height = int(depth_frame.get_height())
    raw_mask = object_mask(det)
    if raw_mask is not None:
        mask = resize_mask_to_image(raw_mask, width, height)
        if mask is not None:
            candidates = [
                ("mask_eroded", erode_mask(mask, pixels=mask_erode_px)),
                ("mask_raw", mask),
            ]
            if int(mask_dilate_fallback_px) > 0:
                candidates.append(
                    ("mask_dilated", dilate_mask(mask, pixels=mask_dilate_fallback_px))
                )
        else:
            candidates = [("bbox_invalid_mask", None)]
    else:
        candidates = [("bbox_no_mask", None)]

    best = None
    for source, candidate_mask in candidates:
        if candidate_mask is None:
            points_optical = deproject_depth_bbox(
                depth_frame,
                intrinsics,
                det.get("bbox"),
                stride=stride,
                max_depth_m=max_depth_m,
            )
        else:
            points_optical = deproject_depth_mask(
                depth_frame,
                intrinsics,
                candidate_mask,
                stride=stride,
                max_depth_m=max_depth_m,
            )
        points_base = transform_points(points_optical, matrix=transform_matrix, point_mode=point_mode)
        points_base, heights = remove_table_points(
            points_base,
            plane,
            min_height_m=min_height_m,
            max_height_m=max_height_m,
        )
        filtered_points_base = remove_outliers(points_base, percentile=outlier_percentile)
        result = {
            "source": source,
            "points_optical": points_optical,
            "object_points_base": filtered_points_base,
            "height_samples_m": heights,
            "raw_point_count": int(len(points_optical)),
            "table_filtered_point_count": int(len(points_base)),
            "outlier_filtered_point_count": int(len(filtered_points_base)),
        }
        if best is None or result["outlier_filtered_point_count"] > best["outlier_filtered_point_count"]:
            best = result
        if result["outlier_filtered_point_count"] >= int(min_object_points):
            return result

    known_height = float(known_object_height_m or 0.0)
    if (
        raw_mask is not None
        and plane is not None
        and known_height > 0
        and (best is None or best["outlier_filtered_point_count"] < int(min_object_points))
    ):
        mask = resize_mask_to_image(raw_mask, width, height)
        if mask is not None:
            projected_mask = erode_mask(mask, pixels=mask_erode_px)
            if not np.any(projected_mask):
                projected_mask = mask
            projected_points = project_mask_to_plane_height(
                intrinsics,
                projected_mask,
                plane,
                known_height,
                stride=stride,
                transform_matrix=transform_matrix,
                point_mode=point_mode,
            )
            projected_points = remove_outliers(projected_points, percentile=outlier_percentile)
            if len(projected_points):
                return {
                    "source": "mask_known_height_fallback",
                    "points_optical": np.empty((0, 3), dtype=float),
                    "object_points_base": projected_points,
                    "height_samples_m": np.full(len(projected_points), known_height, dtype=float),
                    "raw_point_count": int(best["raw_point_count"]) if best is not None else 0,
                    "table_filtered_point_count": int(best["table_filtered_point_count"]) if best is not None else 0,
                    "outlier_filtered_point_count": int(len(projected_points)),
                }
    return best
