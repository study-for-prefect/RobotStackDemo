"""Scene-object selection and visibility helpers."""

from .io import load_json

def object_id_value(obj):
    try:
        return int(obj.get("id", -1))
    except (TypeError, ValueError):
        return -1


def object_base_point(obj):
    geometry_point = obj.get("geometry_center_m") if obj.get("geometry_frame") == "base_link" else None
    point = geometry_point or obj.get("center_3d_base_m")
    if not point or len(point) < 2:
        return None
    try:
        return [float(point[0]), float(point[1]), float(point[2]) if len(point) > 2 else 0.0]
    except (TypeError, ValueError):
        return None


def object_sort_key(obj):
    point = object_base_point(obj)
    if point is None:
        return (float("inf"), float("inf"), object_id_value(obj))
    return (point[0], point[1], object_id_value(obj))


def describe_object(obj):
    point = object_base_point(obj)
    xy = "unknown"
    if point is not None:
        xy = "[{:.4f}, {:.4f}]".format(point[0], point[1])
    return "id={} label={} base_xy={}".format(obj.get("id"), obj.get("label"), xy)


def detected_object_summary(private_state_path):
    try:
        state = load_json(private_state_path)
    except (IOError, ValueError) as exc:
        return "failed to read {}: {}".format(private_state_path, exc)
    objects = [obj for obj in state.get("objects", []) if not obj.get("is_workspace")]
    if not objects:
        return "no non-workspace objects detected"
    objects = sorted(objects, key=object_sort_key)
    return "; ".join(describe_object(obj) for obj in objects)


def target_description(args):
    if args.object_id is not None:
        return "object_id={}".format(args.object_id)
    return "object_label='{}'".format(args.object_label)


def matching_target_objects(args, private_state_path):
    state = load_json(private_state_path)
    objects = [obj for obj in state.get("objects", []) if not obj.get("is_workspace")]
    if args.object_id is not None:
        matches = [obj for obj in objects if object_id_value(obj) == int(args.object_id)]
    else:
        requested = str(args.object_label).strip().lower()
        matches = [obj for obj in objects if requested in str(obj.get("label", "")).lower()]
    return sorted(matches, key=object_sort_key)


def target_visible(args, private_state_path):
    try:
        return bool(matching_target_objects(args, private_state_path))
    except (IOError, ValueError):
        return False


def print_missing_target(stage, args, private_state_path):
    print(
        "\n{}: did not detect requested target {} in this snapshot; it may be outside the camera frame or missed by detection.".format(
            stage,
            target_description(args),
        ),
        flush=True,
    )
    print("Detected objects: {}".format(detected_object_summary(private_state_path)), flush=True)


def camera_vector_to_base(tf_json_path, camera_vector):
    """Rotate a vector from the TF JSON child frame into base_link."""
    payload = load_json(tf_json_path)
    matrix = payload.get("matrix_4x4")
    if not isinstance(matrix, list) or len(matrix) < 3:
        raise RuntimeError("TF JSON missing matrix_4x4: {}".format(tf_json_path))
    return [
        sum(float(matrix[row][col]) * float(camera_vector[col]) for col in range(3))
        for row in range(3)
    ]


def camera_optical_vector_to_base(tf_json_path, optical_vector):
    return camera_vector_to_base(tf_json_path, [float(value) for value in optical_vector])
