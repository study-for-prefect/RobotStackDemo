"""Full-SE(3) target solving and fixed-GF225-TCP airborne trajectories."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from robot_scene_pipeline.grasp_orientation import downward_quaternion_for_yaw

from ..common.config import StackDemoConfig
from ..common.pose3d import (
    axis_alignment_error_deg,
    matrix_to_quaternion,
    pose_to_transform,
    quaternion_inverse,
    quaternion_from_axis_angle,
    quaternion_multiply,
    quaternion_to_matrix,
    rotation_error_deg,
    slerp,
    transform_inverse,
    transform_multiply,
    transform_to_pose,
    transform_vector,
)
from ..common.scene_state import ClutterSceneState, SceneObjectState


FULL_3D_ORIENTATION_SHAPES = frozenset({"rectangle", "concave_rectangle", "triangle"})


def beam_axis_from_supports(
    scene: ClutterSceneState,
    role_bindings: Mapping[str, str],
    config: StackDemoConfig,
    *,
    verified_roof_track_id: str | None = None,
) -> tuple[float, float, float] | None:
    """Return a current-frame structural axis, never a nominal or remembered yaw."""
    left = _scene_object(scene, role_bindings.get("left_support_upper", ""))
    right = _scene_object(scene, role_bindings.get("right_support_upper", ""))
    if left is not None and right is not None:
        if abs(_top(left) - _top(right)) > float(config.section("house")["support_height_tolerance_m"]):
            return None
        dx = right.center_xyz_m[0] - left.center_xyz_m[0]
        dy = right.center_xyz_m[1] - left.center_xyz_m[1]
        length = math.hypot(dx, dy)
        return None if length <= 1e-6 else (dx / length, dy / length, 0.0)

    # On a completed house both upper supports are legitimately occluded by
    # the roof.  This fallback is authorized only by the caller's current-frame
    # 60-67 mm roof completion result and still uses the measured 3-D roof
    # long axis, not configuration yaw or action history.
    if not verified_roof_track_id:
        return None
    roof = _scene_object(scene, verified_roof_track_id)
    long_axis = roof.source.get("long_axis_base") if roof is not None else None
    if not isinstance(long_axis, (list, tuple)) or len(long_axis) != 3:
        return None
    if abs(float(long_axis[2])) > 0.20:
        return None
    dx, dy = float(long_axis[0]), float(long_axis[1])
    length = math.hypot(dx, dy)
    return None if length <= 1e-6 else (dx / length, dy / length, 0.0)


def observed_object_pose(obj: SceneObjectState) -> dict[str, Any] | None:
    orientation = obj.source.get("orientation_xyzw") or obj.source.get("object_orientation_xyzw")
    if not isinstance(orientation, (list, tuple)) or len(orientation) != 4:
        return None
    return {
        "frame_id": "base_link",
        "position_m": list(obj.center_xyz_m),
        "orientation_xyzw": [float(value) for value in orientation],
    }


def special_shape_evidence_ready(obj: SceneObjectState, config: StackDemoConfig) -> bool:
    """Gate final placement on same-revision fused image and depth evidence."""
    source = obj.source
    minimum_shape = float(config.section("special_shape_vlm")["minimum_shape_confidence"])
    minimum_orientation = float(config.section("special_shape_vlm")["minimum_orientation_confidence"])
    semantic = str(source.get("semantic_shape") or obj.shape)
    shape_confident = bool(
        float(source.get("semantic_shape_confidence", 1.0)) >= minimum_shape
        or semantic == "triangle"
        and source.get("metric_triangle_geometry_confirmed") is True
    )
    shape_axis_ready = bool(
        _vector_available(source.get("broad_face_normal_base")) if semantic == "rectangle"
        else _vector_available(source.get("groove_opening_normal_base")) if semantic == "concave_rectangle"
        else _vector_available(source.get("designated_right_angle_edge_base")) if semantic == "triangle"
        else False
    )
    orientation_confident = bool(
        float(source.get("orientation_confidence", obj.orientation_confidence)) >= minimum_orientation
        or (
            semantic == "triangle"
            and source.get("tabletop_apex_up_geometry_verified") is True
            and float(source.get("orientation_confidence", obj.orientation_confidence))
            >= float(config.section("house_orientation")[
                "triangle_tabletop_minimum_orientation_confidence"
            ])
        )
    )
    return bool(
        semantic in FULL_3D_ORIENTATION_SHAPES
        and not source.get("semantic_shape_uncertain", False)
        and shape_confident
        and orientation_confident
        and observed_object_pose(obj) is not None
        and _vector_available(source.get("long_axis_base"))
        and shape_axis_ready
        and int(source.get("evidence_scene_revision", 0)) in {0, int(obj.object_ref.split(":")[0].split("_")[-1])}
    )


def triangle_tabletop_evidence_ready(
    obj: SceneObjectState, config: StackDemoConfig,
) -> bool:
    """Allow a reversible 45-degree table step without weakening roof transport."""
    source = obj.source
    semantic = str(source.get("semantic_shape") or obj.shape)
    return bool(
        semantic == "triangle"
        and source.get("semantic_shape_uncertain", False) is False
        and source.get("metric_triangle_geometry_confirmed") is True
        and float(source.get("orientation_confidence", obj.orientation_confidence))
        >= float(config.section("house_orientation")[
            "triangle_tabletop_minimum_orientation_confidence"
        ])
        and observed_object_pose(obj) is not None
        and _vector_available(source.get("long_axis_base"))
        and _vector_available(source.get("designated_right_angle_edge_base"))
        and int(source.get("evidence_scene_revision", 0))
        in {0, int(obj.object_ref.split(":")[0].split("_")[-1])}
    )


def semantic_face_ready_for_structural_tilt(
    obj: SceneObjectState,
    config: StackDemoConfig,
) -> tuple[bool, dict[str, Any]]:
    """Separate tabletop face selection from the final structural roof tilt."""
    source = obj.source
    orientation = config.section("house_orientation")
    semantic = str(source.get("semantic_shape") or obj.shape)
    if semantic == "rectangle":
        field = "broad_face_normal_base"
        vector = source.get(field)
        component = float(vector[2]) if _vector_available(vector) else float("-inf")
        threshold = float(orientation["minimum_upward_normal_component"])
        passed = component >= threshold
        requirement = "broad_face_up_before_pick"
    elif semantic == "concave_rectangle":
        field = "groove_opening_normal_base"
        vector = source.get(field)
        component = float(vector[2]) if _vector_available(vector) else float("inf")
        threshold = -float(orientation["minimum_downward_normal_component"])
        passed = component <= threshold
        requirement = "groove_opening_down_before_pick"
    elif semantic == "triangle":
        field = "designated_right_angle_edge_base"
        vector = source.get(field)
        component = float(vector[2]) if _vector_available(vector) else float("-inf")
        threshold = float(orientation.get(
            "triangle_minimum_upward_component",
            orientation["minimum_upward_normal_component"],
        ))
        passed = component >= threshold
        requirement = "designated_right_angle_edge_up_before_pick"
    else:
        field, component, threshold, passed = "unsupported", 0.0, 0.0, False
        requirement = "unsupported_special_shape"
    return bool(passed), {
        "semantic_face_check": requirement,
        "semantic_face_vector_field": field,
        "semantic_face_vertical_component": component,
        "semantic_face_vertical_threshold": threshold,
        "semantic_face_ready_before_pick": bool(passed),
        "tabletop_face_reorientation_required": not bool(passed),
    }
def solve_target_object_orientations(
    obj: SceneObjectState,
    target_position_m: Sequence[float],
    beam_axis_world: Sequence[float],
    config: StackDemoConfig,
    *,
    allow_staged_triangle_face_change: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Solve a rigid object pose from measured object-local semantic axes.

    The depth/VLM semantic vectors are expressed in ``base_link`` while the
    target quaternion maps the object's local frame into ``base_link``.  A
    target quaternion therefore cannot be built by treating those measured
    vectors as canonical object X/Z axes.  Recover the axes in the current
    object frame first, then align that complete local basis with the desired
    roof basis.
    """
    orientation = config.section("house_orientation")
    beam = _unit_horizontal(beam_axis_world)
    current_pose = observed_object_pose(obj)
    if current_pose is None:
        return ()
    current_q = current_pose["orientation_xyzw"]
    current_q_inverse = quaternion_inverse(current_q)
    current_long = _normalize(obj.source["long_axis_base"])
    if sum(current_long[index] * beam[index] for index in range(3)) < 0.0:
        beam = tuple(-value for value in beam)
    if obj.shape == "rectangle":
        semantic_name = "broad_face_normal_world"
        semantic_vector = _normalize(obj.source["broad_face_normal_base"])
        broad_normal = semantic_vector
        component = semantic_vector[2]
        semantic_ok = component >= float(orientation["minimum_upward_normal_component"])
    elif obj.shape == "concave_rectangle":
        semantic_name = "groove_opening_normal_world"
        semantic_vector = _normalize(obj.source["groove_opening_normal_base"])
        broad_normal = tuple(-value for value in semantic_vector)
        component = semantic_vector[2]
        semantic_ok = component <= -float(orientation["minimum_downward_normal_component"])
    else:
        semantic_name = "designated_right_angle_edge_world"
        semantic_vector = _normalize(obj.source["designated_right_angle_edge_base"])
        broad_normal = semantic_vector
        component = semantic_vector[2]
        semantic_ok = component >= float(orientation.get(
            "triangle_minimum_upward_component",
            orientation["minimum_upward_normal_component"],
        ))
    if not semantic_ok:
        return ()

    long_local = _normalize(transform_vector(current_q_inverse, current_long))
    semantic_local = _normalize(transform_vector(current_q_inverse, broad_normal))
    source_basis = _orthonormal_basis(long_local, semantic_local)

    # A triangle resting on its broad triangular face must become upright on
    # the roof: its centroid-to-right-angle-vertex direction is world +Z and
    # its hypotenuse/long axis follows the measured roof beam.  For rectangle
    # roofs preserve the already-correct broad/groove face and only remove the
    # small sensor component parallel to the beam.
    desired_semantic = (
        (0.0, 0.0, 1.0)
        if obj.shape == "triangle"
        else broad_normal
    )
    desired_basis = _orthonormal_basis(beam, desired_semantic)
    rotation = _basis_mapping_rotation(source_basis, desired_basis)
    target_q = matrix_to_quaternion(rotation)
    target_long = transform_vector(target_q, source_basis[0])
    long_axis_error = axis_alignment_error_deg(
        target_long, beam,
    )
    target_basis_semantic = transform_vector(target_q, source_basis[2])
    semantic_error = axis_alignment_error_deg(target_basis_semantic, desired_basis[2])
    if (
        long_axis_error > float(orientation["long_axis_tolerance_deg"])
        or semantic_error > float(orientation["long_axis_tolerance_deg"])
    ):
        return ()

    semantic_local_for_output = (
        tuple(-value for value in semantic_local)
        if obj.shape == "concave_rectangle" else semantic_local
    )
    preserved_semantic = transform_vector(target_q, semantic_local_for_output)
    face_normal_base = obj.source.get("visible_face_normal_base")
    if _vector_available(face_normal_base):
        face_normal_local = _normalize(transform_vector(
            current_q_inverse, _normalize(face_normal_base),
        ))
    elif obj.shape == "triangle":
        face_normal_local = source_basis[1]
    else:
        face_normal_local = semantic_local
    target_face_normal = transform_vector(target_q, face_normal_local)
    target_tilt = math.degrees(math.acos(max(0.0, min(1.0, abs(target_face_normal[2])))))
    return ({
        "frame_id": "base_link",
        "position_m": [float(value) for value in target_position_m[:3]],
        "orientation_xyzw": list(target_q),
        "target_tilt_deg": target_tilt,
        "target_orientation_source": "measured_object_local_semantic_basis_alignment",
        "forced_nominal_flip": False,
        "beam_axis_world": list(beam),
        semantic_name: list(preserved_semantic),
        "semantic_component": preserved_semantic[2],
        "source_observed_long_axis_local": list(long_local),
        "source_long_axis_local": list(source_basis[0]),
        "source_semantic_axis_local": list(semantic_local_for_output),
        "target_face_normal_world": list(target_face_normal),
        "semantic_axis_alignment_error_deg": semantic_error,
        "long_axis_alignment_error_deg": long_axis_error,
        "wide_face_is_primary_surface": obj.shape != "rectangle" or semantic_ok,
    },)


def build_triangle_tabletop_step_pose(
    obj: SceneObjectState,
    target_xy_m: Sequence[float],
    config: StackDemoConfig,
) -> dict[str, Any] | None:
    """Rotate one measured rigid 45-degree step toward apex-up on the table."""
    current = observed_object_pose(obj)
    if current is None:
        return None
    long_axis = _normalize(obj.source["long_axis_base"])
    apex_axis = _normalize(obj.source["designated_right_angle_edge_base"])
    step = float(config.section("house_orientation")[
        "triangle_tabletop_reorientation_step_deg"
    ])
    candidates = []
    for signed_step in (-step, step):
        delta = quaternion_from_axis_angle(long_axis, signed_step)
        target_q = quaternion_multiply(delta, current["orientation_xyzw"])
        target_apex = transform_vector(delta, apex_axis)
        candidates.append((target_apex[2], signed_step, target_q, target_apex))
    _, signed_step, target_q, target_apex = max(candidates, key=lambda item: item[0])

    # Put the rotated OBB back on the calibrated table rather than carrying it
    # to the house.  The extra four millimetres are applied later at release.
    rotation = quaternion_to_matrix(target_q)
    half_height = sum(
        abs(rotation[2][axis]) * 0.5 * float(obj.size_xyz_m[axis])
        for axis in range(3)
    )
    table_z = float(config.section("house").get("table_surface_z_m", 0.0))
    return {
        "frame_id": "base_link",
        "position_m": [float(target_xy_m[0]), float(target_xy_m[1]), table_z + half_height],
        "orientation_xyzw": list(target_q),
        "target_orientation_source": "incremental_tabletop_apex_up_step",
        "triangle_tabletop_reorientation_step_deg": signed_step,
        "designated_right_angle_edge_world": list(target_apex),
        "target_apex_vertical_component": float(target_apex[2]),
        "target_tilt_deg": abs(signed_step),
        "forced_nominal_flip": False,
    }


def build_airborne_orientation_plan(
    scene: ClutterSceneState,
    obj: SceneObjectState,
    grasp_yaw_deg: float,
    target_object_pose: Mapping[str, Any],
    config: StackDemoConfig,
) -> dict[str, Any] | None:
    """Solve the rigid grasp relation and rotate around a fixed world TCP point."""
    current_object = observed_object_pose(obj)
    if current_object is None:
        return None
    grasp_tcp = {
        "frame_id": "base_link",
        "position_m": list(obj.center_xyz_m),
        "orientation_xyzw": list(downward_quaternion_for_yaw([1.0, 0.0, 0.0, 0.0], grasp_yaw_deg)),
    }
    grasp_tcp_object = transform_multiply(
        transform_inverse(pose_to_transform(grasp_tcp)), pose_to_transform(current_object),
    )
    release_tcp_transform = transform_multiply(
        pose_to_transform(target_object_pose), transform_inverse(grasp_tcp_object),
    )
    release_tcp = transform_to_pose(release_tcp_transform)
    actual_angle = rotation_error_deg(grasp_tcp["orientation_xyzw"], release_tcp["orientation_xyzw"])
    rotation_radius = 0.5 * math.sqrt(sum(float(value) ** 2 for value in obj.size_xyz_m))
    airborne = _select_airborne_position(scene, obj, rotation_radius, config)
    if airborne is None:
        return None
    fixed_position, candidate_checks = airborne
    step = float(config.section("house_orientation")["orientation_interpolation_step_deg"])
    waypoint_count = max(1, int(math.ceil(actual_angle / step)))
    waypoints = []
    for index in range(waypoint_count + 1):
        fraction = index / waypoint_count
        tcp_pose = {
            "frame_id": "base_link",
            "position_m": list(fixed_position),
            "orientation_xyzw": list(slerp(
                grasp_tcp["orientation_xyzw"], release_tcp["orientation_xyzw"], fraction,
            )),
            "motion_role": "fixed_grasp_tcp_3d_rotation",
        }
        tool0_pose = _tool0_pose_for_tcp(tcp_pose, config)
        waypoints.append({
            **tcp_pose,
            "grasp_tcp_pose": dict(tcp_pose),
            "tool0_pose": tool0_pose,
            "tcp_position_error_m": math.dist(tcp_pose["position_m"], fixed_position),
        })
    tolerance = float(config.section("house_orientation")["fixed_tcp_position_tolerance_m"])
    sweep_checks = _orientation_sweep_checks(scene, obj, fixed_position, rotation_radius, tolerance, waypoints, config)
    if not all(value for value in sweep_checks.values() if isinstance(value, bool)):
        return None
    target_position = list(target_object_pose["position_m"])
    final_pre_place = {
        "frame_id": "base_link",
        "position_m": target_position[:2] + [fixed_position[2]],
        "orientation_xyzw": list(release_tcp["orientation_xyzw"]),
        "motion_role": "final_orientation_transport",
    }
    return {
        "mode": "fixed_grasp_tcp_3d_rotation",
        "adjustment_location": "safe_airborne_zone",
        "adjustment_kind": "minimum_required_target_orientation_alignment",
        "semantic_face_reorientation_required": False,
        "rotation_axis_world": _delta_axis(grasp_tcp["orientation_xyzw"], release_tcp["orientation_xyzw"]),
        "actual_rotation_angle_deg": actual_angle,
        "target_tilt_deg": target_object_pose["target_tilt_deg"],
        "forced_nominal_flip": False,
        "start_orientation_xyzw": list(grasp_tcp["orientation_xyzw"]),
        "target_orientation_xyzw": list(release_tcp["orientation_xyzw"]),
        "fixed_grasp_tcp_position_m": list(fixed_position),
        "waypoints": waypoints,
        "orientation_adjustment_complete_pose": dict(waypoints[-1]["grasp_tcp_pose"]),
        "safe_orientation_adjustment_approach_pose": {
            **dict(waypoints[0]["grasp_tcp_pose"]),
            "motion_role": "safe_orientation_adjustment_approach",
        },
        "final_pre_place_pose": final_pre_place,
        "release_grasp_tcp_pose": release_tcp,
        "grasp_tcp_object_transform": [list(row) for row in grasp_tcp_object],
        "airborne_candidate_checks": candidate_checks,
        "orientation_sweep_checks": sweep_checks,
        "held_object_rotation_radius_m": rotation_radius,
    }


def orientation_airborne_blocking_tracks(
    scene: ClutterSceneState,
    obj: SceneObjectState,
    config: StackDemoConfig,
) -> tuple[str, ...]:
    """Return concrete objects preventing every otherwise legal flip zone."""
    radius = 0.5 * math.sqrt(sum(float(value) ** 2 for value in obj.size_xyz_m))
    available, blockers = _airborne_positions_and_blockers(scene, obj, radius, config)
    return () if available else tuple(sorted(blockers))


def _select_airborne_position(
    scene: ClutterSceneState,
    obj: SceneObjectState,
    radius: float,
    config: StackDemoConfig,
) -> tuple[tuple[float, float, float], dict[str, Any]] | None:
    output, _ = _airborne_positions_and_blockers(scene, obj, radius, config)
    return None if not output else min(output, key=lambda item: (item[0], item[1]))[2:]


def _airborne_positions_and_blockers(
    scene: ClutterSceneState,
    obj: SceneObjectState,
    radius: float,
    config: StackDemoConfig,
) -> tuple[list[tuple[Any, ...]], set[str]]:
    house = config.section("house")
    orientation = config.section("house_orientation")
    safety_margin = float(orientation["airborne_adjustment_safety_margin_m"])
    minimum_house_distance = float(orientation["minimum_airborne_adjustment_house_distance_m"])
    highest = max((_top(item) for item in (*scene.current_objects, *scene.collision_obstacles)), default=obj.center_xyz_m[2])
    z = max(float(config.section("motion")["safe_pre_rotate_height_m"]), highest + radius + safety_margin)
    house_xy = tuple(float(value) for value in house["origin_center_base_m"][:2])
    candidates = [tuple(float(value) for value in item[:2]) for item in house["orientation_staging_candidates_base_m"]]
    output: list[tuple[Any, ...]] = []
    blockers: set[str] = set()
    inset = float(config.section("safety")["workspace_inset_m"])
    for x, y in candidates:
        house_distance = math.dist((x, y), house_xy)
        workspace_safe = bool(
            config.workspace["xmin"] + inset + radius <= x <= config.workspace["xmax"] - inset - radius
            and config.workspace["ymin"] + inset + radius <= y <= config.workspace["ymax"] - inset - radius
            and z + radius <= config.workspace["zmax"]
        )
        obstacle_clearances = [
            (
                math.dist((x, y), item.center_xyz_m[:2])
                - radius - 0.5 * max(item.size_xyz_m[:2]),
                item.track_id,
            )
            for item in (*scene.current_objects, *scene.collision_obstacles)
            if item.track_id != obj.track_id
        ]
        obstacle_clearance = min((item[0] for item in obstacle_clearances), default=1.0)
        base_legal = workspace_safe and house_distance >= minimum_house_distance
        if base_legal and obstacle_clearance < safety_margin:
            blockers.update(
                track_id for clearance, track_id in obstacle_clearances
                if clearance < safety_margin
            )
        if base_legal and obstacle_clearance >= safety_margin:
            output.append((math.dist((x, y), obj.center_xyz_m[:2]), -obstacle_clearance, (x, y, z), {
                "workspace_safe": True, "house_distance_safe": True, "height_safe": True,
                "house_distance_m": house_distance, "minimum_obstacle_clearance_m": obstacle_clearance,
            }))
    return output, blockers


def _orientation_sweep_checks(
    scene: ClutterSceneState,
    obj: SceneObjectState,
    fixed: Sequence[float],
    radius: float,
    tolerance: float,
    waypoints: Sequence[Mapping[str, Any]],
    config: StackDemoConfig,
) -> dict[str, Any]:
    tcp_fixed = all(math.dist(item["position_m"], fixed) <= tolerance for item in waypoints)
    house_xy = tuple(float(value) for value in config.section("house")["origin_center_base_m"][:2])
    house_safe = math.dist(fixed[:2], house_xy) >= float(config.section("house_orientation")["minimum_airborne_adjustment_house_distance_m"])
    obstacle_safe = all(
        math.dist(fixed, other.center_xyz_m) > radius + 0.5 * math.sqrt(sum(value ** 2 for value in other.size_xyz_m))
        for other in (*scene.current_objects, *scene.collision_obstacles)
        if other.track_id != obj.track_id
    )
    return {
        "fixed_tcp_position": tcp_fixed,
        "far_from_house": house_safe,
        "held_object_obb_sweep_clear": obstacle_safe,
        "gf225_fingertip_sweep_clear": obstacle_safe,
        "gf225_palm_sweep_clear": obstacle_safe,
        "d435i_mount_sweep_clear": obstacle_safe,
        "table_clear": fixed[2] - radius > config.workspace["zmin"],
        "workspace_clear": fixed[2] + radius < config.workspace["zmax"],
        "maximum_tcp_position_error_m": max(math.dist(item["position_m"], fixed) for item in waypoints),
    }


def _tool0_pose_for_tcp(tcp_pose: Mapping[str, Any], config: StackDemoConfig) -> dict[str, Any]:
    offset = config.section("gripper")["tcp_offset_tool_m"]
    tool0_tcp = pose_to_transform({
        "position_m": list(offset), "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    })
    return transform_to_pose(transform_multiply(pose_to_transform(tcp_pose), transform_inverse(tool0_tcp)))


def _delta_axis(start: Sequence[float], target: Sequence[float]) -> list[float]:
    delta = quaternion_multiply(target, quaternion_inverse(start))
    sine = math.sqrt(sum(value * value for value in delta[:3]))
    return [0.0, 0.0, 1.0] if sine <= 1e-9 else [value / sine for value in delta[:3]]


def _rotate_about_axis(vector: Sequence[float], axis: Sequence[float], angle_deg: float) -> tuple[float, float, float]:
    unit = _normalize(axis)
    angle = math.radians(angle_deg)
    cross = _cross(unit, vector)
    dot = sum(a * b for a, b in zip(unit, vector))
    return tuple(float(vector[index]) * math.cos(angle) + cross[index] * math.sin(angle) + unit[index] * dot * (1 - math.cos(angle)) for index in range(3))


def _normalize(value: Sequence[float]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(float(item) ** 2 for item in value[:3]))
    if norm <= 1e-9:
        raise ValueError("zero semantic axis")
    return tuple(float(item) / norm for item in value[:3])


def _unit_horizontal(value: Sequence[float]) -> tuple[float, float, float]:
    return _normalize((float(value[0]), float(value[1]), 0.0))


def _cross(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return (left[1] * right[2] - left[2] * right[1], left[2] * right[0] - left[0] * right[2], left[0] * right[1] - left[1] * right[0])


def _orthonormal_basis(
    primary: Sequence[float],
    semantic: Sequence[float],
) -> tuple[tuple[float, float, float], ...]:
    """Return right-handed basis columns (primary, cross, semantic)."""
    # The semantic face/apex observation is the hard constraint.  Depth PCA
    # long axes can be several degrees non-orthogonal, so fit that noisy axis
    # to the plane perpendicular to the semantic axis, as required by the
    # confirmed rigid geometry.
    third = _normalize(semantic)
    component = sum(float(primary[index]) * third[index] for index in range(3))
    first = _normalize(tuple(
        float(primary[index]) - component * third[index] for index in range(3)
    ))
    second = _normalize(_cross(third, first))
    return first, second, third


def _basis_mapping_rotation(
    source: Sequence[Sequence[float]],
    target: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], ...]:
    """Return ``target_columns * source_columns.T``."""
    return tuple(tuple(
        sum(float(target[column][row]) * float(source[column][axis]) for column in range(3))
        for axis in range(3)
    ) for row in range(3))


def _vector_available(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 3


def _scene_object(scene: ClutterSceneState, track_id: str) -> SceneObjectState | None:
    return scene.object_by_track(track_id) or next((item for item in scene.collision_obstacles if item.track_id == track_id), None)


def _top(obj: SceneObjectState) -> float:
    return obj.center_xyz_m[2] + 0.5 * obj.size_xyz_m[2]
