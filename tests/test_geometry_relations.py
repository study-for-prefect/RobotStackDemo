#!/usr/bin/env python3

from robot_scene_pipeline.geometry_relations import (
    blocks_grasp,
    build_geometry_relations,
    is_near,
    is_on,
    is_supporting,
    safe_to_push,
)


def make_object(object_id, center, size=(0.04, 0.04, 0.03), **extra):
    obj = {
        "id": object_id,
        "label": object_id,
        "geometry_center_m": list(center),
        "dimensions_m": list(size),
        "table_yaw_deg": 0.0,
        "visible": True,
    }
    obj.update(extra)
    return obj


def test_near_relation():
    first = make_object("red_1", (0.40, 0.00, 0.015))
    second = make_object("green_1", (0.43, 0.00, 0.015))
    assert is_near(first, second) is True


def test_on_relation():
    red = make_object("square_red_1", (0.40, 0.00, 0.015))
    green = make_object("square_green_1", (0.40, 0.00, 0.045))
    assert is_on(green, red) is True
    assert is_supporting(red, green) is True


def test_blocking_grasp():
    target = make_object("square_green_1", (0.40, 0.00, 0.015))
    obstacle = make_object(
        "rectangle_1",
        (0.47, 0.00, 0.015),
        size=(0.06, 0.03, 0.03),
    )
    assert blocks_grasp(obstacle, target) is True


def test_locked_object_not_safe_to_push():
    target = make_object("square_green_1", (0.40, 0.00, 0.015))
    red = make_object(
        "square_red_1",
        (0.47, 0.00, 0.015),
        role="base",
        state="locked",
        pushable=True,
    )
    assert safe_to_push(red, [red, target], [1.0, 0.0, 0.0]) is False


def test_should_push_away_relation():
    target = make_object("square_green_1", (0.40, 0.00, 0.015))
    obstacle = make_object(
        "rectangle_1",
        (0.47, 0.00, 0.015),
        size=(0.06, 0.03, 0.03),
        pushable=True,
    )
    relations = build_geometry_relations(
        [target, obstacle],
        target_id="square_green_1",
        table_bounds={"xmin": 0.20, "xmax": 0.80, "ymin": -0.30, "ymax": 0.30},
    )
    push_relations = [
        relation
        for relation in relations
        if relation["type"] == "should_push_away"
    ]
    assert len(push_relations) == 1
    assert push_relations[0]["subject"] == "rectangle_1"
    assert push_relations[0]["object"] == "square_green_1"
    assert push_relations[0]["direction_base"] == [1.0, 0.0, 0.0]
    assert push_relations[0]["reason"] == "blocking_grasp_and_safe_to_push"


if __name__ == "__main__":
    test_near_relation()
    test_on_relation()
    test_blocking_grasp()
    test_locked_object_not_safe_to_push()
    test_should_push_away_relation()
    print("geometry_relations tests passed")
