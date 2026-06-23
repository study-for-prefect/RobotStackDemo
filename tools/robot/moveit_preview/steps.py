"""Plan-step selection, validation, and command expansion."""

def selected_steps(plan, args):
    steps = [step for step in plan.get("steps", []) if step.get("status") == "planned"]
    steps = [step for step in steps if step.get("approach_position_m")]
    if args.all_approaches:
        return steps
    for step in steps:
        if int(step.get("step", -1)) == args.step:
            return [step]
    raise RuntimeError("No planned step {} with approach_position_m.".format(args.step))


def validate_complete_plan(plan, args):
    if args.allow_partial_plan:
        return
    bad_steps = []
    for step in plan.get("steps", []):
        status = step.get("status")
        action = step.get("action")
        if action in ("ask_user", "stop"):
            continue
        if status != "planned":
            bad_steps.append((step.get("step"), action, status, "status is not planned"))
            continue
        if not step.get("approach_position_m"):
            bad_steps.append((step.get("step"), action, status, "missing approach_position_m"))
            continue
        if args.path_mode == "full" and not step.get("target_position_m"):
            bad_steps.append((step.get("step"), action, status, "missing target_position_m"))
    if not bad_steps:
        return
    lines = ["Execution refused because the plan is incomplete:"]
    for step_id, action, status, reason in bad_steps:
        lines.append("  step {} {} status={} {}".format(step_id, action, status, reason))
    lines.append("Regenerate the snapshot/plan after making sure all referenced objects are detected with depth.")
    raise RuntimeError("\n".join(lines))


def command_sequence_for_step(step, path_mode, enable_gripper, release_after_pick=False):
    approach = step.get("approach_position_m")
    target = step.get("release_position_m") or step.get("target_position_m")
    action = step.get("action")
    commands = []
    if path_mode == "approach":
        if approach:
            commands.append({"type": "motion", "name": "approach", "position": approach})
        return commands

    if action == "place_on_top" and approach and target:
        if abs(float(approach[0]) - float(target[0])) > 1e-6 or abs(float(approach[1]) - float(target[1])) > 1e-6:
            raise RuntimeError("place_on_top approach and target must have identical XY for vertical descent.")
    if action == "pick" and enable_gripper:
        commands.append({"type": "gripper", "name": "open"})
    if approach:
        commands.append({"type": "motion", "name": "approach", "position": approach})
    if target:
        commands.append(
            {
                "type": "motion",
                "name": "release_z" if action == "place_on_top" else "target",
                "position": target,
            }
        )
    if action == "pick" and enable_gripper:
        commands.append({"type": "gripper", "name": "close"})
    elif action in ("place_relative", "place_on_top") and enable_gripper:
        commands.append({"type": "gripper", "name": "open"})
        if action == "place_on_top":
            commands.append({"type": "wait", "name": "release_wait"})
    if approach and target:
        commands.append({"type": "motion", "name": "retreat", "position": approach})
    if action == "pick" and enable_gripper and release_after_pick and approach and target:
        commands.append({"type": "motion", "name": "release_target", "position": target})
        commands.append({"type": "gripper", "name": "open"})
        commands.append({"type": "motion", "name": "release_retreat", "position": approach})
    return commands


def diagnostic_steps(steps, diagnostic_yaw_deg):
    if not diagnostic_yaw_deg:
        return steps
    output = []
    for step in steps:
        for yaw_deg in diagnostic_yaw_deg:
            diagnostic = dict(step)
            diagnostic["diagnostic_yaw_deg"] = float(yaw_deg)
            diagnostic["target_yaw_deg"] = float(yaw_deg)
            diagnostic["target_yaw_valid"] = True
            diagnostic["yaw_frame"] = "base_link"
            diagnostic["reason"] = "Offset diagnosis at explicit base-link yaw."
            output.append(diagnostic)
    return output
