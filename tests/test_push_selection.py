#!/usr/bin/env python3

import os
import tempfile
from types import SimpleNamespace

from tools.workflows.stack_demo.push_selection import choose_clearance_action


def test_pick_away_candidate_has_deterministic_priority_over_nudge():
    pick_away = {
        "candidate_id": "clear_pick_away",
        "action_type": "pick_away",
        "score": 1.0,
    }
    nudge = {
        "candidate_id": "push_nudge",
        "action_type": "nudge",
        "score": 9.0,
    }
    with tempfile.TemporaryDirectory() as cycle_dir:
        selected, report = choose_clearance_action(
            SimpleNamespace(instruction="stack blocks"),
            cycle_dir,
            {"id": "target"},
            [pick_away, nudge],
            {"action_history": []},
            step_index=1,
        )
        assert selected["candidate_id"] == "clear_pick_away"
        assert report["selection_source"] == "deterministic_pick_away_priority"
        assert os.path.exists(os.path.join(cycle_dir, "clearance_step_01_llm_selection.json"))


if __name__ == "__main__":
    test_pick_away_candidate_has_deterministic_priority_over_nudge()
    print("push selection tests passed")
