"""Outcome and process reward for Sudoku trajectories."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import score_episode  # noqa: E402


def reward_fn(record) -> float:
    """Replay tool calls and reward correct, efficient solving."""

    source = dict(record.source_record)
    puzzle = source["puzzle"]
    solution = source["solution"]
    action_budget = int(source["action_budget"])

    actions: list[dict] = []
    for call in record.tool_calls:
        name = call.get("name")
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return -1.0
        actions.append({"name": name, "arguments": arguments})

    return score_episode(puzzle, solution, actions, action_budget)