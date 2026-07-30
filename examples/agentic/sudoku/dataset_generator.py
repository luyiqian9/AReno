"""Generate reproducible Sudoku puzzle tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from game import (
    DEFAULT_ACTION_BUDGET_RATIO,
    DEFAULT_DIFFICULTY,
    DIFFICULTY_CLUES,
    count_empty,
    generate_board,
)


def generate_records(
    count: int = 256, *, seed: int = 2026, difficulty: str = DEFAULT_DIFFICULTY
) -> list[dict]:
    """Return deterministic uniquely-solvable Sudoku records.

    Each record contains *puzzle*, *solution*, *difficulty*, and *action_budget*.
    """
    records: list[dict] = []
    for i in range(count):
        puzzle, solution = generate_board(difficulty, seed=seed + i)
        empty = count_empty(puzzle)
        records.append(
            {
                "id": f"sudoku-{difficulty}-{i + 1:05d}",
                "puzzle": puzzle,
                "solution": solution,
                "difficulty": difficulty,
                "action_budget": empty * DEFAULT_ACTION_BUDGET_RATIO,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--difficulty",
 choices=list(DIFFICULTY_CLUES),
        default=DEFAULT_DIFFICULTY,
    )
    args = parser.parse_args()
    records = generate_records(args.count, seed=args.seed, difficulty=args.difficulty)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()