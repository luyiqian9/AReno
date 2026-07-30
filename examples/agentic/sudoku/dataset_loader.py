"""Dataset loader for Sudoku agentic training."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_generator  # noqa: E402
from game import DEFAULT_ACTION_BUDGET_RATIO, DEFAULT_DIFFICULTY, count_empty, make_prompt  # noqa: E402


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    """Normalize records while remaining tokenizer and processor independent."""

    raw = _load_raw(dataset_path, default_loader)
    return [_normalize(dict(row), index) for index, row in enumerate(raw, start=1)]


def _load_raw(dataset_path: str, default_loader) -> list[dict]:
    """Return raw records from *dataset_path*, falling back to the generator."""

    path = Path(dataset_path).expanduser()
    if not path.exists():
        return dataset_generator.generate_records()
    return list(default_loader(dataset_path))


def _normalize(record: dict, index: int) -> dict:
    difficulty = record.get("difficulty", DEFAULT_DIFFICULTY)
    record["difficulty"] = difficulty
    # Ensure puzzle is a 9x9 list of ints (0 = empty).
    puzzle = record["puzzle"]
    if not (isinstance(puzzle, list) and len(puzzle) == 9 and all(len(r) == 9 for r in puzzle)):
        raise ValueError(f"sudoku record {index}: puzzle must be a 9x9 grid")
    record["puzzle"] = [[int(v) for v in r] for r in puzzle]
    # Keep solution if present (needed for scoring).
    if "solution" in record:
        record["solution"] = [[int(v) for v in r] for r in record["solution"]]
    # Derive action budget from empty-cell count if not specified.
    if "action_budget" not in record:
        record["action_budget"] = count_empty(record["puzzle"]) * DEFAULT_ACTION_BUDGET_RATIO
    else:
        record["action_budget"] = int(record["action_budget"])
    record["id"] = str(record.get("id", f"sudoku-{difficulty}-{index:05d}"))
    record["prompt"] = make_prompt(record)
    return record