"""Deterministic uniquely-solvable Sudoku rules for the agentic RL demo."""

from __future__ import annotations

import copy
import random
from typing import Any

# -- Constants --------------------------------------------------------------

DEFAULT_DIFFICULTY = "medium"
DEFAULT_ACTION_BUDGET_RATIO = 2  # action_budget = empty_cells * ratio

DIFFICULTY_CLUES: dict[str, int] = {
    "easy": 45,
    "medium": 34,
    "hard": 28,
}

# -- Tool schemas (OpenAI function-calling format) --------------------------

INSPECT_TOOL = {
    "type": "function",
    "function": {
        "name": "inspect_candidates",
        "description": "List the digits that can legally go in an empty cell without violating Sudoku rules.",
        "parameters": {
            "type": "object",
            "properties": {
                "row": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 9,
                    "description": "Row index (1-based).",
                },
                "col": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 9,
                    "description": "Column index (1-based).",
                },
            },
            "required": ["row", "col"],
            "additionalProperties": False,
        },
    },
}

PLACE_TOOL = {
    "type": "function",
    "function": {
        "name": "place_digit",
        "description": "Place a digit into an empty cell. The move is validated against Sudoku rules.",
        "parameters": {
            "type": "object",
            "properties": {
                "row": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 9,
                    "description": "Row index (1-based).",
                },
                "col": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 9,
                    "description": "Column index (1-based).",
                },
                "digit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 9,
                    "description": "Digit to place (1-9).",
                },
            },
            "required": ["row", "col", "digit"],
            "additionalProperties": False,
        },
    },
}

UNDO_TOOL = {
    "type": "function",
    "function": {
        "name": "undo",
        "description": "Undo the most recent place_digit action. Has no effect if no placements have been made.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
}

TOOLS = [INSPECT_TOOL, PLACE_TOOL, UNDO_TOOL]
TOOL_BY_NAME = {tool["function"]["name"]: tool for tool in TOOLS}


# -- Board helpers (0-indexed internally) ------------------------------------

def _is_valid(board: list[list[int]], row: int, col: int, digit: int) -> bool:
    """Check if *digit* is legal at (row, col) on the current *board*.

    All indices are 0-based. The cell must be empty.
    """
    for i in range(9):
        if board[row][i] == digit or board[i][col] == digit:
            return False
    br, bc = row - row % 3, col - col % 3
    for r in range(br, br + 3):
        for c in range(bc, bc + 3):
            if board[r][c] == digit:
                return False
    return True


def _count_solutions(board: list[list[int]], limit: int = 2) -> int:
    """Backtracking solver that stops once *limit* solutions are found.

    The *board* is restored to its original state after the search.
    """
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                count = 0
                for d in range(1, 10):
                    if _is_valid(board, r, c, d):
                        board[r][c] = d
                        count += _count_solutions(board, limit - count)
                        board[r][c] = 0
                        if count >= limit:
                            return count
                return count
    return 1  # board is full -- one solution


def _fill_board(rng: random.Random) -> list[list[int]]:
    """Generate a random complete Sudoku board."""
    board = [[0] * 9 for _ in range(9)]

    # Fill diagonal 3x3 boxes first -- they don't overlap so they're independent.
    for box in range(0, 9, 3):
        digits = list(range(1, 10))
        rng.shuffle(digits)
        idx = 0
        for r in range(box, box + 3):
            for c in range(box, box + 3):
                board[r][c] = digits[idx]
                idx += 1

    # Fill the rest via backtracking with shuffled digit order.
    def _solve(pos: int) -> bool:
        if pos == 81:
            return True
        r, c = divmod(pos, 9)
        if board[r][c] != 0:
            return _solve(pos + 1)
        digits = list(range(1, 10))
        rng.shuffle(digits)
        for d in digits:
            if _is_valid(board, r, c, d):
                board[r][c] = d
                if _solve(pos + 1):
                    return True
                board[r][c] = 0
        return False

    _solve(0)
    return board


def _dig_holes(
    solution: list[list[int]], target_clues: int, rng: random.Random
) -> list[list[int]]:
    """Remove cells from a complete board while keeping a unique solution."""
    puzzle = copy.deepcopy(solution)
    positions = [(r, c) for r in range(9) for c in range(9)]
    rng.shuffle(positions)
    clues = 81
    for r, c in positions:
        if clues <= target_clues:
            break
        backup = puzzle[r][c]
        puzzle[r][c] = 0
        test = copy.deepcopy(puzzle)
        if _count_solutions(test, 2) == 1:
            clues -= 1
        else:
            puzzle[r][c] = backup  # uniqueness violated -- restore
    return puzzle


def count_empty(puzzle: list[list[int]]) -> int:
    """Return the number of empty cells in *puzzle*."""
    return sum(1 for r in range(9) for c in range(9) if puzzle[r][c] == 0)


# -- Public API -------------------------------------------------------------

def generate_board(
    difficulty: str = DEFAULT_DIFFICULTY, *, seed: int = 2026
) -> tuple[list[list[int]], list[list[int]]]:
    """Return *(puzzle, solution)* for the given difficulty.

    The puzzle has a unique solution which is returned as *solution*.
    """
    if difficulty not in DIFFICULTY_CLUES:
        raise ValueError(
            f"difficulty must be one of {list(DIFFICULTY_CLUES)}, got {difficulty!r}"
        )
    rng = random.Random(seed)
    target_clues = DIFFICULTY_CLUES[difficulty]
    puzzle: list[list[int]] = []
    solution: list[list[int]] = []
    for _ in range(20):
        solution = _fill_board(rng)
        puzzle = _dig_holes(solution, target_clues, rng)
        if count_empty(puzzle) >= 81 - target_clues:
            break
    return puzzle, solution


def get_candidates(board: list[list[int]], row: int, col: int) -> list[int]:
    """Return the legal digits for cell (row, col) given the current *board*.

    Indices are 0-based and the cell must be empty.
    """
    return [d for d in range(1, 10) if _is_valid(board, row, col, d)]


def validate_placement(
    board: list[list[int]],
    puzzle: list[list[int]],
    row: int,
    col: int,
    digit: int,
) -> dict[str, Any]:
    """Validate a place_digit move without mutating the board.

    Indices are 0-based. The cell must be empty in both *puzzle* (not a clue)
    and *board* (not already filled), and the digit must not conflict.
    """
    if not (0 <= row < 9 and 0 <= col < 9):
        return {"valid": False, "error": "row and col must be in 0..8"}
    if not (1 <= digit <= 9):
        return {"valid": False, "error": "digit must be in 1..9"}
    if puzzle[row][col] != 0:
        return {"valid": False, "error": f"cell ({row + 1},{col + 1}) is a pre-filled clue"}
    if board[row][col] != 0:
        return {"valid": False, "error": f"cell ({row + 1},{col + 1}) is already filled"}
    if not _is_valid(board, row, col, digit):
        return {"valid": False, "error": f"digit {digit} conflicts in row {row + 1}, column {col + 1}, or box"}
    return {"valid": True}


def is_solved(board: list[list[int]], solution: list[list[int]]) -> bool:
    """Return True if *board* matches *solution* exactly."""
    return all(board[r][c] == solution[r][c] for r in range(9) for c in range(9))


def is_complete(board: list[list[int]]) -> bool:
    """Return True if the board has no empty cells."""
    return all(board[r][c] != 0 for r in range(9) for c in range(9))


def make_prompt(record: dict[str, Any]) -> str:
    """Build the user prompt from a puzzle record without revealing the solution."""
    puzzle = record["puzzle"]
    difficulty = record.get("difficulty", DEFAULT_DIFFICULTY)
    action_budget = int(record.get("action_budget", 0))

    rows = []
    for r in range(9):
        cells = [str(puzzle[r][c]) if puzzle[r][c] else "." for c in range(9)]
        rows.append(" ".join(cells))
    board_str = "\n".join(rows)

    return (
        f"Solve the {difficulty} Sudoku puzzle below. Empty cells are shown as dots.\n"
        f"You have {action_budget} actions. Rows and columns are numbered 1-9.\n\n"
        f"{board_str}\n\n"
        f"Use inspect_candidates to check legal digits for a cell, "
        f"place_digit to fill a cell, and undo to revert your last placement. "
        f"The puzzle has a unique solution."
    )


def replay_episode(
    puzzle: list[list[int]],
    solution: list[list[int]],
    actions: list[dict[str, Any]],
    action_budget: int,
) -> dict[str, Any]:
    """Replay tool-call actions on a copy of the puzzle and return statistics.

    Each action is a dict with keys *name* and *arguments* (already parsed).
    Row/col in arguments are 1-based (as in the tool schema).
    """
    board = copy.deepcopy(puzzle)
    history: list[tuple[int, int, int]] = []
    invalid_count = 0

    for action in actions[:action_budget]:
        name = action.get("name", "")
        args = action.get("arguments", {})
        if not isinstance(args, dict):
            invalid_count += 1
            continue

        if name == "place_digit":
            row = int(args.get("row", 0)) - 1
            col = int(args.get("col", 0)) - 1
            digit = int(args.get("digit", 0))
            result = validate_placement(board, puzzle, row, col, digit)
            if result["valid"]:
                board[row][col] = digit
                history.append((row, col, digit))
            else:
                invalid_count += 1
        elif name == "undo":
            if history:
                r, c, _ = history.pop()
                board[r][c] = 0
            else:
                invalid_count += 1
        elif name == "inspect_candidates":
            pass  # informational -- no board change
        else:
            invalid_count += 1

    total_empty = count_empty(puzzle)
    correct_fills = 0
    wrong_fills = 0
    for r in range(9):
        for c in range(9):
            if puzzle[r][c] == 0 and board[r][c] != 0:
                if board[r][c] == solution[r][c]:
                    correct_fills += 1
                else:
                    wrong_fills += 1

    actions_used = min(len(actions), action_budget)
    solved = is_solved(board, solution)

    return {
        "solved": solved,
        "correct_fills": correct_fills,
        "wrong_fills": wrong_fills,
        "invalid_actions": invalid_count,
        "actions_used": actions_used,
        "action_budget": action_budget,
        "total_empty": total_empty,
        "solve_rate": 1.0 if solved else 0.0,
        "invalid_action_rate": invalid_count / max(actions_used, 1),
    }


def score_episode(
    puzzle: list[list[int]],
    solution: list[list[int]],
    actions: list[dict[str, Any]],
    action_budget: int,
) -> float:
    """Return a float reward for the episode.

    - Fully solved with spare actions: ``0.8 + 0.2 * efficiency``
    - Partial progress (net correct fills): ``0.1 * net / total_empty``
    - No progress or net negative: ``-1.0``
    """
    stats = replay_episode(puzzle, solution, actions, action_budget)

    if stats["solved"]:
        efficiency = (action_budget - stats["actions_used"]) / max(action_budget - 1, 1)
        return round(0.8 + 0.2 * efficiency, 4)

    if not actions:
        return -1.0

    net = stats["correct_fills"] - stats["wrong_fills"]
    if net <= 0:
        if stats["invalid_actions"] > 0 and stats["correct_fills"] == 0:
            return -1.0
        return -0.5

    return round(0.1 * net / max(stats["total_empty"], 1), 4)