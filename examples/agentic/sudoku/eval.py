"""Batch evaluation script for the Sudoku agentic demo.

Runs the multi-turn agent against a served model on multiple Sudoku puzzles
across difficulty levels and reports aggregate statistics:

    solve rate, invalid-action rate, average steps, average reward

Usage::

    areno serve --model-path /path/to/checkpoint --port 8001
    python examples/agentic/sudoku/eval.py \
        --base-url http://127.0.0.1:8001/v1 \
        --difficulties easy medium hard \
        --count 20 --output /tmp/sudoku-eval.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402

SYSTEM_PROMPT = (
    "You are a methodical Sudoku solver. On every turn call exactly one tool: "
    "inspect_candidates to check legal digits, place_digit to fill a cell, or undo "
    "to revert a mistake. Use the clues and prior placements to deduce each digit. "
    "When the board is complete, summarize the outcome without calling a tool."
)


def run_episode(puzzle, solution, action_budget, *, base_url, api_key, model, timeout):
    """Run one full Sudoku episode and return per-episode stats."""
    import requests as _requests

    board = copy.deepcopy(puzzle)
    history: list[tuple[int, int, int]] = []
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": game.make_prompt({
            "puzzle": puzzle,
            "difficulty": "eval",
            "action_budget": action_budget,
        })},
    ]

    for step in range(1, action_budget + 1):
        turn_messages = [
            *messages,
            {"role": "user", "content": f"Action {step} of {action_budget}: call a tool now."},
        ]
        try:
            resp = _requests.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": turn_messages,
                    "tools": game.TOOLS,
                    "tool_choice": "auto",
                    "stream": False,
                },
                timeout=timeout,
            )
            resp.raise_for_status()
        except Exception:
            return {
                "solved": False,
                "steps": step - 1,
                "invalid_actions": 0,
                "error": "request_failed",
            }

        data = resp.json()
        msg = data["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []
        if len(tool_calls) != 1:
            return {"solved": False, "steps": step - 1, "invalid_actions": 0, "error": "no_tool_call"}

        call = tool_calls[0]
        name = call["function"]["name"]
        args_str = call["function"]["arguments"]
        try:
            args = json.loads(args_str)
        except (json.JSONDecodeError, TypeError):
            return {"solved": False, "steps": step, "invalid_actions": 1, "error": "bad_arguments"}

        result, assistant_msg = _execute_tool(name, args, board, puzzle, history)
        messages.append(assistant_msg)
        messages.append({
            "role": "tool",
            "tool_call_id": call["id"],
            "name": name,
            "content": json.dumps(result),
        })

        if game.is_complete(board):
            solved = game.is_solved(board, solution)
            return {"solved": solved, "steps": step, "invalid_actions": 0}

    return {"solved": False, "steps": action_budget, "invalid_actions": 0, "error": "budget_exhausted"}


def _execute_tool(name, args, board, puzzle, history):
    """Execute one tool call; return (result_dict, assistant_message_dict)."""
    invalid = {"valid": False, "error": "unknown"}
    assistant_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "eval", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}],
    }

    if name == "inspect_candidates":
        row = game.safe_int(args.get("row")) - 1
        col = game.safe_int(args.get("col")) - 1
        if not (0 <= row < 9 and 0 <= col < 9):
            return {"valid": False, "error": "row/col out of range"}, assistant_msg
        if puzzle[row][col] != 0:
            return {"valid": False, "error": "clue"}, assistant_msg
        if board[row][col] != 0:
            return {"valid": False, "error": "filled"}, assistant_msg
        return {"valid": True, "candidates": game.get_candidates(board, row, col)}, assistant_msg

    if name == "place_digit":
        row = game.safe_int(args.get("row")) - 1
        col = game.safe_int(args.get("col")) - 1
        digit = game.safe_int(args.get("digit"))
        result = game.validate_placement(board, puzzle, row, col, digit)
        if result["valid"]:
            board[row][col] = digit
            history.append((row, col, digit))
        return result, assistant_msg

    if name == "undo":
        if not history:
            return {"valid": False, "error": "nothing to undo"}, assistant_msg
        r, c, d = history.pop()
        board[r][c] = 0
        return {"valid": True, "undid": {"row": r + 1, "col": c + 1, "digit": d}}, assistant_msg

    return invalid, assistant_msg


def evaluate_difficulty(difficulty, count, *, seed, base_url, api_key, model, timeout):
    """Evaluate *count* puzzles at one difficulty; return aggregate stats."""
    episodes = []
    for i in range(count):
        puzzle, solution = game.generate_board(difficulty, seed=seed + i)
        budget = game.count_empty(puzzle) * game.DEFAULT_ACTION_BUDGET_RATIO
        stats = run_episode(
            puzzle, solution, budget,
            base_url=base_url, api_key=api_key, model=model, timeout=timeout,
        )
        stats["difficulty"] = difficulty
        stats["puzzle_id"] = f"sudoku-{difficulty}-{i+1:04d}"
        episodes.append(stats)
        print(f"  {difficulty} [{i+1}/{count}] solved={stats['solved']} steps={stats['steps']}")
    return episodes


def aggregate(episodes):
    """Aggregate per-episode stats into difficulty-level summary."""
    n = len(episodes)
    if n == 0:
        return {}
    solved = sum(1 for e in episodes if e["solved"])
    invalid = sum(e.get("invalid_actions", 0) for e in episodes)
    total_steps = sum(e["steps"] for e in episodes)
    errors = sum(1 for e in episodes if e.get("error"))
    return {
        "count": n,
        "solved": solved,
        "solve_rate": round(solved / n, 4),
        "invalid_actions": invalid,
        "invalid_action_rate": round(invalid / max(total_steps, 1), 4),
        "avg_steps": round(total_steps / n, 2),
        "error_rate": round(errors / n, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--api-key", default="token")
    parser.add_argument("--model", default="policy")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--count", type=int, default=20, help="Puzzles per difficulty.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--difficulties", nargs="+", default=["easy", "medium", "hard"])
    parser.add_argument("--output", "-o", type=Path, default=None, help="Write JSON results to this file.")
    args = parser.parse_args()

    all_episodes = []
    results = {}

    for diff in args.difficulties:
        print(f"\n{'='*60}")
        print(f"Evaluating {diff} ({args.count} puzzles)...")
        eps = evaluate_difficulty(
            diff, args.count, seed=args.seed,
            base_url=args.base_url, api_key=args.api_key,
            model=args.model, timeout=args.timeout,
        )
        all_episodes.extend(eps)
        results[diff] = aggregate(eps)

    results["overall"] = aggregate(all_episodes)

    print(f"\n{'='*60}")
    print(f"{'difficulty':<12} {'solve_rate':>12} {'invalid_rate':>14} {'avg_steps':>10} {'errors':>8}")
    print("-" * 60)
    for diff in args.difficulties + ["overall"]:
        if diff in results:
            r = results[diff]
            print(f"{diff:<12} {r.get('solve_rate',0):>12.2%} {r.get('invalid_action_rate',0):>14.2%} {r.get('avg_steps',0):>10.1f} {r.get('error_rate',0):>8.2%}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump({"summary": results, "episodes": all_episodes}, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()