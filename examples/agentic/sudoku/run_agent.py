"""Bounded multi-turn agent loop for Sudoku."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import sys
from pathlib import Path

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import TOOLS, get_candidates, is_complete, safe_int, validate_placement  # noqa: E402

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    "You are a methodical Sudoku solver. On every turn call exactly one tool: "
    "inspect_candidates to check legal digits, place_digit to fill a cell, or undo "
    "to revert a mistake. Use the clues and prior placements to deduce each digit. "
    "When the board is complete, summarize the outcome without calling a tool."
)

VALID_TOOL_NAMES = {"inspect_candidates", "place_digit", "undo"}


async def run_agent(ctx, batch):
    """Run bounded concurrent Sudoku episodes and preserve exact model outputs."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Sudoku requires `openai` and `httpx`. Install them with `pip install openai`."
        ) from exc

    items = list(batch.iter_samples())
    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        timeout=httpx.Timeout(900.0, connect=30.0),
    )
    client = AsyncOpenAI(base_url=ctx.get_base_url(), api_key=ctx.api_key, http_client=http_client, max_retries=0)
    try:
        grouped = await asyncio.gather(*(_run_episode(item, client) for item in items))
        return AgentTrajectory(turns=[turn for episode in grouped for turn in episode])
    finally:
        await client.close()


async def _run_episode(item, client) -> list[AgentTrajectoryTurn]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": item.prompt}]
    turns = []

    puzzle = copy.deepcopy(item.record["puzzle"])
    action_budget = int(item.record["action_budget"])
    board = copy.deepcopy(puzzle)
    history: list[tuple[int, int, int]] = []

    for action_num in range(1, action_budget + 1):
        turn_messages = [
            *messages,
            {"role": "user", "content": f"Action {action_num} of {action_budget}: call a tool now."},
        ]
        tool_choice = "auto"
        response = await client.chat.completions.create(
            model="policy",
            messages=turn_messages,
            tools=TOOLS,
            tool_choice=tool_choice,
            stream=False,
        )
        turns.append(
            AgentTrajectoryTurn(
                item=item,
                messages=turn_messages,
                response=response,
                tools=TOOLS,
                tool_choice=tool_choice,
            )
        )

        assistant_message = _assistant_message(response)
        tool_result = _execute_tool(assistant_message, board, puzzle, history)
        if tool_result is None:
            logger.warning("Sudoku model returned no executable tool call")
            break
        messages.extend(_tool_messages(assistant_message, tool_result))

        if is_complete(board):
            finish_messages = [
                *messages,
                {"role": "user", "content": "The board is complete. Briefly summarize the outcome without calling a tool."},
            ]
            finish_response = await client.chat.completions.create(
                model="policy",
                messages=finish_messages,
                stream=False,
            )
            turns.append(
                AgentTrajectoryTurn(
                    item=item,
                    messages=finish_messages,
                    response=finish_response,
                )
            )
            break
    return turns


def _assistant_message(response) -> dict:
    message = response.choices[0].message
    return {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": call.type,
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in (message.tool_calls or [])
        ],
    }


def _execute_tool(assistant_message: dict, board, puzzle, history) -> dict | None:
    """Execute a single tool call on the mutable *board* and return the result dict."""

    calls = assistant_message.get("tool_calls") or []
    if len(calls) != 1:
        return None
    call = calls[0]
    name = call.get("function", {}).get("name", "")
    if name not in VALID_TOOL_NAMES:
        return None
    try:
        arguments = json.loads(call["function"].get("arguments") or "{}")
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(arguments, dict):
        return None

    if name == "inspect_candidates":
        row = safe_int(arguments.get("row")) - 1
        col = safe_int(arguments.get("col")) - 1
        if not (0 <= row < 9 and 0 <= col < 9):
            return {"valid": False, "error": "row and col must be in 1..9"}
        if puzzle[row][col] != 0:
            return {"valid": False, "error": f"cell ({row + 1},{col + 1}) is a pre-filled clue"}
        if board[row][col] != 0:
            return {"valid": False, "error": f"cell ({row + 1},{col + 1}) is already filled"}
        return {"valid": True, "candidates": get_candidates(board, row, col)}

    if name == "place_digit":
        row = safe_int(arguments.get("row")) - 1
        col = safe_int(arguments.get("col")) - 1
        digit = safe_int(arguments.get("digit"))
        result = validate_placement(board, puzzle, row, col, digit)
        if result["valid"]:
            board[row][col] = digit
            history.append((row, col, digit))
        return result

    if name == "undo":
        if not history:
            return {"valid": False, "error": "no placements to undo"}
        r, c, d = history.pop()
        board[r][c] = 0
        return {"valid": True, "undid": {"row": r + 1, "col": c + 1, "digit": d}}

    return None


def _tool_messages(assistant_message: dict, tool_result: dict) -> list[dict]:
    call = assistant_message["tool_calls"][0]
    return [
        assistant_message,
        {
            "role": "tool",
            "tool_call_id": call["id"],
            "name": call["function"]["name"],
            "content": json.dumps(tool_result),
        },
    ]