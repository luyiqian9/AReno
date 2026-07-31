"""CPU tests for the Sudoku agentic RL demo (issue #184)."""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "sudoku"


def _load_module(name: str):
    """Load an example module, handling 'game' name collisions via sys.path.

    When loading *run_agent* we also stub ``areno.api.agentic`` so that
    ``AgentTrajectoryTurn`` does not validate Areno response metadata
    (the real class requires ``response_tokens`` / ``response_logprobs``
    which fake test responses do not carry).  This mirrors the pattern
    used by the Codebreaker and Shopping example tests.
    """
    path = EXAMPLE_DIR / f"{name}.py"
    previous_game = sys.modules.pop("game", None)
    previous_agentic = sys.modules.get("areno.api.agentic")
    if name == "run_agent":
        sys.modules["areno.api.agentic"] = SimpleNamespace(
            AgentTrajectory=type("AgentTrajectory", (), {}),
            AgentTrajectoryTurn=lambda **kwargs: SimpleNamespace(**kwargs),
        )
    sys.path.insert(0, str(EXAMPLE_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            f"agentic_sudoku_{name}_for_tests", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXAMPLE_DIR))
        sys.modules.pop("game", None)
        if previous_game is not None:
            sys.modules["game"] = previous_game
        if name == "run_agent":
            sys.modules.pop("areno.api.agentic", None)
            if previous_agentic is not None:
                sys.modules["areno.api.agentic"] = previous_agentic


# -- Board generation ------------------------------------------------------

def test_generate_board_produces_valid_unique_puzzles():
    game = _load_module("game")
    for difficulty in ("easy", "medium", "hard"):
        puzzle, solution = game.generate_board(difficulty, seed=42)
        assert game.is_solved(solution, solution)
        assert game.count_empty(puzzle) > 0
        clues = 81 - game.count_empty(puzzle)
        target = game.DIFFICULTY_CLUES[difficulty]
        assert clues <= target + 5  # allow minimal slack
        # Verify uniqueness
        test = copy.deepcopy(puzzle)
        assert game._count_solutions(test, 2) == 1


def test_difficulty_levels_differ_in_empty_count():
    game = _load_module("game")
    counts = {}
    for difficulty in ("easy", "medium", "hard"):
        puzzle, _ = game.generate_board(difficulty, seed=7)
        counts[difficulty] = game.count_empty(puzzle)
    assert counts["hard"] >= counts["medium"] >= counts["easy"]


# -- Validation -----------------------------------------------------------

def test_validate_placement_rejects_duplicate_in_row():
    game = _load_module("game")
    puzzle, solution = game.generate_board("easy", seed=1)
    board = copy.deepcopy(puzzle)
    empty = [(r, c) for r in range(9) for c in range(9) if puzzle[r][c] == 0]
    row, col = empty[0]
    candidates = game.get_candidates(board, row, col)
    assert candidates
    digit = candidates[0]
    board[row][col] = digit

    # Same digit in same row -> invalid
    for c2 in range(9):
        if c2 != col and puzzle[row][c2] == 0 and board[row][c2] == 0:
            result = game.validate_placement(board, puzzle, row, c2, digit)
            assert not result["valid"]
            break


def test_validate_placement_rejects_illegal_coordinates():
    game = _load_module("game")
    puzzle, _ = game.generate_board("easy", seed=2)
    board = copy.deepcopy(puzzle)
    assert not game.validate_placement(board, puzzle, -1, 0, 5)["valid"]
    assert not game.validate_placement(board, puzzle, 0, 9, 5)["valid"]
    assert not game.validate_placement(board, puzzle, 0, 0, 0)["valid"]
    assert not game.validate_placement(board, puzzle, 0, 0, 10)["valid"]


def test_validate_placement_rejects_prefilled_clues():
    game = _load_module("game")
    puzzle, _ = game.generate_board("easy", seed=3)
    board = copy.deepcopy(puzzle)
    for r in range(9):
        for c in range(9):
            if puzzle[r][c] != 0:
                result = game.validate_placement(board, puzzle, r, c, puzzle[r][c])
                assert not result["valid"]
                assert "clue" in result["error"]
                return


def test_validate_placement_rejects_already_filled():
    game = _load_module("game")
    puzzle, _ = game.generate_board("easy", seed=4)
    board = copy.deepcopy(puzzle)
    empty = [(r, c) for r in range(9) for c in range(9) if puzzle[r][c] == 0]
    r, c = empty[0]
    board[r][c] = 5
    result = game.validate_placement(board, puzzle, r, c, 5)
    assert not result["valid"]
    assert "already" in result["error"]


# -- Tool schema ----------------------------------------------------------

def test_tool_schemas_are_closed_and_bounded():
    game = _load_module("game")
    for tool in game.TOOLS:
        params = tool["function"]["parameters"]
        assert params["additionalProperties"] is False

    inspect = game.TOOL_BY_NAME["inspect_candidates"]
    assert inspect["function"]["parameters"]["required"] == ["row", "col"]

    place = game.TOOL_BY_NAME["place_digit"]
    p = place["function"]["parameters"]
    assert p["required"] == ["row", "col", "digit"]
    assert p["properties"]["digit"]["minimum"] == 1
    assert p["properties"]["digit"]["maximum"] == 9
    assert p["properties"]["row"]["minimum"] == 1
    assert p["properties"]["row"]["maximum"] == 9

    undo = game.TOOL_BY_NAME["undo"]
    assert undo["function"]["parameters"]["required"] == []


# -- Prompt ---------------------------------------------------------------

def test_make_prompt_hides_solution():
    game = _load_module("game")
    puzzle, solution = game.generate_board("easy", seed=5)
    prompt = game.make_prompt({
        "puzzle": puzzle,
        "difficulty": "easy",
        "action_budget": 72,
    })
    assert "." in prompt  # empty markers
    assert "72" in prompt  # action budget
    # Solution digits for empty cells should not appear in the prompt grid
    for r in range(9):
        for c in range(9):
            if puzzle[r][c] == 0:
                # The prompt row should have "." at that position
                row_str = prompt.split("\n")[3 + r]  # skip 3 header lines
                assert "." in row_str


# -- Scoring --------------------------------------------------------------

def test_score_solved_efficiently():
    game = _load_module("game")
    puzzle, solution = game.generate_board("easy", seed=10)
    empty = [(r, c) for r in range(9) for c in range(9) if puzzle[r][c] == 0]
    budget = len(empty) * 2
    actions = [
        {"name": "place_digit", "arguments": {"row": r + 1, "col": c + 1, "digit": solution[r][c]}}
        for r, c in empty
    ]
    score = game.score_episode(puzzle, solution, actions, budget)
    assert score > 0.8


def test_score_partial_progress():
    game = _load_module("game")
    puzzle, solution = game.generate_board("easy", seed=11)
    empty = [(r, c) for r in range(9) for c in range(9) if puzzle[r][c] == 0]
    budget = len(empty) * 2
    half = len(empty) // 2
    actions = [
        {"name": "place_digit", "arguments": {"row": r + 1, "col": c + 1, "digit": solution[r][c]}}
        for r, c in empty[:half]
    ]
    score = game.score_episode(puzzle, solution, actions, budget)
    assert 0 < score < 0.8
    stats = game.replay_episode(puzzle, solution, actions, budget)
    assert stats["correct_fills"] == half
    assert stats["wrong_fills"] == 0
    assert stats["invalid_action_rate"] == 0.0


def test_score_all_invalid_is_negative():
    game = _load_module("game")
    puzzle, solution = game.generate_board("easy", seed=12)
    empty = [(r, c) for r in range(9) for c in range(9) if puzzle[r][c] == 0]
    budget = len(empty) * 2
    # All out-of-range coordinates -- every placement is invalid
    actions = [
        {"name": "place_digit", "arguments": {"row": 99, "col": 99, "digit": 5}}
        for _ in range(3)
    ]
    score = game.score_episode(puzzle, solution, actions, budget)
    assert score == -1.0


def test_replay_undo_at_history_start_is_invalid():
    game = _load_module("game")
    puzzle, solution = game.generate_board("easy", seed=13)
    empty = [(r, c) for r in range(9) for c in range(9) if puzzle[r][c] == 0]
    budget = len(empty) * 2
    actions = [{"name": "undo", "arguments": {}}]
    stats = game.replay_episode(puzzle, solution, actions, budget)
    assert stats["invalid_actions"] == 1
    assert stats["invalid_action_rate"] == 1.0


def test_replay_action_budget_exhaustion():
    game = _load_module("game")
    puzzle, solution = game.generate_board("easy", seed=14)
    empty = [(r, c) for r in range(9) for c in range(9) if puzzle[r][c] == 0]
    budget = 3
    actions = [
        {"name": "place_digit", "arguments": {"row": r + 1, "col": c + 1, "digit": solution[r][c]}}
        for r, c in empty  # far more than budget
    ]
    stats = game.replay_episode(puzzle, solution, actions, budget)
    assert stats["actions_used"] == budget
    assert not stats["solved"]


def test_replay_undo_reverts_placement():
    game = _load_module("game")
    puzzle, solution = game.generate_board("easy", seed=15)
    empty = [(r, c) for r in range(9) for c in range(9) if puzzle[r][c] == 0]
    budget = len(empty) * 3
    r0, c0 = empty[0]
    actions = [
        {"name": "place_digit", "arguments": {"row": r0 + 1, "col": c0 + 1, "digit": solution[r0][c0]}},
        {"name": "undo", "arguments": {}},
    ]
    stats = game.replay_episode(puzzle, solution, actions, budget)
    assert stats["correct_fills"] == 0  # undone
    assert stats["invalid_actions"] == 0


# -- Reward function ------------------------------------------------------

def test_reward_fn_replays_correctly():
    reward = _load_module("reward")
    game = _load_module("game")
    puzzle, solution = game.generate_board("easy", seed=20)
    empty = [(r, c) for r in range(9) for c in range(9) if puzzle[r][c] == 0]
    budget = len(empty) * 2
    tool_calls = [
        {"name": "place_digit", "arguments": json.dumps({"row": r + 1, "col": c + 1, "digit": solution[r][c]})}
        for r, c in empty
    ]
    record = SimpleNamespace(
        source_record={"puzzle": puzzle, "solution": solution, "action_budget": budget},
        tool_calls=tool_calls,
    )
    assert reward.reward_fn(record) > 0.8


def test_reward_fn_handles_dict_arguments():
    reward = _load_module("reward")
    game = _load_module("game")
    puzzle, solution = game.generate_board("easy", seed=21)
    empty = [(r, c) for r in range(9) for c in range(9) if puzzle[r][c] == 0]
    budget = len(empty) * 2
    # arguments as dict (not str)
    tool_calls = [
        {"name": "place_digit", "arguments": {"row": r + 1, "col": c + 1, "digit": solution[r][c]}}
        for r, c in empty
    ]
    record = SimpleNamespace(
        source_record={"puzzle": puzzle, "solution": solution, "action_budget": budget},
        tool_calls=tool_calls,
    )
    assert reward.reward_fn(record) > 0.8


def test_reward_fn_invalid_json_returns_negative():
    reward = _load_module("reward")
    game = _load_module("game")
    puzzle, solution = game.generate_board("easy", seed=22)
    record = SimpleNamespace(
        source_record={"puzzle": puzzle, "solution": solution, "action_budget": 72},
        tool_calls=[{"name": "place_digit", "arguments": "not-json"}],
    )
    assert reward.reward_fn(record) == -1.0


# -- Generator & loader ---------------------------------------------------

def test_generator_is_reproducible():
    gen = _load_module("dataset_generator")
    r1 = gen.generate_records(5, seed=100, difficulty="easy")
    r2 = gen.generate_records(5, seed=100, difficulty="easy")
    assert r1 == r2


def test_generator_action_budget_is_double_empty_cells():
    gen = _load_module("dataset_generator")
    game = _load_module("game")
    records = gen.generate_records(3, seed=50, difficulty="easy")
    for r in records:
        empty = game.count_empty(r["puzzle"])
        assert r["action_budget"] == empty * game.DEFAULT_ACTION_BUDGET_RATIO


def test_loader_normalizes_records_and_adds_prompt():
    loader = _load_module("dataset_loader")
    game = _load_module("game")
    puzzle, solution = game.generate_board("easy", seed=30)
    raw = json.dumps({"puzzle": puzzle, "solution": solution, "difficulty": "easy"})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(raw + "\n")
        tmpfile = f.name
    try:
        def fake_loader(path):
            return [{"puzzle": puzzle, "solution": solution, "difficulty": "easy"}]

        records = loader.load_training_dataset(tmpfile, default_loader=fake_loader)
        assert len(records) == 1
        record = records[0]
        assert "prompt" in record
        assert record["action_budget"] > 0
        assert record["difficulty"] == "easy"
    finally:
        Path(tmpfile).unlink(missing_ok=True)


def test_loader_fallback_when_file_missing():
    loader = _load_module("dataset_loader")

    def fake_loader_ignored(path):
        raise FileNotFoundError("should not be called")

    records = loader.load_training_dataset("/tmp/nonexistent-sudoku-12345.jsonl", default_loader=fake_loader_ignored)
    assert len(records) > 0
    assert all("prompt" in r for r in records)
    assert all("puzzle" in r for r in records)
    assert all("solution" in r for r in records)


# -- Agent episode --------------------------------------------------------

def _fake_tool_response(call_id, name, args):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id=call_id,
                            type="function",
                            function=SimpleNamespace(
                                name=name,
                                arguments=json.dumps(args),
                            ),
                        )
                    ],
                )
            )
        ]
    )


def _fake_text_response(content):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=None)
            )
        ]
    )


def test_agent_episode_places_all_then_finishes():
    run_agent = _load_module("run_agent")
    game = _load_module("game")

    puzzle, solution = game.generate_board("easy", seed=40)
    empty = [(r, c) for r in range(9) for c in range(9) if puzzle[r][c] == 0]

    responses = [
        _fake_tool_response(f"call-{i}", "place_digit", {"row": r + 1, "col": c + 1, "digit": solution[r][c]})
        for i, (r, c) in enumerate(empty)
    ]
    responses.append(_fake_text_response("Solved!"))

    class FakeCompletions:
        def __init__(self, responses):
            self.responses = iter(responses)
            self.messages = []

        async def create(self, **kwargs):
            self.messages.append(kwargs["messages"])
            return next(self.responses)

    item = SimpleNamespace(
        prompt="solve it",
        record={"puzzle": puzzle, "solution": solution, "action_budget": len(empty) * 2},
    )
    completions = FakeCompletions(responses)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    turns = asyncio.run(run_agent._run_episode(item, client))

    assert len(turns) == len(empty) + 1
    # Verify tool messages alternate (from second turn onward the prior
    # assistant+tool pair is at the tail of the copied message list).
    for i in range(1, len(empty)):
        msgs = completions.messages[i]
        assert msgs[-3]["role"] == "assistant"
        assert msgs[-2]["role"] == "tool"
        assert msgs[-2]["name"] == "place_digit"


def test_agent_episode_undo_lifecycle():
    run_agent = _load_module("run_agent")
    game = _load_module("game")

    puzzle, solution = game.generate_board("easy", seed=41)
    empty = [(r, c) for r in range(9) for c in range(9) if puzzle[r][c] == 0]
    r0, c0 = empty[0]

    responses = [
        _fake_tool_response("call-0", "place_digit", {"row": r0 + 1, "col": c0 + 1, "digit": solution[r0][c0]}),
        _fake_tool_response("call-1", "undo", {}),
        _fake_tool_response("call-2", "place_digit", {"row": r0 + 1, "col": c0 + 1, "digit": solution[r0][c0]}),
    ]
    for i, (r, c) in enumerate(empty[1:], start=3):
        responses.append(_fake_tool_response(f"call-{i}", "place_digit", {"row": r + 1, "col": c + 1, "digit": solution[r][c]}))
    responses.append(_fake_text_response("Done"))

    class FakeCompletions:
        def __init__(self, responses):
            self.responses = iter(responses)

        async def create(self, **kwargs):
            return next(self.responses)

    item = SimpleNamespace(
        prompt="solve it",
        record={"puzzle": puzzle, "solution": solution, "action_budget": len(empty) * 3},
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(responses)))
    turns = asyncio.run(run_agent._run_episode(item, client))

    # place + undo + re-place + remaining(n-1) + finish = n + 3
    assert len(turns) == len(empty) + 3


def test_agent_episode_stops_on_no_tool_call():
    run_agent = _load_module("run_agent")
    game = _load_module("game")

    puzzle, solution = game.generate_board("easy", seed=42)

    class FakeCompletions:
        async def create(self, **kwargs):
            return _fake_text_response("I give up.")

    item = SimpleNamespace(
        prompt="solve it",
        record={"puzzle": puzzle, "solution": solution, "action_budget": 100},
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    turns = asyncio.run(run_agent._run_episode(item, client))
    assert len(turns) == 1