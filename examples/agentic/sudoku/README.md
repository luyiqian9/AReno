# Agentic Sudoku

Sudoku is a multi-turn constraint-satisfaction game. The policy receives a
partially filled 9x9 grid with a unique solution and must fill every empty cell
using three tools: `inspect_candidates`, `place_digit`, and `undo`. Each tool
call is validated against Sudoku rules without revealing the solution.

## Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `inspect_candidates` | `row` (1-9), `col` (1-9) | Returns the digits that are legally placeable in the given cell. |
| `place_digit` | `row` (1-9), `col` (1-9), `digit` (1-9) | Places a digit in an empty cell. Validated against row, column, and box constraints. |
| `undo` | (none) | Reverts the most recent `place_digit` action. |

The solution is never exposed in prompts or tool results. Invalid placements
return an error but do not end the episode; the action budget is still consumed.

## Difficulty levels

| Difficulty | Clues | Empty cells | Default action budget |
|-----------|-------|-------------|-----------------------|
| easy | 45 | 36 | 72 |
| medium | 34 | 47 | 94 |
| hard | 28 | 53 | 106 |

The action budget is set to 2x the number of empty cells, giving the agent room
to make and correct mistakes.

## Generate data

```bash
python examples/agentic/sudoku/dataset_generator.py \
  --output /tmp/sudoku.jsonl --count 256 --seed 2026 --difficulty medium
```

## Train

```bash
areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --dataset-path /tmp/sudoku.jsonl \
  --dataset-loader-fn examples/agentic/sudoku/dataset_loader.py \
  --reward-fn-path examples/agentic/sudoku/reward.py \
  --agent-fn examples/agentic/sudoku/run_agent.py \
  --algo gspo --tp-size 1 --world-size 2 \
  --batch-size 1 --n-samples 2 --max-new-tokens 128
```

## Reward design

The reward function replays the tool-call trajectory on a copy of the puzzle:

- **Fully solved**: `0.8 + 0.2 * efficiency` where efficiency reflects unused
  action budget.
- **Partial progress** (net correct fills > 0): `0.1 * net / total_empty`.
- **No progress or net negative**: `-1.0` (all invalid) or `-0.5` (tried but
  more wrong than right).

## Web UI

After serving a trained (or base) model, launch the web UI to watch the agent
solve puzzles step by step:

```bash
areno serve --model-path /path/to/checkpoint --port 8001
python examples/agentic/sudoku/web_ui.py --base-url http://127.0.0.1:8001/v1
```

Open `http://127.0.0.1:8769` in a browser. The 9x9 grid shows the puzzle with
clues in gray and agent placements in blue. A **Step** button runs one tool call
at a time; **Auto Run** executes continuously until the episode ends. The
action log on the right records each tool call and its result. When the episode
terminates, correct fills turn green and wrong fills turn red.

Controls:
- **Difficulty** — easy / medium / hard
- **New Puzzle** — generate a fresh puzzle
- **Step** — execute one agent action
- **Auto Run** — run actions automatically (toggle to stop)

## Testing

```bash
pytest tests/ -k sudoku
```