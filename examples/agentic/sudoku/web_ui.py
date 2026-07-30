"""Cartoon web UI server for the Sudoku agentic example.

Run from the repository root after starting an inference server:

    areno serve --model-path /path/to/checkpoint --port 8001
    python examples/agentic/sudoku/web_ui.py --base-url http://127.0.0.1:8001/v1
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8769

SYSTEM_PROMPT = (
    "You are a methodical Sudoku solver. On every turn call exactly one tool: "
    "inspect_candidates to check legal digits, place_digit to fill a cell, or undo "
    "to revert a mistake. Use the clues and prior placements to deduce each digit. "
    "When the board is complete, summarize the outcome without calling a tool."
)


# -- Server state -----------------------------------------------------------

class SudokuServer(ThreadingHTTPServer):
    """Small stateful HTTP server for one Sudoku episode."""

    def __init__(self, server_address, request_handler, *, seed: int, difficulty: str, base_url: str, api_key: str, model: str):
        super().__init__(server_address, request_handler)
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self._reset(seed, difficulty)

    def _reset(self, seed: int, difficulty: str) -> None:
        self.puzzle, self.solution = game.generate_board(difficulty, seed=seed)
        self.board = copy.deepcopy(self.puzzle)
        self.history: list[tuple[int, int, int]] = []
        self.messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": game.make_prompt({
                "puzzle": self.puzzle,
                "difficulty": difficulty,
                "action_budget": game.count_empty(self.puzzle) * game.DEFAULT_ACTION_BUDGET_RATIO,
            })},
        ]
        self.action_budget = game.count_empty(self.puzzle) * game.DEFAULT_ACTION_BUDGET_RATIO
        self.action_num = 0
        self.events: list[str] = [f"Puzzle generated (difficulty={difficulty}, empty={game.count_empty(self.puzzle)}, budget={self.action_budget})"]
        self.terminal = False
        self.difficulty = difficulty

    def reset(self, seed: int, difficulty: str) -> None:
        self._reset(seed, difficulty)


# -- Request handling -------------------------------------------------------

class SudokuHandler(BaseHTTPRequestHandler):
    server: SudokuServer

    def do_GET(self) -> None:
        path = _route_path(self.path)
        if path == "/" or path == "":
            self._send_html(INDEX_HTML)
        elif path == "/api/state":
            self._send_json(_payload(self.server))
        else:
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = _route_path(self.path)
        body = self._read_json()
        if path == "/api/new":
            difficulty = body.get("difficulty", self.server.difficulty)
            seed = int(body.get("seed", 0)) or None
            if seed is None:
                import random
                seed = random.randint(1, 999999)
            self.server.reset(seed, difficulty)
            self._send_json(_payload(self.server))
        elif path == "/api/step":
            result = _agent_step(self.server)
            self._send_json(result)
        else:
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("sudoku-web: " + fmt % args + "\n")

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_html(self, html: str) -> None:
        encoded = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


# -- Helpers ----------------------------------------------------------------

def _route_path(raw_path: str) -> str:
    parsed = raw_path.split("?", 1)[0]
    return parsed if parsed.startswith("/") else "/" + parsed


def _payload(server: SudokuServer) -> dict[str, Any]:
    correct, wrong = 0, 0
    for r in range(9):
        for c in range(9):
            if server.puzzle[r][c] == 0 and server.board[r][c] != 0:
                if server.board[r][c] == server.solution[r][c]:
                    correct += 1
                else:
                    wrong += 1
    return {
        "board": server.board,
        "puzzle": server.puzzle,
        "solution": server.solution if server.terminal else None,
        "difficulty": server.difficulty,
        "action_num": server.action_num,
        "action_budget": server.action_budget,
        "events": server.events[-12:],
        "terminal": server.terminal,
        "solved": game.is_solved(server.board, server.solution) if server.terminal else False,
        "correct_fills": correct,
        "wrong_fills": wrong,
        "total_empty": game.count_empty(server.puzzle),
    }


def _agent_step(server: SudokuServer) -> dict[str, Any]:
    if server.terminal:
        return _payload(server)

    server.action_num += 1
    turn_messages = [
        *server.messages,
        {"role": "user", "content": f"Action {server.action_num} of {server.action_budget}: call a tool now."},
    ]

    try:
        response = _llm_chat(server, turn_messages)
    except Exception as exc:
        server.events.append(f"Error: {exc}")
        server.terminal = True
        return _payload(server)

    assistant_message = _extract_assistant(response)
    tool_result = _execute_tool(assistant_message, server)

    if tool_result is None:
        server.events.append("Model returned no tool call — stopping.")
        server.terminal = True
        return _payload(server)

    server.messages.extend(_tool_messages(assistant_message, tool_result))
    tool_name = assistant_message["tool_calls"][0]["function"]["name"]
    server.events.append(_format_event(tool_name, tool_result, server.action_num))

    if game.is_complete(server.board):
        server.terminal = True
        solved = game.is_solved(server.board, server.solution)
        server.events.append(f"Board complete — {'SOLVED' if solved else 'WRONG'}")
    elif server.action_num >= server.action_budget:
        server.terminal = True
        server.events.append("Action budget exhausted.")

    return _payload(server)


def _llm_chat(server: SudokuServer, messages: list[dict]) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Web UI requires `openai`. Install it with `pip install openai`.") from exc
    client = OpenAI(base_url=server.base_url, api_key=server.api_key, max_retries=0)
    return client.chat.completions.create(
        model=server.model,
        messages=messages,
        tools=game.TOOLS,
        tool_choice="auto",
        stream=False,
    )


def _extract_assistant(response: Any) -> dict:
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


def _execute_tool(assistant_message: dict, server: SudokuServer) -> dict | None:
    calls = assistant_message.get("tool_calls") or []
    if len(calls) != 1:
        return None
    call = calls[0]
    name = call.get("function", {}).get("name", "")
    if name not in {"inspect_candidates", "place_digit", "undo"}:
        return None
    try:
        arguments = json.loads(call["function"].get("arguments") or "{}")
    except (json.JSONDecodeError, TypeError):
        return None

    if name == "inspect_candidates":
        row = int(arguments.get("row", 0)) - 1
        col = int(arguments.get("col", 0)) - 1
        if not (0 <= row < 9 and 0 <= col < 9):
            return {"valid": False, "error": "row and col must be 1..9"}
        if server.puzzle[row][col] != 0:
            return {"valid": False, "error": f"({row+1},{col+1}) is a pre-filled clue"}
        if server.board[row][col] != 0:
            return {"valid": False, "error": f"({row+1},{col+1}) is already filled"}
        return {"valid": True, "candidates": game.get_candidates(server.board, row, col)}

    if name == "place_digit":
        row = int(arguments.get("row", 0)) - 1
        col = int(arguments.get("col", 0)) - 1
        digit = int(arguments.get("digit", 0))
        result = game.validate_placement(server.board, server.puzzle, row, col, digit)
        if result["valid"]:
            server.board[row][col] = digit
            server.history.append((row, col, digit))
        return result

    if name == "undo":
        if not server.history:
            return {"valid": False, "error": "no placements to undo"}
        r, c, d = server.history.pop()
        server.board[r][c] = 0
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


def _format_event(name: str, result: dict, step: int) -> str:
    if name == "inspect_candidates":
        if result.get("valid"):
            cands = result.get("candidates", [])
            return f"Step {step}: inspect → candidates {cands}"
        return f"Step {step}: inspect → {result.get('error', 'invalid')}"
    if name == "place_digit":
        if result.get("valid"):
            return f"Step {step}: place → accepted"
        return f"Step {step}: place → rejected ({result.get('error', 'invalid')})"
    if name == "undo":
        if result.get("valid"):
            undid = result.get("undid", {})
            return f"Step {step}: undo → reverted ({undid.get('row')},{undid.get('col')})"
        return f"Step {step}: undo → {result.get('error', 'nothing to undo')}"
    return f"Step {step}: {name}"


# -- HTML -------------------------------------------------------------------

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sudoku Agent</title>
<style>
:root{font-family:Inter,ui-rounded,system-ui,sans-serif;color:#24313a;background:#e8f0fe}
body{margin:0;min-height:100vh;background:linear-gradient(135deg,#e8f0fe,#c3d9f5 58%,#9ec5e8);display:grid;place-items:center}
.app{width:min(1000px,94vw);display:grid;grid-template-columns:minmax(420px,520px) 1fr;gap:22px;align-items:start}
.panel{background:#fcfdff;border:4px solid #27313a;border-radius:24px;box-shadow:8px 8px 0 #27313a;padding:18px}
h1{font-size:36px;line-height:1;margin:0 0 6px;color:#1a6fd9;text-shadow:2px 2px 0 #b3d4f5}
.subtitle{font-weight:900;color:#3d5d8d;margin-bottom:14px}
.grid{display:grid;grid-template-columns:repeat(9,1fr);gap:2px;background:#27313a;border:5px solid #27313a;border-radius:16px;padding:5px;box-shadow:inset 0 4px 0 rgba(255,255,255,.18),8px 10px 0 rgba(39,49,58,.22)}
.cell{aspect-ratio:1;background:#fff7df;border:2px solid #6a7a8a;border-radius:6px;display:grid;place-items:center;font-size:clamp(18px,3.8vw,32px);font-weight:900;transition:.15s}
.cell.clue{background:#d4e0ec;color:#27313a}
.cell.filled{background:#fff;color:#1a6fd9;animation:pop .22s ease}
.cell.wrong{background:#ffb3b3;color:#c00}
.cell.correct{background:#b3ffb3;color:#070}
.cell.invalid-flash{background:#ff6b6b;animation:flash .4s ease}
.cell.inspect-hl{background:#ffffb3;border-color:#e0a800}
@keyframes pop{0%{transform:scale(.5)}70%{transform:scale(1.15)}100%{transform:scale(1)}}
@keyframes flash{0%{background:#ff6b6b}100%{background:#ffb3b3}}
.thick-r{margin-right:5px}
.thick-b{margin-bottom:5px}
.stats,.controls{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}
.pill{background:#fff;border:3px solid #27313a;border-radius:999px;padding:6px 14px;font-weight:900}
button{border:3px solid #27313a;border-radius:14px;background:#ffd166;box-shadow:4px 4px 0 #27313a;color:#27313a;font-weight:900;padding:10px 14px;cursor:pointer}
button:hover{transform:translateY(-1px)}button:disabled{filter:grayscale(.75);opacity:.55;cursor:not-allowed}
select{border:3px solid #27313a;border-radius:14px;background:#fff;padding:9px;font-weight:900;color:#27313a}
.events{display:grid;gap:6px;margin-top:8px;max-height:340px;overflow-y:auto}
.event{background:#fff;border:3px solid #27313a;border-radius:12px;padding:8px 10px;font-weight:800;font-size:14px}
.event.err{background:#ffe0e0}.event.ok{background:#e0ffe0}.event.undo{background:#fff4d0}
.thinking{display:none;margin:10px 0;padding:8px 12px;border:3px solid #27313a;border-radius:14px;background:#dff6ff;font-weight:900}
.thinking.on{display:block}
.dots::after{content:"";animation:dots 1s steps(4,end) infinite}@keyframes dots{0%{content:""}25%{content:"."}50%{content:".."}75%{content:"..."}100%{content:""}}
@media(max-width:760px){.app{grid-template-columns:1fr}}
</style>
</head>
<body>
<main class="app">
  <section class="panel">
    <h1>Sudoku Agent</h1>
    <div class="subtitle">Watch an LLM solve Sudoku with tools.</div>
    <div id="grid" class="grid" aria-label="Sudoku board"></div>
    <div class="stats">
      <span class="pill" id="stepPill">Step 0 / 0</span>
      <span class="pill" id="fillsPill"></span>
      <span class="pill" id="statusPill"></span>
    </div>
    <div id="thinking" class="thinking">Agent is thinking<span class="dots"></span></div>
    <div class="controls">
      <select id="diff"><option value="easy">Easy</option><option value="medium" selected>Medium</option><option value="hard">Hard</option></select>
      <button id="newBtn">New Puzzle</button>
      <button id="stepBtn">Step</button>
      <button id="runBtn">Auto Run</button>
    </div>
  </section>
  <aside class="panel">
    <h1 style="font-size:24px;color:#1a6fd9">Action Log</h1>
    <div id="events" class="events"></div>
  </aside>
</main>
<script>
let state=null, autoTimer=null, running=false;
async function req(path,body){
  if(running) return;
  running=true;
  const opts=body?{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}:{};
  const res=await fetch(path,opts);
  state=await res.json();
  render();
  running=false;
}
async function step(){
  if(running||state.terminal) return;
  document.getElementById("thinking").classList.add("on");
  running=true;
  const res=await fetch("/api/step",{method:"POST",headers:{"Content-Type":"application/json"},body:"{}"});
  state=await res.json();
  running=false;
  document.getElementById("thinking").classList.remove("on");
  render();
}
async function autoRun(){
  if(autoTimer){clearInterval(autoTimer);autoTimer=null;document.getElementById("runBtn").textContent="Auto Run";return;}
  document.getElementById("runBtn").textContent="Stop";
  autoTimer=setInterval(async()=>{
    if(state.terminal){clearInterval(autoTimer);autoTimer=null;document.getElementById("runBtn").textContent="Auto Run";return;}
    await step();
  },800);
}
function render(){
  const grid=document.getElementById("grid");
  grid.innerHTML="";
  for(let r=0;r<9;r++){
    for(let c=0;c<9;c++){
      const pv=state.puzzle[r][c], bv=state.board[r][c];
      const div=document.createElement("div");
      let cls="cell";
      let txt="";
      if(pv!==0){cls+=" clue";txt=pv;}
      else if(bv!==0){
        cls+=" filled";
        txt=bv;
        if(state.terminal&&state.solution){cls+=(bv===state.solution[r][c])?" correct":" wrong";}
      }
      if(c===2||c===5)cls+=" thick-r";
      if(r===2||r===5)cls+=" thick-b";
      div.className=cls;
      div.textContent=txt;
      grid.appendChild(div);
    }
  }
  document.getElementById("stepPill").textContent=`Step ${state.action_num} / ${state.action_budget}`;
  let fillTxt="";
  if(state.correct_fills!==undefined)fillTxt=`Correct ${state.correct_fills}`;
  if(state.wrong_fills>0)fillTxt+=` Wrong ${state.wrong_fills}`;
  if(state.total_empty)fillTxt+=` / ${state.total_empty}`;
  document.getElementById("fillsPill").textContent=fillTxt;
  let status="Solving";
  if(state.terminal){status=state.solved?"SOLVED":"Failed";}
  document.getElementById("statusPill").textContent=status;
  const ev=document.getElementById("events");
  ev.innerHTML=state.events.map(e=>{
    let c="event";
    if(e.includes("rejected")||e.includes("Error")||e.includes("invalid"))c+=" err";
    else if(e.includes("SOLVED")||e.includes("accepted"))c+=" ok";
    else if(e.includes("undo"))c+=" undo";
    return `<div class="${c}">${esc(e)}</div>`;
  }).join("");
  ev.scrollTop=ev.scrollHeight;
  document.getElementById("stepBtn").disabled=state.terminal;
  document.getElementById("runBtn").disabled=false;
  document.getElementById("newBtn").disabled=false;
}
function esc(t){return t.replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#039;"}[c]))}
document.getElementById("newBtn").onclick=()=>{if(autoTimer){clearInterval(autoTimer);autoTimer=null;document.getElementById("runBtn").textContent="Auto Run";}req("/api/new",{difficulty:document.getElementById("diff").value});};
document.getElementById("stepBtn").onclick=step;
document.getElementById("runBtn").onclick=autoRun;
fetch("/api/state").then(r=>r.json()).then(s=>{state=s;render();}).catch(()=>{});
</script>
</body>
</html>
"""


# -- Entry point ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--difficulty", choices=list(game.DIFFICULTY_CLUES), default=game.DEFAULT_DIFFICULTY)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--api-key", default="token")
    parser.add_argument("--model", default="policy")
    args = parser.parse_args()

    server = SudokuServer(
        (args.host, args.port),
        SudokuHandler,
        seed=args.seed,
        difficulty=args.difficulty,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
    )
    print(f"Sudoku web UI on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()