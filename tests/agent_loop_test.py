"""Tool-use loop test - proves the model-driven path without spending a token.

Fakes gateway.chat_stream (the one seam every provider shares) and checks that Cortex
  1. offers tool schemas to the model,
  2. replays the assistant tool_call turn and a matching tool message,
  3. executes the real tool and feeds the result back,
  4. withholds tools on the final round so the model must answer,
  5. survives a failing tool and a hallucinated tool name.

Usage:  python tests/agent_loop_test.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TMP = Path(tempfile.mkdtemp(prefix="cortex_loop_"))
os.environ["CORTEX_DATA_DIR"] = str(TMP)
os.environ["CORTEX_FREE_MODEL"] = "0"
os.environ["CORTEX_LOCAL_MODEL"] = "0"
# hermetic: a developer's real .env must not leak into these assertions
for _var in ("AI_GATEWAY_API_KEY", "VERCEL_AI_GATEWAY_KEY", "VERCEL_TOKEN", "OPENROUTER_API_KEY",
             "OPENAI_API_KEY", "GROQ_API_KEY", "DEEPSEEK_API_KEY", "TOGETHER_API_KEY",
             "GEMINI_API_KEY", "GOOGLE_API_KEY", "CORTEX_API_KEY", "CORTEX_PROVIDER",
             "CORTEX_BASE_URL"):
    os.environ[_var] = ""

os.environ["CORTEX_DB"] = str(TMP / "test.db")

from app import agent, db, gateway, tools  # noqa: E402

db.init()

PASS, FAIL = 0, 0
GREEN, RED, OFF = "\033[32m", "\033[31m", "\033[0m"

SOURCES = [{"doc_name": "Handbook.md", "text": "Annual leave is 25 days.", "page": None,
            "heading": "Leave", "score": 0.9, "ext": ".md"}]


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  {GREEN}✓{OFF} {name}")
    else:
        FAIL += 1
        print(f"  {RED}✗{OFF} {name} {extra}")


def tool_call(bid, name, args):
    return {"type": "tool_call", "id": bid, "name": name, "arguments": args}


def say(text):
    return {"type": "delta", "text": text}


USAGE = {"type": "usage", "input_tokens": 100, "output_tokens": 25}


class FakeGateway:
    """Stands in for gateway.chat_stream; records exactly what the agent sends."""

    def __init__(self, rounds):
        self.rounds = rounds
        self.calls = []

    async def chat_stream(self, model, messages, tools=None, max_tokens=None, temperature=None,
                          first_token=None):
        self.calls.append({"model": model, "messages": [dict(m) for m in messages],
                           "tools": tools, "max_tokens": max_tokens})
        script = self.rounds[len(self.calls) - 1]
        for ev in script:
            yield dict(ev)
        yield dict(USAGE)


def run_loop(rounds, question="What is 18% of 4750?", specs=None):
    fake = FakeGateway(rounds)
    gateway.chat_stream = fake.chat_stream
    gateway.configured = lambda: True
    specs = specs if specs is not None else tools.available_specs()

    async def collect():
        return [ev async for ev in agent.stream(
            question, SOURCES, [], conv_id="test-conv", model="openai/gpt-5.3-codex",
            specs=specs)]

    return asyncio.run(collect()), fake.calls


print("\n\033[1msingle tool round\033[0m")
events, calls = run_loop([
    [tool_call("tu_1", "calculator", {"expression": "18/100*4750"})],
    [say("18% of 4 750 is "), say("**855**.")],
])
kinds = [e["type"] for e in events]
check("model was given the usable tool schemas",
      len(calls[0]["tools"]) == len(tools.available_specs()) >= 5,
      str(len(calls[0].get("tools") or [])))
check("first call allows tools", calls[0]["tools"] is not None)
check("tool_start surfaced to the UI",
      any(e["type"] == "tool_start" and e["name"] == "calculator" for e in events))
check("real result came back, not a stub",
      any(e["type"] == "tool_result" and e["result"].get("result") == 855 for e in events))
check("event stream announces the route", "route" in kinds)
turn = calls[1]["messages"][-2]
check("assistant tool_call turn replayed verbatim",
      turn["role"] == "assistant" and turn["tool_calls"][0]["id"] == "tu_1"
      and turn["tool_calls"][0]["function"]["name"] == "calculator"
      and json.loads(turn["tool_calls"][0]["function"]["arguments"]) == {"expression": "18/100*4750"})
replay = calls[1]["messages"][-1]
check("tool message references the same id",
      replay["role"] == "tool" and replay["tool_call_id"] == "tu_1")
check("tool result carries the computed value to the model", "855" in replay["content"])
check("result is not flagged as an error", "error" not in replay["content"].lower())
answer = "".join(e.get("text", "") for e in events if e["type"] == "delta")
check("final text reaches the user", "855" in answer, answer)
usage = [e for e in events if e["type"] == "usage"]
check("usage sums both rounds",
      usage and usage[-1]["input_tokens"] == 200 and usage[-1]["tool_calls"] == 1,
      str(usage[-1] if usage else None))
check("usage names the model that answered", usage and usage[-1].get("model") == "openai/gpt-5.3-codex")

print("\n\033[1mfailing tool\033[0m")
events, calls = run_loop([
    [tool_call("tu_2", "fetch_url", {"url": "http://localhost/admin"})],
    [say("I could not fetch that: the host is blocked.")],
])
tr = [e for e in events if e["type"] == "tool_result"]
check("failure is reported as a tool result", bool(tr) and tr[0]["ok"] is False)
check("model sees the error", "error" in calls[1]["messages"][-1]["content"].lower())
check("loop keeps going after a failure",
      any(e["type"] == "delta" and "blocked" in e.get("text", "") for e in events))

print("\n\033[1mround cap\033[0m")
original = agent.MAX_ROUNDS
agent.MAX_ROUNDS = 1
events, calls = run_loop([
    [tool_call("tu_3", "current_datetime", {})],
    [tool_call("tu_4", "current_datetime", {})],   # the model still wants tools on the last round
])
agent.MAX_ROUNDS = original
check("last round withholds the tools", calls[-1]["tools"] is None)
check("loop terminates instead of spinning", len(calls) == 2, str(len(calls)))
check("it still returns usage", any(e["type"] == "usage" for e in events))

print("\n\033[1munknown tool name\033[0m")
events, calls = run_loop([
    [tool_call("tu_5", "wire_money", {"to": "attacker"})],
    [say("That tool does not exist.")],
])
tr = [e for e in events if e["type"] == "tool_result"]
check("hallucinated tool returns an error result",
      bool(tr) and tr[0]["ok"] is False and "not enabled" in json.dumps(tr[0]).lower(),
      json.dumps(tr[0]) if tr else "")
check("model is told, not crashed", "error" in calls[1]["messages"][-1]["content"].lower())

print("\n\033[1mtool call written as text (what Nemotron does)\033[0m")
# The model prints the call into the answer instead of using the tool channel.
# It must still run, and none of the markup may reach the user.
events, calls = run_loop([
    [say("Let me check that site.\n\n<tool_call>\n<function=calculator>\n"),
     say("<parameter=expression>"), say("18/100*4750</parameter>\n</function>\n</tool_call>\n")],
    [say("18% of 4 750 is **855**.")],
])
answer = "".join(e.get("text", "") for e in events if e["type"] == "delta")
check("the text-emitted call actually ran",
      any(e["type"] == "tool_result" and e["result"].get("result") == 855 for e in events),
      str(events)[:300])
check("no <tool_call> markup reaches the user",
      "<tool_call>" not in answer and "<parameter" not in answer, answer)
check("the prose around the call is kept", "Let me check that site" in answer, answer)
check("the final answer still arrives", "855" in answer, answer)
replay = calls[1]["messages"][-2]
check("the recovered call is replayed as a real tool_call",
      bool(replay.get("tool_calls")) and replay["tool_calls"][0]["function"]["name"] == "calculator",
      str(replay)[:250])
check("the markup is not replayed as assistant text",
      "<tool_call>" not in (replay.get("content") or ""), str(replay.get("content"))[:120])
check("the tool message is attached", calls[1]["messages"][-1]["role"] == "tool")

print("\n\033[1mmarkup-only reply still produces an answer\033[0m")
events, _ = run_loop([
    [say("<tool_call><function=calculator><parameter=expression>2+2</parameter>"
         "</function></tool_call>")],
    [say("")],
])
answer = "".join(e.get("text", "") for e in events if e["type"] == "delta")
final = [e["text"] for e in events if e["type"] == "text_replace"]
answer = final[-1] if final else answer      # the UI swaps in text_replace verbatim
check("the user is never shown raw markup", "<tool_call>" not in answer, answer)
check("a usable answer comes from the tool result", "4" in answer, repr(answer))

print(f"\n\033[1m{PASS} passed, {FAIL} failed{OFF}\n")
sys.exit(1 if FAIL else 0)
