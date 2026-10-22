"""Text-emitted tool calls: the parser and the streaming filter.

The case that prompted this: Nemotron on OpenRouter printed

    <tool_call><function=fetch_url><parameter=url>…</parameter></function></tool_call>

into the answer instead of using the tool channel. The user saw raw markup and
the tool never ran. These checks pin the recovery down, including the nasty
chunk boundary cases (a tag split across two stream deltas) and the "it wasn't a
call after all" cases where text must not be eaten.

Usage:  python tests/tooltext_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.tooltext import ToolTextFilter, extract, parse_block  # noqa: E402

PASS = FAIL = 0
GREEN, RED, OFF = "\033[32m", "\033[31m", "\033[0m"


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  {GREEN}✓{OFF} {name}")
    else:
        FAIL += 1
        print(f"  {RED}✗{OFF} {name} {extra}")


TOOLS = {"web_search", "fetch_url", "calculator", "write_file", "run_code", "wikipedia"}


def run(text: str, step: int | None = None) -> tuple[str, list[dict]]:
    """Feed `text` through the filter, one character at a time when step is set."""
    filt = ToolTextFilter(TOOLS)
    chunks = [text] if not step else [text[i:i + step] for i in range(0, len(text), step)]
    out, calls = [], []
    for chunk in chunks:
        for kind, value in filt.feed(chunk):
            if kind == "text":
                out.append(value)
            else:
                calls.append(value)
    for kind, value in filt.flush():
        if kind == "text":
            out.append(value)
        else:
            calls.append(value)
    return "".join(out), calls


SCREENSHOT = """Let me check that site for you.

<tool_call>
<function=fetch_url>
<parameter=url>
http://jamiatkzn.org
</parameter>
</function>
</tool_call>
"""

print("\n\033[1mthe markup that leaked into the answer\033[0m")
for step, label in ((None, "in one chunk"), (1, "one character at a time"), (7, "seven-char chunks")):
    text, calls = run(SCREENSHOT, step)
    check(f"markup removed ({label})", "<tool_call>" not in text and "parameter" not in text, text)
    check(f"prose before it survives ({label})", text.strip().startswith("Let me check that site"), text)
    check(f"call recovered ({label})",
          calls == [{"name": "fetch_url", "arguments": {"url": "http://jamiatkzn.org"}}], str(calls))

print("\n\033[1mparameter typing\033[0m")
call = parse_block("<function=web_search><parameter=query>fatwa</parameter>"
                   "<parameter=max_results>5</parameter></function>")
check("integers become integers", call["arguments"]["max_results"] == 5 and
      isinstance(call["arguments"]["max_results"], int), str(call))
call = parse_block('<function=write_file><parameter=path>a.py</parameter>'
                   '<parameter=content>print("hi")\n\nx = 1</parameter></function>')
check("newlines inside a value are kept", "x = 1" in call["arguments"]["content"], str(call))
check("file content is not mangled by tag stripping", 'print("hi")' in call["arguments"]["content"])
call = parse_block('<function=calculator><parameter=expression>{"a": 2}</parameter></function>')
check("json-looking values stay json", call["arguments"]["expression"] == {"a": 2}, str(call))
call = parse_block('{"name": "calculator", "arguments": {"expression": "1+1"}}')
check("a JSON call is understood too", call == {"name": "calculator",
                                                "arguments": {"expression": "1+1"}}, str(call))

print("\n\033[1mmultiple and interleaved calls\033[0m")
raw = ('Searching.<tool_call><function=web_search><parameter=query>a</parameter></function></tool_call>'
       ' then <tool_call><function=web_search><parameter=query>b</parameter></function></tool_call> done')
text, calls = run(raw)
check("both calls found", len(calls) == 2, str(calls))
check("both queries intact", [c["arguments"]["query"] for c in calls] == ["a", "b"], str(calls))
check("surrounding words kept", text == "Searching. then  done", repr(text))

print("\n\033[1mthings that must NOT be eaten\033[0m")
raw = "Use the tag <tool_call> in your prompt to trigger a call. Nothing else."
text, calls = run(raw, 3)
check("a stray opening tag stays visible", raw == text, repr(text))
check("no call invented from prose", calls == [], str(calls))

raw = "done<tool_call>not really a call</tool_call>ok"
text, calls = run(raw, 2)
check("a block that is not a call is passed through", "not really a call" in text, repr(text))
check("still no call", calls == [], str(calls))

raw = "Here is the format: <function=name> with <parameter=key>value</parameter>. That is all."
text, calls = run(raw)
check("a mentioned format without a real call is not executed", calls == [], str(calls))
check("and the explanation survives", "Here is the format" in text, repr(text))

print("\n\033[1mtruncated replies\033[0m")
raw = "<tool_call>\n<function=web_search>\n<parameter=query>half a que"   # cut off mid-value
text, calls = run(raw, 4)
check("an unfinished call is not shown as prose", "<parameter" not in text, repr(text))

print("\n\033[1mthe non-streaming helper agrees\033[0m")
text, calls = extract(SCREENSHOT)
check("extract() strips and recovers", "<tool_call>" not in text and len(calls) == 1, str(calls))
text, calls = extract("plain answer, no tools")
check("plain text passes through untouched", text == "plain answer, no tools" and calls == [])

print("\n\033[1morder of text around a call\033[0m")
text, calls = run("before <tool_call><function=calculator><parameter=expression>2*3</parameter>"
                  "</function></tool_call> after", 2)
check("text on both sides keeps its order", text == "before  after", repr(text))
check("call still parsed", calls == [{"name": "calculator", "arguments": {"expression": "2*3"}}])

print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
