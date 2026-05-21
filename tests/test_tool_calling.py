"""Tool-calling tests for ``ChatCodexPlus`` + the protocol layer.

Covers:

* Tool definitions get serialized into the Codex ``tools`` field with
  the correct flat shape (``type: function`` + name/description/parameters
  at top level, NOT nested under ``function``).
* AIMessage with ``tool_calls`` round-trips into multiple input
  entries (assistant message + function_call items).
* ToolMessage maps to ``function_call_output``.
* SSE function_call events accumulate into ``AIMessage.tool_calls``.
* Streaming yields ``tool_call_chunks`` for function-call deltas.
* The ``bind_tools`` helper accepts dict, Pydantic, and BaseTool forms.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from langchain_codex_plus import ChatCodexPlus
from langchain_codex_plus.codex_protocol import (
    CodexToolCall,
    build_request_body,
    consume_events,
    parse_sse_stream,
)

# Shared helpers come from ``tests/conftest.py`` (auto-discovered by
# pytest). ``auth_file`` is a pytest fixture; tests pull it in by
# parameter name without an explicit import here.
from tests.conftest import _CaptureTransport, _make_llm, _sse_bytes

# ─── Tool serialization (request body) ─────────────────────────────────


def test_build_body_includes_tools_when_provided():
    body = build_request_body(
        [HumanMessage("hi")],
        model="gpt-5.4",
        tools=[{
            "type": "function",
            "name": "get_weather",
            "description": "Look up the weather.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }],
        tool_choice="auto",
    )
    assert body["tools"] == [{
        "type": "function",
        "name": "get_weather",
        "description": "Look up the weather.",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    }]
    assert body["tool_choice"] == "auto"


def test_build_body_omits_tool_fields_when_unset():
    """Callers without tools shouldn't see ``tools`` / ``tool_choice``
    in the body — keeps the wire request minimal."""
    body = build_request_body([HumanMessage("hi")], model="gpt-5.4")
    assert "tools" not in body
    assert "tool_choice" not in body


def test_build_body_specific_tool_choice():
    body = build_request_body(
        [HumanMessage("hi")],
        model="gpt-5.4",
        tools=[{"type": "function", "name": "x", "parameters": {}}],
        tool_choice={"type": "function", "name": "x"},
    )
    assert body["tool_choice"] == {"type": "function", "name": "x"}


# ─── AIMessage with tool_calls → input entries ─────────────────────────


def test_ai_message_tool_calls_become_function_call_entries():
    """Round-trip: an AIMessage with ``tool_calls`` produces
    function_call entries in the next request's input, with the
    call_id we'll later see on the ToolMessage."""
    ai = AIMessage(
        content="I'll check the weather.",
        tool_calls=[{
            "name": "get_weather",
            "args": {"location": "SF"},
            "id": "call_abc",
            "type": "tool_call",
        }],
    )
    body = build_request_body(
        [HumanMessage("weather in SF"), ai], model="gpt-5.4"
    )
    # Order: user message, assistant message (text), function_call item.
    assert len(body["input"]) == 3
    assert body["input"][1] == {
        "role": "assistant",
        "content": [{
            "type": "output_text",
            "text": "I'll check the weather.",
        }],
    }
    assert body["input"][2] == {
        "type": "function_call",
        "call_id": "call_abc",
        "name": "get_weather",
        "arguments": '{"location": "SF"}',
    }


def test_ai_message_tool_calls_only_no_text():
    """When the assistant emitted only tool calls (no text), don't
    inject an empty assistant message — Codex rejects empty
    output_text blocks."""
    ai = AIMessage(
        content="",
        tool_calls=[{
            "name": "ping", "args": {}, "id": "c1", "type": "tool_call",
        }],
    )
    body = build_request_body([HumanMessage("go"), ai], model="gpt-5.4")
    # Only the user message + the function_call (no empty assistant entry).
    types = [e.get("type") or e.get("role") for e in body["input"]]
    assert types == ["user", "function_call"]


def test_ai_message_multiple_tool_calls():
    """Multiple parallel tool_calls each become their own
    function_call entry."""
    ai = AIMessage(
        content="",
        tool_calls=[
            {"name": "a", "args": {"x": 1}, "id": "c1", "type": "tool_call"},
            {"name": "b", "args": {"y": 2}, "id": "c2", "type": "tool_call"},
        ],
    )
    body = build_request_body([HumanMessage("go"), ai], model="gpt-5.4")
    fcs = [e for e in body["input"] if e.get("type") == "function_call"]
    assert len(fcs) == 2
    assert fcs[0]["call_id"] == "c1"
    assert fcs[1]["call_id"] == "c2"


def test_ai_message_tool_call_missing_id_or_name_dropped():
    """Defensive: malformed ToolCall entries (no id, no name) get
    silently dropped rather than emitted as broken function_call
    items the gateway would 400 on."""
    ai = AIMessage(
        content="",
        tool_calls=[
            {"name": "", "args": {}, "id": "c1", "type": "tool_call"},
            {"name": "good", "args": {}, "id": "", "type": "tool_call"},
            {"name": "good", "args": {}, "id": "c3", "type": "tool_call"},
        ],
    )
    body = build_request_body([HumanMessage("go"), ai], model="gpt-5.4")
    fcs = [e for e in body["input"] if e.get("type") == "function_call"]
    assert len(fcs) == 1
    assert fcs[0]["call_id"] == "c3"


# ─── SSE → tool_calls on the AIMessage ─────────────────────────────────


def _tool_call_sse(text_before: str = "") -> bytes:
    """Build a realistic Codex SSE stream that calls a tool.

    Order mirrors what we observed in ``openai/codex`` test fixtures:
    output_item.added → function_call_arguments.delta (multiple) →
    response.completed.
    """
    events = []
    if text_before:
        events.extend([
            ("response.output_item.added", {
                "item": {"type": "message", "id": "msg_1"},
            }),
            ("response.output_text.delta", {"delta": text_before}),
        ])
    events.extend([
        ("response.output_item.added", {
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_xyz",
                "name": "get_weather",
                "arguments": "",
                "status": "in_progress",
            },
        }),
        ("response.function_call_arguments.delta", {
            "item_id": "fc_1",
            "delta": '{"locat',
        }),
        ("response.function_call_arguments.delta", {
            "item_id": "fc_1",
            "delta": 'ion":"SF"}',
        }),
        ("response.completed", {
            "response": {
                "id": "resp_test",
                "status": "completed",
                "model": "gpt-5.4",
                "usage": {
                    "input_tokens": 8, "output_tokens": 6, "total_tokens": 14,
                },
            },
        }),
    ])
    return _sse_bytes(events)


def test_consume_events_collects_tool_call_from_stream():
    body = _tool_call_sse()
    events = list(parse_sse_stream(body.decode().splitlines()))
    completion = consume_events(events)
    assert completion.response_id == "resp_test"
    assert completion.final_text == ""
    assert completion.tool_calls == [
        CodexToolCall(
            id="fc_1",
            call_id="call_xyz",
            name="get_weather",
            arguments_json='{"location":"SF"}',
        )
    ]


def test_invoke_returns_ai_message_with_tool_calls(auth_file):
    transport = _CaptureTransport(body=_tool_call_sse("Checking the weather. "))
    llm = _make_llm(auth_file, transport=transport)
    msg = llm.invoke([HumanMessage("weather in SF")])
    assert isinstance(msg, AIMessage)
    assert msg.content == "Checking the weather. "
    assert len(msg.tool_calls) == 1
    tc = msg.tool_calls[0]
    assert tc["name"] == "get_weather"
    assert tc["args"] == {"location": "SF"}
    assert tc["id"] == "call_xyz"


def test_invoke_records_invalid_tool_call_on_bad_json(auth_file):
    """If the model emits malformed JSON args, we surface an entry
    in ``invalid_tool_calls`` rather than silently dropping the call
    or crashing."""
    body = _sse_bytes([
        ("response.output_item.added", {
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_bad",
                "name": "do_thing",
                "arguments": "",
                "status": "in_progress",
            },
        }),
        ("response.function_call_arguments.delta", {
            "item_id": "fc_1",
            "delta": "{not json",
        }),
        ("response.completed", {
            "response": {"id": "r", "status": "completed", "model": "gpt-5.4"},
        }),
    ])
    transport = _CaptureTransport(body=body)
    llm = _make_llm(auth_file, transport=transport)
    msg = llm.invoke([HumanMessage("go")])
    assert msg.tool_calls == []
    assert len(msg.invalid_tool_calls) == 1
    assert msg.invalid_tool_calls[0]["name"] == "do_thing"
    assert msg.invalid_tool_calls[0]["id"] == "call_bad"


# ─── Streaming: tool_call_chunks ────────────────────────────────────────


def test_stream_emits_tool_call_chunks(auth_file):
    transport = _CaptureTransport(body=_tool_call_sse())
    llm = _make_llm(auth_file, transport=transport)
    chunks = list(llm.stream([HumanMessage("go")]))
    # Collect any chunk that has tool_call_chunks set.
    tcc = []
    for c in chunks:
        if getattr(c, "tool_call_chunks", None):
            tcc.extend(c.tool_call_chunks)
    # Kickoff chunk + 2 arg deltas = 3 tool_call_chunks total.
    assert len(tcc) == 3
    # Kickoff carries name + call_id.
    kickoff = tcc[0]
    assert kickoff["name"] == "get_weather"
    assert kickoff["id"] == "call_xyz"
    assert kickoff["index"] == 0
    # Subsequent chunks have None for name/id, just args deltas.
    assert tcc[1]["name"] is None
    assert tcc[1]["id"] is None
    assert tcc[1]["index"] == 0
    # Concatenated args = the full JSON
    args_joined = "".join(c["args"] or "" for c in tcc)
    assert args_joined == '{"location":"SF"}'


def test_stream_tool_call_chunks_indexed_per_call(auth_file):
    """Two parallel tool calls should get distinct indexes so the
    LangChain reducer assembles them into two separate ToolCalls."""
    body = _sse_bytes([
        ("response.output_item.added", {
            "item": {
                "type": "function_call", "id": "fc_a",
                "call_id": "ca", "name": "a", "arguments": "",
            },
        }),
        ("response.output_item.added", {
            "item": {
                "type": "function_call", "id": "fc_b",
                "call_id": "cb", "name": "b", "arguments": "",
            },
        }),
        ("response.function_call_arguments.delta", {
            "item_id": "fc_a", "delta": "{}",
        }),
        ("response.function_call_arguments.delta", {
            "item_id": "fc_b", "delta": "{}",
        }),
        ("response.completed", {
            "response": {"id": "r", "status": "completed"},
        }),
    ])
    transport = _CaptureTransport(body=body)
    llm = _make_llm(auth_file, transport=transport)
    chunks = list(llm.stream([HumanMessage("go")]))
    by_index: dict[int, list[Any]] = {}
    for c in chunks:
        for tc in getattr(c, "tool_call_chunks", None) or []:
            by_index.setdefault(tc["index"], []).append(tc)
    assert set(by_index.keys()) == {0, 1}


# ─── bind_tools accepts multiple input shapes ──────────────────────────


class WeatherArgs(BaseModel):
    """Look up the weather for a city."""
    location: str = Field(description="City name")


@tool
def get_weather(location: str) -> str:
    """Look up the weather."""
    return f"sunny in {location}"


def test_bind_tools_accepts_basetool(auth_file):
    llm = ChatCodexPlus(auth_file_path=auth_file)
    bound = llm.bind_tools([get_weather])
    # The bound runnable carries our tools in its kwargs — extract.
    bound_tools = bound.kwargs.get("tools")
    assert bound_tools is not None and len(bound_tools) == 1
    t = bound_tools[0]
    assert t["type"] == "function"
    assert t["name"] == "get_weather"
    # Codex shape is flat — name should NOT be nested under "function".
    assert "function" not in t
    assert "parameters" in t and t["parameters"]["type"] == "object"


def test_bind_tools_accepts_pydantic_model(auth_file):
    llm = ChatCodexPlus(auth_file_path=auth_file)
    bound = llm.bind_tools([WeatherArgs])
    bound_tools = bound.kwargs.get("tools")
    assert len(bound_tools) == 1
    assert bound_tools[0]["type"] == "function"
    assert bound_tools[0]["name"] == "WeatherArgs"


def test_bind_tools_accepts_pre_formed_dict(auth_file):
    """If the caller already has a Codex-shape dict, pass it through
    unchanged — no double-wrapping."""
    pre = {
        "type": "function",
        "name": "custom",
        "description": "Custom one.",
        "parameters": {"type": "object", "properties": {}},
    }
    llm = ChatCodexPlus(auth_file_path=auth_file)
    bound = llm.bind_tools([pre])
    assert bound.kwargs["tools"][0] == pre


def test_bind_tools_with_tool_choice(auth_file):
    llm = ChatCodexPlus(auth_file_path=auth_file)
    bound = llm.bind_tools(
        [get_weather], tool_choice={"type": "function", "name": "get_weather"}
    )
    assert bound.kwargs["tool_choice"] == {
        "type": "function", "name": "get_weather",
    }


def test_bound_runnable_actually_sends_tools_on_invoke(auth_file):
    """End-to-end: a bound runnable's invoke() puts the tools into
    the wire request."""
    transport = _CaptureTransport(body=_sse_bytes([
        ("response.completed", {
            "response": {
                "id": "r", "status": "completed", "model": "gpt-5.4",
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        }),
    ]))
    llm = _make_llm(auth_file, transport=transport)
    bound = llm.bind_tools([get_weather])
    bound.invoke([HumanMessage("go")])
    body = json.loads(transport.last_request.content)
    assert "tools" in body
    assert body["tools"][0]["name"] == "get_weather"
