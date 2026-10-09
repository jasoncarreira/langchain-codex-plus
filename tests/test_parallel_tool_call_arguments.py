"""Parallel tool calls keep their arguments whichever frame carries them.

Since 2026-10-06 Codex often delivers parallel ``function_call`` arguments
without usable ``response.function_call_arguments.delta`` events. The final
``response.function_call_arguments.done`` / ``response.output_item.done``
frames, or only the ``response.completed`` ``output`` array, carry them.
A delta-only parser turned 70-85% of parallel batches into empty arguments
in a live deployment. Every scenario below runs through ``invoke``,
``ainvoke``, ``stream`` and ``astream``.
"""
from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage

from langchain_codex_plus.codex_protocol import (
    ToolCallAssembler,
    consume_events,
    parse_sse_stream,
)
from tests.conftest import (
    _AsyncCaptureTransport,
    _CaptureTransport,
    _make_llm,
    _sse_bytes,
)

ARGS_A = '{"repository":"o/r","pull_request":1}'
ARGS_B = '{"path":"/tmp"}'


def _added(item_id: str, call_id: str, name: str, index: int) -> tuple[str, dict]:
    return ("response.output_item.added", {
        "output_index": index,
        "item": {"type": "function_call", "id": item_id, "call_id": call_id,
                 "name": name, "arguments": "", "status": "in_progress"},
    })


def _item_done(item_id: str, call_id: str, name: str, index: int, args: str) -> tuple[str, dict]:
    return ("response.output_item.done", {
        "output_index": index,
        "item": {"type": "function_call", "id": item_id, "call_id": call_id,
                 "name": name, "arguments": args, "status": "completed"},
    })


def _args_done(item_id: str, index: int, args: str) -> tuple[str, dict]:
    return ("response.function_call_arguments.done",
            {"item_id": item_id, "output_index": index, "arguments": args})


def _delta(item_id: str, index: int, delta: str) -> tuple[str, dict]:
    return ("response.function_call_arguments.delta",
            {"item_id": item_id, "output_index": index, "delta": delta})


def _completed(output: list[dict[str, Any]] | None = None) -> tuple[str, dict]:
    return ("response.completed", {"response": {
        "id": "resp_p", "status": "completed", "output": output or [],
    }})


def _fc(item_id: str, call_id: str, name: str, args: str) -> dict[str, Any]:
    return {"type": "function_call", "id": item_id, "call_id": call_id,
            "name": name, "arguments": args}


SCENARIOS: dict[str, list[tuple[str, dict]]] = {
    # Arguments arrive only on output_item.done (no deltas at all).
    "item_done_only": [
        _added("fc_a", "call_a", "pr_metadata", 0),
        _added("fc_b", "call_b", "ls", 1),
        _item_done("fc_a", "call_a", "pr_metadata", 0, ARGS_A),
        _item_done("fc_b", "call_b", "ls", 1, ARGS_B),
        _completed(),
    ],
    # Arguments arrive only on function_call_arguments.done.
    "arguments_done_only": [
        _added("fc_a", "call_a", "pr_metadata", 0),
        _added("fc_b", "call_b", "ls", 1),
        _args_done("fc_a", 0, ARGS_A),
        _args_done("fc_b", 1, ARGS_B),
        _completed(),
    ],
    # Deltas carry item ids that don't match the added frames; output_index
    # still identifies the call.
    "mismatched_delta_item_ids": [
        _added("fc_a", "call_a", "pr_metadata", 0),
        _added("fc_b", "call_b", "ls", 1),
        _delta("other_a", 0, ARGS_A[:10]),
        _delta("other_a", 0, ARGS_A[10:]),
        _delta("other_b", 1, ARGS_B),
        _completed(),
    ],
    # Nothing but the completed output array carries the arguments.
    "completed_output_only": [
        _added("fc_a", "call_a", "pr_metadata", 0),
        _added("fc_b", "call_b", "ls", 1),
        _completed([_fc("fc_a", "call_a", "pr_metadata", ARGS_A),
                    _fc("fc_b", "call_b", "ls", ARGS_B)]),
    ],
    # Calls that never had an added frame appear first in output_item.done.
    "done_without_added": [
        _item_done("fc_a", "call_a", "pr_metadata", 0, ARGS_A),
        _item_done("fc_b", "call_b", "ls", 1, ARGS_B),
        _completed(),
    ],
    # The classic shape: full deltas, then done frames repeating them. The
    # final frames must not duplicate argument text.
    "deltas_then_done": [
        _added("fc_a", "call_a", "pr_metadata", 0),
        _added("fc_b", "call_b", "ls", 1),
        _delta("fc_a", 0, ARGS_A[:7]),
        _delta("fc_b", 1, ARGS_B),
        _delta("fc_a", 0, ARGS_A[7:]),
        _args_done("fc_a", 0, ARGS_A),
        _item_done("fc_a", "call_a", "pr_metadata", 0, ARGS_A),
        _args_done("fc_b", 1, ARGS_B),
        _item_done("fc_b", "call_b", "ls", 1, ARGS_B),
        _completed([_fc("fc_a", "call_a", "pr_metadata", ARGS_A),
                    _fc("fc_b", "call_b", "ls", ARGS_B)]),
    ],
    # Partial deltas; the done frame completes the rest.
    "partial_deltas_then_done": [
        _added("fc_a", "call_a", "pr_metadata", 0),
        _added("fc_b", "call_b", "ls", 1),
        _delta("fc_a", 0, ARGS_A[:5]),
        _item_done("fc_a", "call_a", "pr_metadata", 0, ARGS_A),
        _item_done("fc_b", "call_b", "ls", 1, ARGS_B),
        _completed(),
    ],
}

EXPECTED = [
    {"name": "pr_metadata", "id": "call_a",
     "args": {"repository": "o/r", "pull_request": 1}},
    {"name": "ls", "id": "call_b", "args": {"path": "/tmp"}},
]


def _calls(message: Any) -> list[dict[str, Any]]:
    return [{"name": c["name"], "id": c["id"], "args": c["args"]}
            for c in message.tool_calls]


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_invoke_keeps_parallel_arguments(auth_file, scenario):
    llm = _make_llm(auth_file, transport=_CaptureTransport(body=_sse_bytes(SCENARIOS[scenario])))
    message = llm.invoke([HumanMessage("go")])
    assert _calls(message) == EXPECTED
    assert message.invalid_tool_calls == []


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
async def test_ainvoke_keeps_parallel_arguments(auth_file, scenario):
    llm = _make_llm(auth_file, transport=_AsyncCaptureTransport(body=_sse_bytes(SCENARIOS[scenario])))
    message = await llm.ainvoke([HumanMessage("go")])
    assert _calls(message) == EXPECTED
    assert message.invalid_tool_calls == []


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_stream_keeps_parallel_arguments(auth_file, scenario):
    llm = _make_llm(auth_file, transport=_CaptureTransport(body=_sse_bytes(SCENARIOS[scenario])))
    merged: AIMessageChunk | None = None
    for chunk in llm.stream([HumanMessage("go")]):
        merged = chunk if merged is None else merged + chunk
    assert merged is not None
    assert _calls(merged) == EXPECTED
    assert merged.invalid_tool_calls == []


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
async def test_astream_keeps_parallel_arguments(auth_file, scenario):
    llm = _make_llm(auth_file, transport=_AsyncCaptureTransport(body=_sse_bytes(SCENARIOS[scenario])))
    merged: AIMessageChunk | None = None
    async for chunk in llm.astream([HumanMessage("go")]):
        merged = chunk if merged is None else merged + chunk
    assert merged is not None
    assert _calls(merged) == EXPECTED
    assert merged.invalid_tool_calls == []


def test_single_call_with_deltas_and_done_is_not_duplicated():
    events = list(parse_sse_stream(_sse_bytes([
        _added("fc_a", "call_a", "pr_metadata", 0),
        _delta("fc_a", 0, ARGS_A[:9]),
        _delta("fc_a", 0, ARGS_A[9:]),
        _args_done("fc_a", 0, ARGS_A),
        _item_done("fc_a", "call_a", "pr_metadata", 0, ARGS_A),
        _completed([_fc("fc_a", "call_a", "pr_metadata", ARGS_A)]),
    ]).decode().splitlines()))
    assembler = ToolCallAssembler()
    streamed = "".join(f.args for event in events for f in assembler.feed(event))
    assert streamed == ARGS_A
    assert [c.arguments_json for c in assembler.tool_calls()] == [ARGS_A]


def test_diverged_stream_uses_final_arguments_for_completion():
    """When streamed text disagrees with the final frame, the non-streaming
    completion trusts the final arguments."""
    events = list(parse_sse_stream(_sse_bytes([
        _added("fc_a", "call_a", "ls", 0),
        _delta("fc_a", 0, '{"path":"/wrong"}'),
        _item_done("fc_a", "call_a", "ls", 0, ARGS_B),
        _completed(),
    ]).decode().splitlines()))
    completion = consume_events(events)
    assert [c.arguments_json for c in completion.tool_calls] == [ARGS_B]


def test_unrelated_output_items_are_ignored():
    assembler = ToolCallAssembler()
    for name, data in [
        ("response.output_item.added", {"item": {"type": "message", "id": "m"}}),
        ("response.output_item.done", {"item": {"type": "message", "id": "m"}}),
        ("response.function_call_arguments.delta", {"item_id": "nope", "delta": "{}"}),
        ("response.function_call_arguments.done", {"item_id": "nope", "arguments": "{}"}),
    ]:
        assert assembler.feed(type("E", (), {"event": name, "data": data})()) == []
    assert assembler.tool_calls() == []
