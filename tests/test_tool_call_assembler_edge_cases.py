"""Edge cases for ``ToolCallAssembler`` matching and reconciliation.

``response.output_item.done`` is the server's contract frame; the completed
``output`` array is only a backstop, and its list positions are not trusted.
These cases pin the matching order (item id, then ``call_id``, then
``output_index``) and the rule that the first final frame wins.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from langchain_codex_plus.codex_protocol import consume_events, parse_sse_stream
from tests.conftest import _CaptureTransport, _make_llm, _sse_bytes

ARGS_A = '{"repository":"o/r","pull_request":1}'
ARGS_B = '{"path":"/tmp"}'


def _events(events):
    return list(parse_sse_stream(_sse_bytes(events).decode().splitlines()))


def _added(item_id, call_id, name, output_index=None):
    item = {"type": "function_call", "id": item_id, "name": name, "arguments": ""}
    if call_id is not None:
        item["call_id"] = call_id
    data = {"item": item}
    if output_index is not None:
        data["output_index"] = output_index
    return ("response.output_item.added", data)


def _done(item_id, call_id, name, args, output_index=None):
    data = {"item": {"type": "function_call", "id": item_id, "call_id": call_id,
                     "name": name, "arguments": args}}
    if output_index is not None:
        data["output_index"] = output_index
    return ("response.output_item.done", data)


def _completed(output=None):
    return ("response.completed", {"response": {"id": "r", "output": output or []}})


def test_completed_output_positions_never_rewrite_finished_calls():
    """A reasoning item shifts output_index; the completed array omits it and
    the item ids, so its positions point at the wrong calls."""
    events = _events([
        ("response.output_item.added",
         {"output_index": 0, "item": {"type": "reasoning", "id": "rs"}}),
        _added("fc_a", "call_a", "pr_metadata", 1),
        _added("fc_b", "call_b", "ls", 2),
        _done("fc_a", "call_a", "pr_metadata", ARGS_A, 1),
        _done("fc_b", "call_b", "ls", ARGS_B, 2),
        _completed([
            {"type": "function_call", "call_id": "call_a", "name": "pr_metadata",
             "arguments": ARGS_A},
            {"type": "function_call", "call_id": "call_b", "name": "ls",
             "arguments": ARGS_B},
        ]),
    ])
    calls = consume_events(events).tool_calls
    assert [(c.name, c.arguments_json) for c in calls] == [
        ("pr_metadata", ARGS_A), ("ls", ARGS_B),
    ]


def test_done_with_new_item_id_matches_by_call_id_without_a_duplicate():
    events = _events([
        _added("fc_a", "call_a", "list_schedules"),
        _done("fc_a_final", "call_a", "list_schedules", "{}", 0),
        _completed(),
    ])
    calls = consume_events(events).tool_calls
    assert [(c.call_id, c.arguments_json) for c in calls] == [("call_a", "{}")]


def _call_id_only_at_done_body():
    return _sse_bytes([
        _added("fc_a", None, "ls", 0),
        _done("fc_a", "call_a", "ls", ARGS_B, 0),
        _completed(),
    ])


def test_call_id_learned_only_at_done_reaches_stream_consumers(auth_file):
    llm = _make_llm(auth_file,
                    transport=_CaptureTransport(body=_call_id_only_at_done_body()))
    merged = None
    for chunk in llm.stream([HumanMessage("go")]):
        merged = chunk if merged is None else merged + chunk
    assert [c["id"] for c in merged.tool_calls] == ["call_a"]


def test_call_id_learned_only_at_done_reaches_invoke(auth_file):
    llm = _make_llm(auth_file,
                    transport=_CaptureTransport(body=_call_id_only_at_done_body()))
    message = llm.invoke([HumanMessage("go")])
    assert [c["id"] for c in message.tool_calls] == ["call_a"]


def test_parallel_calls_after_text_survive_stop_sequence_streaming(auth_file):
    body = _sse_bytes([
        ("response.output_text.delta", {"delta": "ok "}),
        _added("fc_a", "call_a", "ls", 1),
        _done("fc_a", "call_a", "ls", ARGS_B, 1),
        _added("fc_b", "call_b", "ls", 2),
        _done("fc_b", "call_b", "ls", ARGS_B, 2),
        _completed(),
    ])
    llm = _make_llm(auth_file, transport=_CaptureTransport(body=body))
    merged = None
    for chunk in llm.stream([HumanMessage("go")], stop=["ZZZ"]):
        merged = chunk if merged is None else merged + chunk
    assert [c["args"] for c in merged.tool_calls] == [{"path": "/tmp"}, {"path": "/tmp"}]
    assert merged.invalid_tool_calls == []


def _delta(item_id, output_index, delta):
    return ("response.function_call_arguments.delta",
            {"item_id": item_id, "output_index": output_index, "delta": delta})


def test_completed_output_without_ids_never_matches_streamed_calls_by_position():
    """Calls streamed only as deltas are never finalized by a completed
    array whose positions are shifted and whose items carry no ids."""
    events = _events([
        ("response.output_item.added",
         {"output_index": 0, "item": {"type": "reasoning", "id": "rs"}}),
        _added("fc_a", "call_a", "pr_metadata", 1),
        _added("fc_b", "call_b", "ls", 2),
        _delta("fc_a", 1, ARGS_A),
        _delta("fc_b", 2, ARGS_B),
        _completed([
            {"type": "function_call", "name": "pr_metadata", "arguments": ARGS_A},
            {"type": "function_call", "name": "ls", "arguments": ARGS_B},
        ]),
    ])
    calls = consume_events(events).tool_calls
    assert [(c.name, c.arguments_json) for c in calls] == [
        ("pr_metadata", ARGS_A), ("ls", ARGS_B),
    ]


def test_first_final_frame_wins_over_later_frames():
    events = _events([
        _added("fc_a", "call_a", "ls", 0),
        _done("fc_a", "call_a", "ls", ARGS_B, 0),
        ("response.function_call_arguments.done",
         {"item_id": "fc_a", "output_index": 0, "arguments": "{}"}),
        _completed([{"type": "function_call", "id": "fc_a", "call_id": "call_a",
                     "name": "ls", "arguments": '{"path":"/other"}'}]),
    ])
    calls = consume_events(events).tool_calls
    assert [c.arguments_json for c in calls] == [ARGS_B]


def _calls(events):
    return [(c.call_id, c.name, c.arguments_json)
            for c in consume_events(_events(events)).tool_calls]


def _done_without_id(call_id, name, args, output_index):
    return ("response.output_item.done", {
        "output_index": output_index,
        "item": {"type": "function_call", "call_id": call_id, "name": name,
                 "arguments": args},
    })


def test_completed_only_calls_without_item_ids_are_kept():
    assert _calls([_completed([
        {"type": "function_call", "call_id": "call_a", "name": "pr_metadata",
         "arguments": ARGS_A},
        {"type": "function_call", "call_id": "call_b", "name": "ls",
         "arguments": ARGS_B},
    ])]) == [("call_a", "pr_metadata", ARGS_A), ("call_b", "ls", ARGS_B)]


def test_item_done_only_calls_without_item_ids_are_kept():
    assert _calls([
        _done_without_id("call_a", "pr_metadata", ARGS_A, 0),
        _done_without_id("call_b", "ls", ARGS_B, 1),
        _completed(),
    ]) == [("call_a", "pr_metadata", ARGS_A), ("call_b", "ls", ARGS_B)]


def test_call_id_on_item_done_after_arguments_done_is_kept(auth_file):
    """The server normally sends arguments.done before output_item.done; a
    call_id that only the latter carries must still reach every consumer."""
    events = [
        _added("fc_a", None, "ls", 0),
        ("response.function_call_arguments.done",
         {"item_id": "fc_a", "output_index": 0, "arguments": ARGS_B}),
        _done("fc_a", "call_a", "ls", ARGS_B, 0),
        _completed(),
    ]
    assert _calls(events) == [("call_a", "ls", ARGS_B)]
    llm = _make_llm(auth_file, transport=_CaptureTransport(body=_sse_bytes(events)))
    merged = None
    for chunk in llm.stream([HumanMessage("go")]):
        merged = chunk if merged is None else merged + chunk
    assert [(c["id"], c["args"]) for c in merged.tool_calls] == [("call_a", {"path": "/tmp"})]
