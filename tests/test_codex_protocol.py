"""Tests for ``langchain_codex_plus.codex_protocol``.

SSE fixtures here are derived from a real Codex Plus ``/codex/
responses`` SSE stream captured 2026-05-20. Event ordering and field
names are faithful to what the server actually sends.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_codex_plus.codex_protocol import (
    CodexCompletion,
    CodexResponseError,
    SseEvent,
    build_request_body,
    consume_events,
    iter_text_deltas,
    parse_error_body,
    parse_sse_stream,
)

# ─── build_request_body ─────────────────────────────────────────────────


def test_build_body_user_only():
    body = build_request_body([HumanMessage("hello")], model="gpt-5.4")
    assert body["model"] == "gpt-5.4"
    assert body["stream"] is True
    assert body["store"] is False
    assert body["instructions"] == ""
    assert body["reasoning"] == {"effort": "none"}
    assert body["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "hello"}]}
    ]


def test_build_body_system_folds_into_instructions():
    """SystemMessage doesn't go in input — it's the ``instructions``
    field, mirroring how Codex CLI sends it."""
    body = build_request_body(
        [SystemMessage("be concise"), HumanMessage("hello")],
        model="gpt-5.4",
    )
    assert body["instructions"] == "be concise"
    # Only the human message survived into input.
    assert len(body["input"]) == 1
    assert body["input"][0]["role"] == "user"


def test_build_body_multiple_system_messages_concat():
    """Multiple SystemMessages: join with double newlines so each
    instruction is visually separated to the model."""
    body = build_request_body(
        [
            SystemMessage("rule 1"),
            SystemMessage("rule 2"),
            HumanMessage("go"),
        ],
        model="gpt-5.4",
    )
    assert body["instructions"] == "rule 1\n\nrule 2"


def test_build_body_explicit_instructions_override_wins():
    """Caller-supplied instructions override SystemMessages — useful
    when the caller wants programmatic system-prompt control."""
    body = build_request_body(
        [SystemMessage("ignored"), HumanMessage("go")],
        model="gpt-5.4",
        instructions_override="use this instead",
    )
    assert body["instructions"] == "use this instead"


def test_build_body_assistant_message_uses_output_text():
    """AIMessage in history gets ``role: assistant`` with content type
    ``output_text`` (per Codex's input schema)."""
    body = build_request_body(
        [
            HumanMessage("hi"),
            AIMessage("hello"),
            HumanMessage("how are you"),
        ],
        model="gpt-5.4",
    )
    assert body["input"][1] == {
        "role": "assistant",
        "content": [{"type": "output_text", "text": "hello"}],
    }


def test_build_body_tool_message_becomes_function_call_output():
    """``ToolMessage`` → Codex ``function_call_output`` entry with
    ``call_id`` echoing the original ``tool_call_id``."""
    body = build_request_body(
        [
            HumanMessage("call X"),
            ToolMessage(content="result-from-X", tool_call_id="call_t1"),
            HumanMessage("now what"),
        ],
        model="gpt-5.4",
    )
    fco = [e for e in body["input"] if e.get("type") == "function_call_output"]
    assert len(fco) == 1
    assert fco[0] == {
        "type": "function_call_output",
        "call_id": "call_t1",
        "output": "result-from-X",
    }


def test_build_body_reasoning_effort_passthrough():
    body = build_request_body(
        [HumanMessage("hi")], model="gpt-5.4", reasoning_effort="low"
    )
    assert body["reasoning"] == {"effort": "low"}


def test_build_body_extra_merges_after():
    """``extra`` is applied last → caller can override defaults
    (escape hatch for service_tier, temperature, etc.)."""
    body = build_request_body(
        [HumanMessage("hi")],
        model="gpt-5.4",
        extra={"temperature": 0.5, "service_tier": "default"},
    )
    assert body["temperature"] == 0.5
    assert body["service_tier"] == "default"


def test_build_body_extra_can_override_store_flag():
    body = build_request_body(
        [HumanMessage("hi")], model="gpt-5.4", extra={"store": True}
    )
    assert body["store"] is True


def test_build_body_multimodal_content_blocks_preserved():
    """Multi-block content with an image between text fragments
    surfaces as three Codex blocks (text → image → text) — the
    text doesn't collapse across the image. See ``test_multimodal``
    for the full set of accepted image-block shapes."""
    msg = HumanMessage(content=[
        {"type": "text", "text": "describe this"},
        {"type": "image_url", "image_url": "https://example.com/x.png"},
        {"type": "text", "text": " in detail"},
    ])
    body = build_request_body([msg], model="gpt-5.4")
    blocks = body["input"][0]["content"]
    assert [b["type"] for b in blocks] == [
        "input_text", "input_image", "input_text",
    ]
    assert blocks[0]["text"] == "describe this"
    assert blocks[1]["image_url"] == "https://example.com/x.png"
    assert blocks[2]["text"] == " in detail"


# ─── parse_sse_stream ───────────────────────────────────────────────────


SAMPLE_STREAM_LINES = [
    "event: response.created",
    'data: {"type":"response.created","response":{"id":"resp_abc","status":"in_progress"}}',
    "",
    "event: response.output_text.delta",
    'data: {"type":"response.output_text.delta","delta":"hel","sequence_number":4}',
    "",
    "event: response.output_text.delta",
    'data: {"type":"response.output_text.delta","delta":"lo","sequence_number":5}',
    "",
    "event: response.completed",
    'data: {"type":"response.completed","response":{"id":"resp_abc","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"hello"}]}],"usage":{"input_tokens":4,"output_tokens":2,"total_tokens":6}}}',
    "",
]


def test_parse_sse_stream_yields_one_event_per_block():
    events = list(parse_sse_stream(SAMPLE_STREAM_LINES))
    assert [e.event for e in events] == [
        "response.created",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.completed",
    ]


def test_parse_sse_stream_extracts_data_payloads():
    events = list(parse_sse_stream(SAMPLE_STREAM_LINES))
    assert events[1].data["delta"] == "hel"
    assert events[2].data["delta"] == "lo"
    assert events[3].data["response"]["id"] == "resp_abc"


def test_parse_sse_stream_skips_comments_and_blank_runs():
    """SSE keepalives (lines starting with ``:``) and stray blank lines
    must not produce ghost events."""
    lines = [
        ": keepalive",
        "",
        "",
        "event: response.created",
        'data: {"x": 1}',
        "",
        ": another comment",
        "",
    ]
    events = list(parse_sse_stream(lines))
    assert len(events) == 1
    assert events[0].event == "response.created"


def test_parse_sse_stream_handles_crlf():
    """Servers sometimes send CRLF; parser must strip both."""
    lines = [
        "event: response.created\r\n",
        'data: {"x": 1}\r\n',
        "\r\n",
    ]
    events = list(parse_sse_stream(lines))
    assert len(events) == 1
    assert events[0].data == {"x": 1}


def test_parse_sse_stream_invalid_json_yields_empty_data():
    """We don't crash on a malformed ``data:`` payload — yield with
    empty data so the consumer can decide whether to bail or skip."""
    lines = [
        "event: response.broken",
        "data: {not json}",
        "",
    ]
    events = list(parse_sse_stream(lines))
    assert events[0].event == "response.broken"
    assert events[0].data == {}


# ─── iter_text_deltas ───────────────────────────────────────────────────


def test_iter_text_deltas_yields_only_delta_strings():
    events = list(parse_sse_stream(SAMPLE_STREAM_LINES))
    assert list(iter_text_deltas(events)) == ["hel", "lo"]


def test_iter_text_deltas_skips_empty_deltas():
    """Defensive: a delta with empty string shouldn't propagate as
    an empty chunk (LangChain treats those as no-ops anyway, but
    cleaner to filter at the source)."""
    events = [
        SseEvent("response.output_text.delta", {"delta": ""}),
        SseEvent("response.output_text.delta", {"delta": "ok"}),
    ]
    assert list(iter_text_deltas(events)) == ["ok"]


# ─── consume_events ─────────────────────────────────────────────────────


def test_consume_events_returns_completion():
    events = list(parse_sse_stream(SAMPLE_STREAM_LINES))
    completion = consume_events(events)
    assert isinstance(completion, CodexCompletion)
    assert completion.response_id == "resp_abc"
    assert completion.final_text == "hello"
    assert completion.usage == {
        "input_tokens": 4,
        "output_tokens": 2,
        "total_tokens": 6,
    }
    assert completion.raw_response["status"] == "completed"


def test_consume_events_extracts_text_from_completed_when_no_deltas():
    """If the SSE stream had no deltas (shouldn't happen in practice
    but defensive), fall back to walking the completed event's
    ``output`` array."""
    events = [
        SseEvent("response.completed", {
            "type": "response.completed",
            "response": {
                "id": "resp_xyz",
                "output": [{
                    "type": "message",
                    "content": [{"type": "output_text", "text": "from-output"}],
                }],
                "usage": None,
            },
        }),
    ]
    completion = consume_events(events)
    assert completion.final_text == "from-output"
    assert completion.usage is None


def test_consume_events_raises_on_response_error():
    """A ``response.error`` event mid-stream is fatal."""
    events = [
        SseEvent("response.created", {"response": {"id": "resp_1"}}),
        SseEvent("response.error", {
            "error": {
                "message": "model unavailable",
                "type": "server_error",
                "code": "model_overloaded",
            },
        }),
    ]
    with pytest.raises(CodexResponseError) as exc:
        consume_events(events)
    assert "model unavailable" in str(exc.value)
    assert exc.value.code == "model_overloaded"
    assert exc.value.type == "server_error"


def test_consume_events_raises_when_stream_ends_without_completed():
    """A truncated stream (network drop, server crash) is fatal — but
    the error carries any text we managed to collect so the caller
    can log partial output."""
    events = [
        SseEvent("response.created", {"response": {"id": "r"}}),
        SseEvent("response.output_text.delta", {"delta": "partial"}),
        # ...no response.completed
    ]
    with pytest.raises(CodexResponseError) as exc:
        consume_events(events)
    assert exc.value.type == "stream_terminated_early"
    assert exc.value.raw == {"partial_text": "partial"}


# ─── parse_error_body ───────────────────────────────────────────────────


def test_parse_error_body_openai_shape():
    body = b'{"error":{"message":"bad model","type":"invalid_request_error","code":"unsupported_value","param":"model"}}'
    err = parse_error_body(body)
    assert err.message == "bad model"
    assert err.type == "invalid_request_error"
    assert err.code == "unsupported_value"


def test_parse_error_body_detail_shape():
    """Codex validators sometimes return ``{"detail": "..."}`` instead
    of the structured error shape (e.g., 'Stream must be set to true')."""
    body = b'{"detail":"Stream must be set to true"}'
    err = parse_error_body(body)
    assert err.message == "Stream must be set to true"
    assert err.type == "validation_error"


def test_parse_error_body_handles_garbage():
    """Some 5xx responses return HTML or empty bytes; surface them as
    the error message verbatim so the caller has something to log."""
    err = parse_error_body(b"<html>504 Gateway Timeout</html>")
    assert "504" in err.message


def test_parse_error_body_empty():
    err = parse_error_body(b"")
    assert err.message == "<empty response body>"


# ─── async incremental SSE parsing (0.0.4) ─────────────────────────────


async def test_aparse_sse_stream_matches_sync_parser():
    """The async parser yields identical events to the sync one."""
    import asyncio  # noqa: F401 — parity check only

    from langchain_codex_plus.codex_protocol import aparse_sse_stream

    raw = [
        "event: response.created",
        'data: {"response": {"id": "r1"}}',
        "",
        "event: response.output_text.delta",
        'data: {"delta": "hi"}',
        "",
        "event: response.completed",
        'data: {"response": {"id": "r1"}}',
        "",
    ]

    async def alines():
        for line in raw:
            yield line

    got = [(e.event, e.data) async for e in aparse_sse_stream(alines())]
    expected = [(e.event, e.data) for e in parse_sse_stream(raw)]
    assert got == expected


async def test_aparse_sse_stream_yields_before_stream_completes():
    """The async parser yields each event AS its lines arrive, without
    draining the rest of the stream — the property that lets ``_astream``
    surface Codex token deltas in real time instead of post-completion."""
    import asyncio

    from langchain_codex_plus.codex_protocol import aparse_sse_stream

    gate = asyncio.Event()

    async def alines():
        yield "event: response.output_text.delta"
        yield 'data: {"delta": "hel"}'
        yield ""  # boundary → first event flushes here
        # Block until the consumer has the first event; if the parser had
        # to drain the whole stream before yielding, this would deadlock.
        await gate.wait()
        yield "event: response.output_text.delta"
        yield 'data: {"delta": "lo"}'
        yield ""

    agen = aparse_sse_stream(alines())
    first = await agen.__anext__()
    assert first.data["delta"] == "hel"  # arrived while producer is blocked
    gate.set()
    second = await agen.__anext__()
    assert second.data["delta"] == "lo"
    await agen.aclose()
