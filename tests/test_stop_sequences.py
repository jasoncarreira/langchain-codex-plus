"""Client-side stop-sequence tests.

Codex's ``/codex/responses`` endpoint rejects the ``stop`` parameter
outright (probed 2026-05-20: 400 ``"Unsupported parameter: stop"``).
This package implements stop sequences entirely on the client side:
buffer output as deltas arrive, search the running text for any stop
seq, truncate + exit early on the first match.

Caveat: Codex may keep generating tokens server-side after we close
the SSE connection. The quota impact is small on a subscription
account (windows charge by response, not by token), but worth noting.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from langchain_codex_plus.codex_protocol import (
    SseEvent,
    consume_events,
    first_stop_match,
)

# ─── first_stop_match ─────────────────────────────────────────────────


def test_first_stop_match_finds_earliest():
    """When multiple stops match, return the lowest index — the
    caller wants to truncate at the *first* match, not any match."""
    assert first_stop_match("abc DONE xyz END", ["END", "DONE"]) == 4


def test_first_stop_match_returns_none_when_no_match():
    assert first_stop_match("hello world", ["STOP", "FIN"]) is None


def test_first_stop_match_ignores_empty_strings():
    """An empty stop seq would otherwise match at index 0 every time
    — that's a footgun; treat it as a no-op."""
    assert first_stop_match("hello", ["", "world"]) is None
    assert first_stop_match("hello world", [""]) is None


def test_first_stop_match_handles_empty_text():
    assert first_stop_match("", ["STOP"]) is None


# ─── consume_events with stop_sequences ───────────────────────────────


def _events(*pairs: tuple[str, dict]) -> list[SseEvent]:
    return [SseEvent(event=e, data=d) for e, d in pairs]


def test_consume_events_no_stop_unchanged():
    """Without stop_sequences, behavior matches the existing contract:
    text accumulates, response.completed is required."""
    events = _events(
        ("response.created", {"response": {"id": "r"}}),
        ("response.output_text.delta", {"delta": "hello world"}),
        ("response.completed", {"response": {"id": "r", "status": "completed"}}),
    )
    c = consume_events(events)
    assert c.final_text == "hello world"


def test_consume_events_stops_within_single_delta():
    """A stop seq inside a single delta truncates at the match point."""
    events = _events(
        ("response.created", {"response": {"id": "r"}}),
        ("response.output_text.delta", {"delta": "answer: 42 END more text"}),
    )
    c = consume_events(events, stop_sequences=["END"])
    assert c.final_text == "answer: 42 "
    assert c.raw_response.get("stopped_at_client") is True
    assert c.raw_response.get("status") == "stopped_at_client"
    # No usage when we stopped early (no response.completed seen).
    assert c.usage is None


def test_consume_events_stops_across_delta_boundary():
    """Stop seq straddles two deltas — must still match. Common case
    since the model emits one token at a time."""
    events = _events(
        ("response.created", {"response": {"id": "r"}}),
        ("response.output_text.delta", {"delta": "hello "}),
        ("response.output_text.delta", {"delta": "EN"}),  # partial stop
        ("response.output_text.delta", {"delta": "D after"}),
    )
    c = consume_events(events, stop_sequences=["END"])
    assert c.final_text == "hello "
    assert c.raw_response.get("stopped_at_client") is True


def test_consume_events_first_matching_stop_wins():
    """When multiple stop seqs are provided, the earliest match in
    text wins (not the first in the list)."""
    events = _events(
        ("response.created", {"response": {"id": "r"}}),
        (
            "response.output_text.delta",
            {"delta": "x FIN y END z"},
        ),
    )
    c = consume_events(events, stop_sequences=["END", "FIN"])
    # FIN appears first in the text at index 2, even though END is
    # first in the stop list. Truncate at FIN.
    assert c.final_text == "x "


def test_consume_events_no_match_proceeds_normally():
    """Stop sequences set but no match — return as if stop weren't
    set at all (need response.completed)."""
    events = _events(
        ("response.created", {"response": {"id": "r"}}),
        ("response.output_text.delta", {"delta": "no match here"}),
        ("response.completed", {"response": {"id": "r", "status": "completed"}}),
    )
    c = consume_events(events, stop_sequences=["END"])
    assert c.final_text == "no match here"
    assert c.raw_response.get("status") == "completed"


def test_consume_events_stop_at_position_zero():
    """Edge case: stop seq matches at the very start of the first
    delta — emit empty text and stop immediately."""
    events = _events(
        ("response.created", {"response": {"id": "r"}}),
        ("response.output_text.delta", {"delta": "ENDx"}),
    )
    c = consume_events(events, stop_sequences=["END"])
    assert c.final_text == ""
    assert c.raw_response.get("stopped_at_client") is True


# ─── _generate path with stop ─────────────────────────────────────────


def _generate_test_sse(text: str = "hello END world"):
    """Build SSE bytes for an invoke()-shaped test that includes the
    stop sequence inside the body. We use ``_sse_bytes`` from conftest."""
    from tests.conftest import _sse_bytes
    return _sse_bytes([
        ("response.created", {"response": {"id": "r"}}),
        ("response.output_text.delta", {"delta": text}),
        ("response.completed", {"response": {"id": "r", "status": "completed"}}),
    ])


def test_invoke_with_stop_truncates_content(auth_file):
    from tests.conftest import _CaptureTransport, _make_llm

    transport = _CaptureTransport(body=_generate_test_sse("hello END world"))
    llm = _make_llm(auth_file, transport=transport)
    msg = llm.invoke([HumanMessage("hi")], stop=["END"])
    assert msg.content == "hello "
    # No usage_metadata when stopped early.
    assert msg.usage_metadata is None
    # finish_reason reflects the synthesized completion.
    assert msg.response_metadata["finish_reason"] == "stopped_at_client"


def test_invoke_without_stop_unchanged(auth_file):
    """Regression: no stop arg → behavior matches pre-stop tests."""
    from tests.conftest import _CaptureTransport, _make_llm

    transport = _CaptureTransport(body=_generate_test_sse("hello END world"))
    llm = _make_llm(auth_file, transport=transport)
    msg = llm.invoke([HumanMessage("hi")])
    assert msg.content == "hello END world"


# ─── _stream path with stop ───────────────────────────────────────────


def test_stream_truncates_chunks_at_stop(auth_file):
    """Streaming version of the truncation test. Once stop is matched,
    no further chunks should be yielded — and the chunk containing the
    match point should have its delta truncated."""
    from tests.conftest import _CaptureTransport, _make_llm, _sse_bytes

    body = _sse_bytes([
        ("response.created", {"response": {"id": "r"}}),
        ("response.output_text.delta", {"delta": "hello "}),
        ("response.output_text.delta", {"delta": "END"}),  # matches entirely
        ("response.output_text.delta", {"delta": " never seen"}),
        ("response.completed", {"response": {"id": "r", "status": "completed"}}),
    ])
    transport = _CaptureTransport(body=body)
    llm = _make_llm(auth_file, transport=transport)
    chunks = list(llm.stream([HumanMessage("hi")], stop=["END"]))
    text = "".join(c.content for c in chunks if c.content)
    # Only "hello " comes through; the "END" chunk truncates to "" and
    # we exit before the " never seen" chunk emits anything.
    assert text == "hello "
    # No usage chunk since we stopped early.
    usage_chunks = [c for c in chunks if c.usage_metadata is not None]
    assert usage_chunks == []


def test_stream_stop_split_across_chunks(auth_file):
    """Mid-stop split across deltas — most realistic case since the
    model tokenizes."""
    from tests.conftest import _CaptureTransport, _make_llm, _sse_bytes

    body = _sse_bytes([
        ("response.created", {"response": {"id": "r"}}),
        ("response.output_text.delta", {"delta": "see "}),
        ("response.output_text.delta", {"delta": "ST"}),
        ("response.output_text.delta", {"delta": "OP now"}),
        ("response.completed", {"response": {"id": "r", "status": "completed"}}),
    ])
    transport = _CaptureTransport(body=body)
    llm = _make_llm(auth_file, transport=transport)
    chunks = list(llm.stream([HumanMessage("hi")], stop=["STOP"]))
    text = "".join(c.content for c in chunks if c.content)
    assert text == "see "


# ─── Async paths ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ainvoke_with_stop(auth_file):
    from tests.conftest import _AsyncCaptureTransport, _make_llm

    transport = _AsyncCaptureTransport(body=_generate_test_sse("hi END bye"))
    llm = _make_llm(auth_file, transport=transport)
    msg = await llm.ainvoke([HumanMessage("hi")], stop=["END"])
    assert msg.content == "hi "
    assert msg.response_metadata["finish_reason"] == "stopped_at_client"


@pytest.mark.asyncio
async def test_astream_with_stop(auth_file):
    from tests.conftest import _AsyncCaptureTransport, _make_llm, _sse_bytes

    body = _sse_bytes([
        ("response.created", {"response": {"id": "r"}}),
        ("response.output_text.delta", {"delta": "alpha "}),
        ("response.output_text.delta", {"delta": "STOP "}),
        ("response.output_text.delta", {"delta": "beta"}),
        ("response.completed", {"response": {"id": "r", "status": "completed"}}),
    ])
    transport = _AsyncCaptureTransport(body=body)
    llm = _make_llm(auth_file, transport=transport)
    text = ""
    async for chunk in llm.astream([HumanMessage("hi")], stop=["STOP"]):
        text += chunk.content or ""
    assert text == "alpha "


# ─── Empty / no-op cases ──────────────────────────────────────────────


def test_stop_none_or_empty_list_behaves_like_no_stop(auth_file):
    from tests.conftest import _CaptureTransport, _make_llm

    transport = _CaptureTransport(body=_generate_test_sse("hello END world"))
    llm = _make_llm(auth_file, transport=transport)
    # Empty list → no stops to match → full text comes through.
    msg = llm.invoke([HumanMessage("hi")], stop=[])
    assert msg.content == "hello END world"
