"""Tests for ``ChatCodexPlus``.

Strategy: mock the HTTP transport at the ``httpx.Client.send`` level
so we exercise the real request-building + SSE-parsing + completion-
mapping code paths without touching the network. One real-account
smoke test is included but gated on a ``CODEX_PLUS_E2E=1`` env var.
"""
from __future__ import annotations

import json
import os

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from langchain_codex_plus import (
    ChatCodexPlus,
    CodexAuthNotFoundError,
    CodexRateLimits,
    CodexResponseError,
)

# Shared helpers (``_sse_bytes``, ``_make_llm``, ``_CaptureTransport``,
# ``_AsyncCaptureTransport``, ``auth_file`` fixture) come from
# ``tests/conftest.py``.
from tests.conftest import (
    _AsyncCaptureTransport,
    _CaptureTransport,
    _make_llm,
    _sse_bytes,
)


def _ok_sse_body(text: str = "ok") -> bytes:
    """A full streamed ``hello -> ok`` exchange — covers created,
    deltas, and completed events with usage."""
    return _sse_bytes([
        ("response.created", {
            "type": "response.created",
            "response": {"id": "resp_test", "status": "in_progress"},
        }),
        ("response.output_text.delta", {
            "type": "response.output_text.delta",
            "delta": text[:1],
            "sequence_number": 0,
        }),
        ("response.output_text.delta", {
            "type": "response.output_text.delta",
            "delta": text[1:],
            "sequence_number": 1,
        }),
        ("response.completed", {
            "type": "response.completed",
            "response": {
                "id": "resp_test",
                "status": "completed",
                "model": "gpt-5.4",
                "output": [{
                    "type": "message",
                    "content": [{"type": "output_text", "text": text}],
                }],
                "usage": {
                    "input_tokens": 3,
                    "output_tokens": 1,
                    "total_tokens": 4,
                },
            },
        }),
    ])


def _real_rl_headers() -> dict[str, str]:
    """Real ``x-codex-*`` headers captured 2026-05-20 from a Plus
    account — same fixture as in test_rate_limits."""
    return {
        "Content-Type": "text/event-stream",
        "x-codex-plan-type": "plus",
        "x-codex-active-limit": "premium",
        "x-codex-primary-used-percent": "1",
        "x-codex-primary-window-minutes": "300",
        "x-codex-primary-reset-after-seconds": "18000",
        "x-codex-primary-reset-at": "1779343790",
        "x-codex-primary-over-secondary-limit-percent": "0",
        "x-codex-secondary-used-percent": "0",
        "x-codex-secondary-window-minutes": "10080",
        "x-codex-secondary-reset-after-seconds": "604800",
        "x-codex-secondary-reset-at": "1779930590",
        "x-codex-credits-balance": "",
        "x-codex-credits-has-credits": "False",
        "x-codex-credits-unlimited": "False",
    }


# ─── Construction & auth ────────────────────────────────────────────────


def test_llm_type_and_identifying_params(auth_file):
    llm = ChatCodexPlus(auth_file_path=auth_file)
    assert llm._llm_type == "codex-plus"
    params = llm._identifying_params
    assert params["model"] == "gpt-5.4"
    assert params["reasoning_effort"] == "none"


def test_resolve_auth_raises_when_file_missing(tmp_path):
    """Strict-mode load: missing auth.json on _generate is a fatal
    config error, not silent degradation."""
    llm = ChatCodexPlus(auth_file_path=tmp_path / "nope.json")
    with pytest.raises(CodexAuthNotFoundError):
        llm._resolve_auth()


def test_resolve_auth_caches_after_first_load(auth_file):
    """Auth file is read once and cached on the instance."""
    llm = ChatCodexPlus(auth_file_path=auth_file)
    a1 = llm._resolve_auth()
    # Delete file; cached value should survive.
    auth_file.unlink()
    a2 = llm._resolve_auth()
    assert a1 is a2


# ─── Request building (via _build_body — pure) ──────────────────────────


def test_build_body_routes_through_protocol(auth_file):
    llm = ChatCodexPlus(
        auth_file_path=auth_file,
        model="gpt-5.4",
        reasoning_effort="low",
        store=True,
    )
    body = llm._build_body([
        SystemMessage("be terse"),
        HumanMessage("hi"),
    ])
    assert body["model"] == "gpt-5.4"
    assert body["instructions"] == "be terse"
    assert body["reasoning"] == {"effort": "low"}
    assert body["stream"] is True
    assert body["store"] is True
    assert body["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
    ]


def test_build_body_explicit_instructions_field_overrides(auth_file):
    llm = ChatCodexPlus(
        auth_file_path=auth_file, instructions="OVERRIDE"
    )
    body = llm._build_body([SystemMessage("ignored"), HumanMessage("hi")])
    assert body["instructions"] == "OVERRIDE"


def test_request_url_includes_client_version(auth_file):
    llm = ChatCodexPlus(auth_file_path=auth_file, client_version="9.9.9")
    assert llm._request_url() == (
        "https://chatgpt.com/backend-api/codex/responses"
        "?client_version=9.9.9"
    )


def test_request_headers_carry_oauth_bearer(auth_file):
    llm = ChatCodexPlus(auth_file_path=auth_file)
    auth = llm._resolve_auth()
    headers = llm._request_headers(auth)
    assert headers["Authorization"] == "Bearer atk-test"
    assert headers["Accept"] == "text/event-stream"
    assert headers["Content-Type"] == "application/json"
    assert "originator" in headers
    assert "User-Agent" in headers


def test_extra_headers_merge_in(auth_file):
    llm = ChatCodexPlus(
        auth_file_path=auth_file,
        extra_headers={"X-Trace-Id": "abc", "User-Agent": "custom"},
    )
    auth = llm._resolve_auth()
    headers = llm._request_headers(auth)
    assert headers["X-Trace-Id"] == "abc"
    assert headers["User-Agent"] == "custom"  # override default


# ─── Sync _generate path ────────────────────────────────────────────────


def test_generate_returns_chat_result(auth_file):
    transport = _CaptureTransport(
        body=_ok_sse_body("ok"), headers=_real_rl_headers()
    )
    llm = _make_llm(auth_file, transport=transport)
    result = llm.invoke([HumanMessage("say ok")])
    assert isinstance(result, AIMessage)
    assert result.content == "ok"
    assert result.id == "resp_test"
    assert result.response_metadata["finish_reason"] == "completed"
    assert result.response_metadata["model_name"] == "gpt-5.4"
    assert result.usage_metadata == {
        "input_tokens": 3,
        "output_tokens": 1,
        "total_tokens": 4,
    }


def test_generate_request_body_is_what_codex_expects(auth_file):
    transport = _CaptureTransport(body=_ok_sse_body())
    llm = _make_llm(auth_file, transport=transport, model="gpt-5.4")
    llm.invoke([HumanMessage("hi")])
    assert transport.last_request is not None
    body = json.loads(transport.last_request.content)
    assert body == {
        "model": "gpt-5.4",
        "instructions": "",
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "hi"}],
            }
        ],
        "stream": True,
        "store": False,
        "reasoning": {"effort": "none"},
    }


def test_generate_invokes_rate_limit_callback(auth_file):
    transport = _CaptureTransport(
        body=_ok_sse_body(), headers=_real_rl_headers()
    )
    seen: list[CodexRateLimits] = []
    llm = _make_llm(
        auth_file, transport=transport, rate_limit_callback=seen.append
    )
    llm.invoke([HumanMessage("hi")])
    assert len(seen) == 1
    rl = seen[0]
    assert rl.plan_type == "plus"
    assert rl.primary is not None and rl.primary.used_percent == 1.0
    assert rl.secondary is not None and rl.secondary.window_minutes == 10080


def test_generate_swallows_callback_exceptions(auth_file):
    """A broken rate_limit_callback must not break the response."""
    transport = _CaptureTransport(
        body=_ok_sse_body(), headers=_real_rl_headers()
    )

    def explode(_rl):
        raise RuntimeError("monitoring is down")

    llm = _make_llm(
        auth_file, transport=transport, rate_limit_callback=explode
    )
    # Should not raise.
    result = llm.invoke([HumanMessage("hi")])
    assert result.content == "ok"


def test_generate_raises_on_http_error(auth_file):
    transport = _CaptureTransport(
        status_code=400,
        body=b'{"detail":"Instructions are required"}',
        headers={"Content-Type": "application/json"},
    )
    llm = _make_llm(auth_file, transport=transport)
    with pytest.raises(CodexResponseError) as exc:
        llm.invoke([HumanMessage("hi")])
    assert "HTTP 400" in str(exc.value)
    assert "Instructions are required" in str(exc.value)


def test_generate_raises_on_oai_error_shape(auth_file):
    transport = _CaptureTransport(
        status_code=400,
        body=(
            b'{"error":{"message":"bad model","type":'
            b'"invalid_request_error","code":"unsupported_value"}}'
        ),
        headers={"Content-Type": "application/json"},
    )
    llm = _make_llm(auth_file, transport=transport)
    with pytest.raises(CodexResponseError) as exc:
        llm.invoke([HumanMessage("hi")])
    assert exc.value.code == "unsupported_value"
    assert exc.value.type == "invalid_request_error"


def test_generate_429_surfaces_rate_limit_headers(auth_file):
    """A 429 must surface status_code + headers + parsed rate_limits on
    the exception (so callers can pause until the real window reset) AND
    fire the rate-limit callback even on the refusal. Previously the
    headers were discarded — _raise_for_http_error ran before the
    callback on the success path, so a 429 was opaque."""
    transport = _CaptureTransport(
        status_code=429,
        body=b'{"detail":"Rate limit exceeded"}',
        headers=_real_rl_headers(),
    )
    seen: list[CodexRateLimits] = []
    llm = _make_llm(
        auth_file, transport=transport, rate_limit_callback=seen.append
    )
    with pytest.raises(CodexResponseError) as exc:
        llm.invoke([HumanMessage("hi")])
    err = exc.value
    assert err.status_code == 429
    assert err.headers is not None
    assert err.headers.get("x-codex-primary-reset-at") == "1779343790"
    assert err.rate_limits is not None
    assert err.rate_limits.primary is not None
    assert err.rate_limits.primary.reset_at == 1779343790
    # The callback fired on the 429 (usage snapshot updates on refusals).
    assert len(seen) == 1
    assert seen[0].primary.reset_at == 1779343790


def test_generate_429_without_codex_headers_still_sets_status(auth_file):
    """A 429 with no ``x-codex-*`` headers still carries status_code (and
    rate_limits is None) — callers fall back to a backoff."""
    transport = _CaptureTransport(
        status_code=429,
        body=b'{"detail":"Rate limit exceeded"}',
        headers={"Content-Type": "application/json"},
    )
    llm = _make_llm(auth_file, transport=transport)
    with pytest.raises(CodexResponseError) as exc:
        llm.invoke([HumanMessage("hi")])
    assert exc.value.status_code == 429
    assert exc.value.rate_limits is None


def test_generate_stop_argument_is_ignored_silently(auth_file, caplog):
    """Codex Responses API doesn't expose stop sequences. We log at
    DEBUG and proceed — silent drop in production logs."""
    transport = _CaptureTransport(body=_ok_sse_body())
    llm = _make_llm(auth_file, transport=transport)
    result = llm.invoke(
        [HumanMessage("hi")], stop=["END"]
    )
    assert result.content == "ok"


# ─── Sync _stream path ──────────────────────────────────────────────────


def test_stream_yields_chunks_per_delta(auth_file):
    transport = _CaptureTransport(body=_ok_sse_body("hello"))
    llm = _make_llm(auth_file, transport=transport)
    chunks = list(llm.stream([HumanMessage("hi")]))
    text_chunks = [c for c in chunks if c.content]
    # 2 deltas: "h" then "ello"
    assert len(text_chunks) == 2
    assert text_chunks[0].content == "h"
    assert text_chunks[1].content == "ello"


def test_stream_final_chunk_carries_usage_metadata(auth_file):
    """The chunk we emit after the ``response.completed`` event
    carries ``usage_metadata``. LangChain's stream layer appends its
    own terminal sentinel chunk after ours (with a synthetic ``lc_run``
    id), so we find the usage-bearing chunk by inspection rather than
    asserting on position."""
    transport = _CaptureTransport(body=_ok_sse_body("ok"))
    llm = _make_llm(auth_file, transport=transport)
    chunks = list(llm.stream([HumanMessage("hi")]))
    with_usage = [c for c in chunks if c.usage_metadata is not None]
    assert len(with_usage) == 1
    usage = with_usage[0].usage_metadata
    assert usage == {
        "input_tokens": 3,
        "output_tokens": 1,
        "total_tokens": 4,
    }


def test_stream_raises_on_response_error_event(auth_file):
    """A ``response.error`` event mid-stream must surface to the
    caller, not silently truncate."""
    body = _sse_bytes([
        ("response.created", {"response": {"id": "r"}}),
        ("response.error", {
            "error": {
                "message": "model unavailable",
                "type": "server_error",
                "code": "overloaded",
            }
        }),
    ])
    transport = _CaptureTransport(body=body)
    llm = _make_llm(auth_file, transport=transport)
    with pytest.raises(CodexResponseError) as exc:
        list(llm.stream([HumanMessage("hi")]))
    assert exc.value.code == "overloaded"


# ─── Async paths ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agenerate_returns_chat_result(auth_file):
    transport = _AsyncCaptureTransport(
        body=_ok_sse_body("ok"), headers=_real_rl_headers()
    )
    llm = _make_llm(auth_file, transport=transport)
    result = await llm.ainvoke([HumanMessage("say ok")])
    assert result.content == "ok"
    assert result.usage_metadata == {
        "input_tokens": 3,
        "output_tokens": 1,
        "total_tokens": 4,
    }


@pytest.mark.asyncio
async def test_astream_yields_chunks(auth_file):
    transport = _AsyncCaptureTransport(body=_ok_sse_body("hi!"))
    llm = _make_llm(auth_file, transport=transport)
    text = ""
    async for chunk in llm.astream([HumanMessage("go")]):
        text += chunk.content or ""
    assert text == "hi!"


@pytest.mark.asyncio
async def test_agenerate_raises_on_http_error(auth_file):
    """401 surfacing path with ``auto_refresh=False`` — refresh is
    disabled so the original error bubbles up unchanged. The
    auto-refresh + retry behavior is covered in test_oauth_refresh."""
    transport = _AsyncCaptureTransport(
        status_code=401,
        body=b'{"detail":"Invalid bearer token"}',
        headers={"Content-Type": "application/json"},
    )
    llm = _make_llm(auth_file, transport=transport, auto_refresh=False)
    with pytest.raises(CodexResponseError) as exc:
        await llm.ainvoke([HumanMessage("hi")])
    assert "HTTP 401" in str(exc.value)


# ─── Real-account smoke test (gated) ────────────────────────────────────


@pytest.mark.skipif(
    os.environ.get("CODEX_PLUS_E2E") != "1",
    reason="set CODEX_PLUS_E2E=1 to run; consumes a small amount of Codex Plus quota",
)
def test_real_account_smoke():
    """Hits the live Codex Plus endpoint. Use sparingly — consumes
    a few tokens of quota per invocation. Validates the full stack
    end-to-end."""
    llm = ChatCodexPlus(model="gpt-5.4", reasoning_effort="none")
    msg = llm.invoke([
        SystemMessage("Reply with exactly one short word."),
        HumanMessage("Say ok."),
    ])
    assert msg.content.strip()
    assert msg.usage_metadata is not None
    assert msg.usage_metadata.get("total_tokens", 0) > 0


# ── tool_choice normalization (the with_structured_output 400) ──────────────


def test_normalize_tool_choice_translates_anthropic_any_to_required() -> None:
    """`BaseChatModel.with_structured_output` calls bind_tools(tool_choice="any").

    "any" is Anthropic vocabulary. The Codex endpoint accepts only none/auto/
    required, so passing it through returns HTTP 400 and kills the turn. Every
    model that does not override with_structured_output inherits that call, so
    this is reached without any caller choosing it.
    """
    from langchain_codex_plus.codex_chat_model import _normalize_tool_choice

    assert _normalize_tool_choice("any") == "required"


@pytest.mark.parametrize("value", ["none", "auto", "required"])
def test_normalize_tool_choice_passes_protocol_values_through(value: str) -> None:
    from langchain_codex_plus.codex_chat_model import _normalize_tool_choice

    assert _normalize_tool_choice(value) == value


def test_normalize_tool_choice_handles_booleans_and_names_and_dicts() -> None:
    from langchain_codex_plus.codex_chat_model import _normalize_tool_choice

    assert _normalize_tool_choice(True) == "required"
    assert _normalize_tool_choice(False) == "none"
    assert _normalize_tool_choice(None) is None
    # A bare tool name, which ChatOpenAI also accepts.
    assert _normalize_tool_choice("my_tool") == {"type": "function", "name": "my_tool"}
    passthrough = {"type": "function", "name": "already_shaped"}
    assert _normalize_tool_choice(passthrough) == passthrough


def test_bind_tools_normalizes_before_the_request_is_built() -> None:
    """End-to-end through bind_tools, not just the helper.

    Testing the helper alone would pass even if bind_tools never called it --
    which is exactly how this bug shipped.
    """
    from langchain_codex_plus.codex_chat_model import ChatCodexPlus

    def sample_tool(x: int) -> int:
        """Double a number."""
        return x * 2

    model = ChatCodexPlus(model="gpt-5.6-sol")
    bound = model.bind_tools([sample_tool], tool_choice="any")
    assert bound.kwargs["tool_choice"] == "required"
