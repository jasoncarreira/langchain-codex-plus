"""Content-type safety + request diagnostics.

Two related defenses against ``HTTP 400: Unsupported content type`` from
``/codex/responses``:

1. Unsupported ``data:``/base64 image MIMEs are dropped at serialization
   (the backend rejects them, failing the whole turn).
2. ``CodexResponseError`` carries a PII-light ``request_summary`` so a
   content rejection names which content types were in the request.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from langchain_codex_plus.codex_protocol import (
    CodexResponseError,
    build_request_body,
    summarize_request_content,
)

PNG_1PX_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB"
    "0C8AAAAASUVORK5CYII="
)


def _user_content(msg: HumanMessage) -> list[dict]:
    body = build_request_body([msg], model="gpt-5.4")
    return body["input"][0]["content"]


# ─── image MIME validation (unsupported = dropped, not sent) ─────────────


def test_unsupported_data_url_image_is_dropped():
    """A ``data:image/svg+xml`` block is dropped; the text survives."""
    msg = HumanMessage(content=[
        {"type": "text", "text": "look"},
        {"type": "image_url",
         "image_url": "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4="},
    ])
    content = _user_content(msg)
    assert content == [{"type": "input_text", "text": "look"}]
    assert not any(b.get("type") == "input_image" for b in content)


def test_unsupported_base64_image_mime_is_dropped():
    """``source_type: base64`` with an unsupported ``mime_type`` is dropped."""
    msg = HumanMessage(content=[
        {"type": "text", "text": "look"},
        {"type": "image", "source_type": "base64",
         "data": PNG_1PX_BASE64, "mime_type": "image/tiff"},
    ])
    content = _user_content(msg)
    assert content == [{"type": "input_text", "text": "look"}]


def test_supported_data_url_image_is_kept():
    """A ``data:image/png`` block round-trips into ``input_image``."""
    data_url = f"data:image/png;base64,{PNG_1PX_BASE64}"
    msg = HumanMessage(content=[
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": data_url},
    ])
    content = _user_content(msg)
    assert {"type": "input_image", "image_url": data_url} in content


def test_remote_url_image_passes_through():
    """Remote URLs can't be MIME-validated locally — pass them through
    (the backend validates on fetch)."""
    msg = HumanMessage(content=[
        {"type": "image_url", "image_url": "https://example.com/cat.png"},
    ])
    content = _user_content(msg)
    assert {"type": "input_image",
            "image_url": "https://example.com/cat.png"} in content


def test_supported_base64_image_is_kept():
    msg = HumanMessage(content=[
        {"type": "image", "source_type": "base64",
         "data": PNG_1PX_BASE64, "mime_type": "image/png"},
    ])
    content = _user_content(msg)
    assert content == [{
        "type": "input_image",
        "image_url": f"data:image/png;base64,{PNG_1PX_BASE64}",
    }]


# ─── summarize_request_content ───────────────────────────────────────────


def test_summary_counts_types_and_image_scheme():
    body = build_request_body([
        HumanMessage(content=[
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": "https://x.example/a.png"},
        ])
    ], model="gpt-5.4")
    s = summarize_request_content(body)
    assert s["message_count"] == 1
    assert s["content_types_by_role"]["user"]["input_text"] == 1
    assert s["content_types_by_role"]["user"]["input_image"] == 1
    assert s["images"] == [{"role": "user", "scheme": "https"}]
    assert s["unexpected_content_types"] == []
    assert s["approx_content_chars"] == len("hello")


def test_summary_reports_data_url_mime():
    body = build_request_body([
        HumanMessage(content=[
            {"type": "image_url",
             "image_url": f"data:image/png;base64,{PNG_1PX_BASE64}"},
        ])
    ], model="gpt-5.4")
    s = summarize_request_content(body)
    assert s["images"] == [{"role": "user", "mime": "image/png"}]


def test_summary_surfaces_unexpected_content_type():
    """The prime suspect for a rejection: a content-part ``type`` outside
    the schema's known set. Built by hand since the serializer never emits
    one — this guards what the diagnostic reports if the backend ever does."""
    body = {"input": [
        {"role": "user", "content": [
            {"type": "input_text", "text": "x"},
            {"type": "input_audio", "audio": "..."},
        ]},
    ]}
    s = summarize_request_content(body)
    assert s["unexpected_content_types"] == ["input_audio"]
    assert s["content_types_by_role"]["user"]["input_audio"] == 1


def test_summary_handles_function_call_entries():
    body = {"input": [
        {"type": "function_call", "name": "f", "arguments": "{}", "call_id": "c1"},
        {"type": "function_call_output", "call_id": "c1", "output": "done"},
    ]}
    s = summarize_request_content(body)
    assert s["content_types_by_role"]["function_call"]["_count"] == 1
    assert s["content_types_by_role"]["function_call_output"]["_count"] == 1
    assert s["message_count"] == 2


def test_summary_no_raw_text_leaks():
    """Diagnostics must never carry raw message text (PII)."""
    secret = "SENSITIVE-OPERATOR-SECRET-12345"
    body = build_request_body([
        HumanMessage(content=secret),
    ], model="gpt-5.4")
    s = summarize_request_content(body)
    assert secret not in repr(s)
    assert s["approx_content_chars"] == len(secret)


# ─── CodexResponseError.request_summary ──────────────────────────────────


def test_error_carries_request_summary():
    summary = {"images": [{"role": "user", "mime": "image/svg+xml"}],
               "unexpected_content_types": []}
    err = CodexResponseError(
        message="HTTP 400: Unsupported content type",
        status_code=400,
        request_summary=summary,
    )
    assert err.request_summary == summary
    assert "request_summary" in repr(err)


def test_error_request_summary_defaults_none():
    err = CodexResponseError(message="boom")
    assert err.request_summary is None
    assert "request_summary" not in repr(err)
