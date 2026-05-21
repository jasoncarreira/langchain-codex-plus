"""Multimodal (image) content tests.

Codex's content schema accepts ``input_image`` blocks with either an
HTTP(S) URL or a ``data:`` URL. LangChain has two image-block
conventions; both must round-trip cleanly into the request body.
"""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage

from langchain_codex_plus.codex_protocol import build_request_body

# Note: the conftest fixtures ``auth_file`` / ``_make_llm`` /
# ``_CaptureTransport`` aren't needed here — multimodal mapping is
# pure protocol-layer logic. We exercise it via build_request_body.


PNG_1PX_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB"
    "0C8AAAAASUVORK5CYII="
)


def test_image_url_string_form():
    """LangChain convention 1: ``image_url`` value is a bare URL string."""
    msg = HumanMessage(content=[
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": "https://example.com/cat.png"},
    ])
    body = build_request_body([msg], model="gpt-5.4")
    assert body["input"] == [{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "What's in this image?"},
            {"type": "input_image", "image_url": "https://example.com/cat.png"},
        ],
    }]


def test_image_url_object_form_with_detail():
    """Convention 2: ``image_url`` value is a dict with ``url`` + ``detail``."""
    msg = HumanMessage(content=[
        {"type": "image_url", "image_url": {
            "url": "https://example.com/dog.png",
            "detail": "high",
        }},
        {"type": "text", "text": "Describe."},
    ])
    body = build_request_body([msg], model="gpt-5.4")
    blocks = body["input"][0]["content"]
    # Image block first (order preserved), then text.
    assert blocks[0] == {
        "type": "input_image",
        "image_url": "https://example.com/dog.png",
        "detail": "high",
    }
    assert blocks[1] == {"type": "input_text", "text": "Describe."}


def test_image_url_invalid_detail_dropped():
    """``detail`` must be one of ``auto|low|high``; an unknown value
    is omitted rather than passed through (Codex would 400)."""
    msg = HumanMessage(content=[
        {"type": "image_url", "image_url": {
            "url": "https://example.com/x.png",
            "detail": "ultra-extreme",  # invalid
        }},
    ])
    body = build_request_body([msg], model="gpt-5.4")
    img = body["input"][0]["content"][0]
    assert "detail" not in img
    assert img["image_url"] == "https://example.com/x.png"


def test_image_block_convention_url():
    """Newer LangChain shape: ``{type: image, source_type: url, url: ...}``."""
    msg = HumanMessage(content=[
        {
            "type": "image",
            "source_type": "url",
            "url": "https://example.com/a.jpg",
            "detail": "low",
        },
    ])
    body = build_request_body([msg], model="gpt-5.4")
    assert body["input"][0]["content"] == [{
        "type": "input_image",
        "image_url": "https://example.com/a.jpg",
        "detail": "low",
    }]


def test_image_block_convention_base64():
    """Newer LangChain shape: ``{type: image, source_type: base64, data: ...,
    mime_type: ...}`` → encoded as a ``data:`` URL."""
    msg = HumanMessage(content=[
        {
            "type": "image",
            "source_type": "base64",
            "data": PNG_1PX_BASE64,
            "mime_type": "image/png",
        },
    ])
    body = build_request_body([msg], model="gpt-5.4")
    img = body["input"][0]["content"][0]
    assert img["type"] == "input_image"
    assert img["image_url"].startswith("data:image/png;base64,")
    assert img["image_url"].endswith(PNG_1PX_BASE64)


def test_image_block_base64_defaults_to_png_mime():
    """When ``mime_type`` is missing, default to ``image/png`` rather
    than emitting a malformed data URL."""
    msg = HumanMessage(content=[
        {"type": "image", "source_type": "base64", "data": PNG_1PX_BASE64},
    ])
    body = build_request_body([msg], model="gpt-5.4")
    img = body["input"][0]["content"][0]
    assert img["image_url"].startswith("data:image/png;base64,")


def test_mixed_text_and_image_preserves_order():
    """Text and image blocks intermix in caller order; consecutive
    text fragments are merged into one block."""
    msg = HumanMessage(content=[
        {"type": "text", "text": "Compare "},
        {"type": "text", "text": "these "},
        {"type": "image_url", "image_url": "https://example.com/a.png"},
        {"type": "text", "text": "and "},
        {"type": "image_url", "image_url": "https://example.com/b.png"},
        {"type": "text", "text": " carefully."},
    ])
    body = build_request_body([msg], model="gpt-5.4")
    blocks = body["input"][0]["content"]
    # Coalesce adjacent text: "Compare these " + "and " + " carefully."
    assert [b["type"] for b in blocks] == [
        "input_text", "input_image", "input_text", "input_image", "input_text",
    ]
    assert blocks[0]["text"] == "Compare these "
    assert blocks[2]["text"] == "and "
    assert blocks[4]["text"] == " carefully."


def test_image_in_assistant_message_dropped():
    """Codex doesn't accept image blocks in assistant content (the
    model doesn't emit them; only text outputs). Drop silently rather
    than emit a malformed entry."""
    from langchain_core.messages import AIMessage

    ai = AIMessage(content=[
        {"type": "text", "text": "Looking at this."},
        {"type": "image_url", "image_url": "https://example.com/x.png"},
    ])
    body = build_request_body([HumanMessage("hi"), ai], model="gpt-5.4")
    # Assistant entry should have only the text block.
    assistant = next(e for e in body["input"] if e.get("role") == "assistant")
    assert assistant["content"] == [{
        "type": "output_text",
        "text": "Looking at this.",
    }]


def test_unknown_image_url_format_skipped():
    """A block with ``type: image_url`` but no ``url`` field is
    dropped rather than passed through as malformed."""
    msg = HumanMessage(content=[
        {"type": "image_url", "image_url": {"no_url": "oops"}},
        {"type": "text", "text": "real text"},
    ])
    body = build_request_body([msg], model="gpt-5.4")
    # Only the text block survives.
    assert body["input"][0]["content"] == [
        {"type": "input_text", "text": "real text"},
    ]


def test_string_content_still_works():
    """Regression: plain-string content (the common case) still
    produces a single input_text block."""
    msg = HumanMessage(content="just text")
    body = build_request_body([msg], model="gpt-5.4")
    assert body["input"] == [{
        "role": "user",
        "content": [{"type": "input_text", "text": "just text"}],
    }]


def test_empty_human_message_dropped():
    """Empty HumanMessage produces an entry with empty content list,
    which is invalid for Codex. We drop the entry entirely so the
    wire request stays valid."""
    msg = HumanMessage(content="")
    body = build_request_body([msg, HumanMessage("real")], model="gpt-5.4")
    # Only the real message survives.
    assert len(body["input"]) == 1
    assert body["input"][0]["content"][0]["text"] == "real"


def test_request_body_serializes_to_valid_json():
    """Defensive: a mixed multimodal body must JSON-encode cleanly
    (no non-serializable objects sneaking through the coercion)."""
    msg = HumanMessage(content=[
        {"type": "text", "text": "hi"},
        {"type": "image_url", "image_url": "https://example.com/x.png"},
    ])
    body = build_request_body([msg], model="gpt-5.4")
    encoded = json.dumps(body)
    assert '"input_image"' in encoded
    assert '"input_text"' in encoded
