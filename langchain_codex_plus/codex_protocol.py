"""Codex ``/codex/responses`` protocol — request building + SSE parsing.

Pure functions; no I/O. The chat model in ``codex_chat_model.py`` wraps
these with httpx and pydantic.

Verified empirically 2026-05-20 against a real Plus account:

* ``POST /codex/responses?client_version=<v>`` requires:
    - ``model`` (e.g., ``"gpt-5.4"``)
    - ``instructions`` (string; SystemMessage content concat'd here)
    - ``input`` (list of messages; ``{"role": ..., "content": [{"type":
      "input_text" | "output_text", "text": ...}]}``)
    - ``stream: true`` (non-streaming NOT supported)
* Does NOT accept ``max_output_tokens``.
* ``reasoning.effort`` for ``gpt-5.4`` must be one of:
  ``none | low | medium | high | xhigh`` (NOT ``"minimal"``).

SSE event types observed (text-only flow):

* ``response.created`` — initial envelope, contains response.id
* ``response.in_progress`` — server has started
* ``response.output_item.added`` — a new message item begins
* ``response.content_part.added`` — a content block within the message
* ``response.output_text.delta`` — incremental text token (``data.delta``)
* ``response.output_text.done`` — final text for the content part
* ``response.content_part.done`` — closes the content block
* ``response.output_item.done`` — closes the message item
* ``response.completed`` — terminal event; ``data.response.usage`` has totals

Errors mid-stream arrive as ``event: response.error`` or as a plain JSON
body when the HTTP envelope is non-200; both shapes carry
``{"error": {"message": ..., "type": ..., "code": ...}}``.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

# ─── Request building ──────────────────────────────────────────────────


VALID_REASONING_EFFORTS = frozenset({"none", "low", "medium", "high", "xhigh"})
"""For ``gpt-5.4``. Other models may accept different values; the
chat model passes whatever the caller sets and lets the gateway
validate (caller gets a clear 400 if it's wrong)."""


def _message_to_input_entries(message: Any) -> list[dict[str, Any]]:
    """Convert one LangChain ``BaseMessage`` to Codex input entries.

    Returns a list (possibly empty) — one message can expand to
    multiple entries when an AIMessage has both text content and
    tool calls (Codex represents each ``tool_call`` as its own
    top-level ``function_call`` entry, not nested under the assistant
    message). SystemMessage returns an empty list — it's folded into
    ``instructions`` separately.
    """
    role = getattr(message, "type", None)
    if role == "system":
        return []
    if role == "human":
        blocks = _coerce_to_codex_blocks(message, role="user")
        if not blocks:
            return []
        return [{"role": "user", "content": blocks}]
    if role == "ai":
        return _ai_message_to_entries(message)
    if role == "tool":
        return [_tool_message_to_function_call_output(message)]
    # Unknown message type — pass as user. Better visible than silent.
    blocks = _coerce_to_codex_blocks(message, role="user")
    if not blocks:
        return []
    return [{"role": "user", "content": blocks}]


def _ai_message_to_entries(message: Any) -> list[dict[str, Any]]:
    """Map an ``AIMessage`` (possibly carrying tool_calls) to Codex
    input entries.

    Codex's Responses API represents an assistant turn that produced
    tool calls as **multiple** top-level input entries:

    1. (Optional) a ``role: assistant`` message with the text content,
       only emitted when there's non-empty text — Codex rejects empty
       output_text blocks.
    2. One ``type: function_call`` entry per tool_call, with
       ``call_id``, ``name``, and JSON-stringified ``arguments``.

    LangChain's ``AIMessage.tool_calls`` is a list of
    :class:`langchain_core.messages.tool.ToolCall` dicts with
    ``{"name", "args" (dict), "id"}``. We serialize ``args`` to
    JSON for the wire.
    """
    entries: list[dict[str, Any]] = []
    # Assistant messages emit ``output_text`` blocks (no images — Codex
    # doesn't accept image content in assistant turns; the model
    # doesn't generate them anyway).
    blocks = _coerce_to_codex_blocks(message, role="assistant")
    if blocks:
        entries.append({"role": "assistant", "content": blocks})
    for tc in getattr(message, "tool_calls", None) or []:
        # ToolCall is a TypedDict in newer langchain-core; both
        # attribute and item access work via dict lookup on TypedDict
        # so accept either shape defensively.
        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
        args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
        call_id = (
            tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
        )
        if not name or not call_id:
            continue  # malformed tool call — skip rather than emit garbage
        entries.append({
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": json.dumps(args or {}),
        })
    return entries


def _tool_message_to_function_call_output(message: Any) -> dict[str, Any]:
    """``ToolMessage`` → Codex ``function_call_output`` entry.

    LangChain's ``ToolMessage`` has ``content`` (typically string)
    and ``tool_call_id`` referencing the original tool_call. Codex
    wants ``call_id`` matching the function_call's ``call_id``.
    """
    return {
        "type": "function_call_output",
        "call_id": getattr(message, "tool_call_id", None) or "",
        "output": _coerce_message_content_to_text(message),
    }


def _coerce_message_content_to_text(message: Any) -> str:
    """Flatten a LangChain message's content to a single string.

    Used by call sites that need a string (``instructions`` field,
    ``function_call_output.output``). Image blocks are dropped here
    by design — those contexts don't accept multimodal content.

    For preserving multimodal content into Codex content blocks, see
    :func:`_coerce_to_codex_blocks` instead.
    """
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content) if content is not None else ""


# Codex ``ImageDetail`` from app-server-protocol/ContentItem.ts.
_VALID_IMAGE_DETAILS = frozenset({"auto", "low", "high"})


def _coerce_to_codex_blocks(
    message: Any, *, role: str
) -> list[dict[str, Any]]:
    """Convert a LangChain message's content to Codex content blocks.

    Codex's content schema (``ContentItem.ts``):

    * ``{"type": "input_text", "text": <str>}`` — user / tool input
    * ``{"type": "input_image", "image_url": <url-or-data-url>,
      "detail"?: "auto" | "low" | "high"}`` — image input
    * ``{"type": "output_text", "text": <str>}`` — assistant output

    Accepted LangChain image-block conventions:

    1. ``{"type": "image_url", "image_url": <url-string>}``
    2. ``{"type": "image_url", "image_url": {"url": ..., "detail": ...}}``
    3. ``{"type": "image", "source_type": "url", "url": ...}``
    4. ``{"type": "image", "source_type": "base64", "data": ...,
       "mime_type": "image/png"}`` — encoded as ``data:`` URL

    The ``role`` parameter picks the text-block type: ``"user"`` /
    ``"tool"`` → ``input_text``, ``"assistant"`` → ``output_text``.
    Image blocks always serialize to ``input_image`` (Codex doesn't
    expose an output_image type — the assistant turn never carries
    images, so we drop image blocks on assistant content with a debug
    log path via the no-op).

    Returns ``[]`` for genuinely empty content — caller decides
    whether to drop the message entirely or treat empty as a no-op.
    """
    text_type = "output_text" if role == "assistant" else "input_text"
    content = getattr(message, "content", "")
    blocks: list[dict[str, Any]] = []
    if isinstance(content, str):
        if content:
            blocks.append({"type": text_type, "text": content})
        return blocks
    if not isinstance(content, list):
        # Defensive — coerce odd shapes (e.g., Pydantic models) to str.
        s = str(content) if content is not None else ""
        if s:
            blocks.append({"type": text_type, "text": s})
        return blocks

    accumulated_text: list[str] = []

    def _flush_text() -> None:
        """Coalesce adjacent text fragments into a single block. The
        caller-visible content stays a one-text-block + N-image-blocks
        list, not an explosion of single-character text blocks."""
        if accumulated_text:
            joined = "".join(accumulated_text)
            if joined:
                blocks.append({"type": text_type, "text": joined})
            accumulated_text.clear()

    for block in content:
        if isinstance(block, str):
            accumulated_text.append(block)
            continue
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text" or ("text" in block and "type" not in block):
            text = block.get("text")
            if isinstance(text, str):
                accumulated_text.append(text)
            continue
        # Multimodal: image variants. Only accept on non-assistant roles.
        if role == "assistant":
            # Codex doesn't accept image blocks in assistant content;
            # silently drop them rather than emit a malformed entry.
            continue
        image_block = _try_extract_image_block(block, btype)
        if image_block is not None:
            _flush_text()
            blocks.append(image_block)
            continue
        # Unknown block type with a text-ish field — last-ditch fallback
        # so a stray ``{"type": "custom", "text": "..."}`` doesn't get
        # dropped silently.
        fallback = block.get("text") or block.get("content")
        if isinstance(fallback, str):
            accumulated_text.append(fallback)

    _flush_text()
    return blocks


def _try_extract_image_block(
    block: dict[str, Any], btype: str | None
) -> dict[str, Any] | None:
    """Map a LangChain image block to a Codex ``input_image`` block.

    Returns ``None`` if the block isn't an image variant or doesn't
    carry the URL/data needed to construct one.
    """
    # Convention 1 / 2: ``{"type": "image_url", "image_url": <str|dict>}``
    if btype == "image_url":
        iu = block.get("image_url")
        if isinstance(iu, str) and iu:
            return {"type": "input_image", "image_url": iu}
        if isinstance(iu, dict):
            url = iu.get("url")
            if not isinstance(url, str) or not url:
                return None
            out: dict[str, Any] = {"type": "input_image", "image_url": url}
            detail = iu.get("detail")
            if isinstance(detail, str) and detail in _VALID_IMAGE_DETAILS:
                out["detail"] = detail
            return out
        return None
    # Convention 3 / 4: ``{"type": "image", "source_type": ..., ...}``
    if btype == "image":
        source_type = block.get("source_type")
        if source_type == "url":
            url = block.get("url")
            if isinstance(url, str) and url:
                out = {"type": "input_image", "image_url": url}
                detail = block.get("detail")
                if isinstance(detail, str) and detail in _VALID_IMAGE_DETAILS:
                    out["detail"] = detail
                return out
            return None
        if source_type == "base64":
            data = block.get("data")
            mime = block.get("mime_type") or "image/png"
            if isinstance(data, str) and data:
                return {
                    "type": "input_image",
                    "image_url": f"data:{mime};base64,{data}",
                }
            return None
        return None
    return None


def _extract_instructions(messages: Iterable[Any]) -> str:
    """Concatenate all SystemMessage contents into a single
    ``instructions`` string. Codex's instructions field is a string,
    not a list — multi-system-message inputs get joined with
    double-newlines."""
    parts: list[str] = []
    for m in messages:
        if getattr(m, "type", None) == "system":
            text = _coerce_message_content_to_text(m)
            if text:
                parts.append(text)
    return "\n\n".join(parts)


ToolChoice = (
    str  # "auto" | "none" | "required"
    | dict[str, Any]  # {"type": "function", "name": "..."}
)


def build_request_body(
    messages: Iterable[Any],
    *,
    model: str,
    reasoning_effort: str = "none",
    instructions_override: str | None = None,
    store: bool = False,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: ToolChoice | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON body for ``POST /codex/responses``.

    Always sets ``stream: true`` (Codex requires it). ``instructions``
    comes from ``instructions_override`` if provided, else from
    SystemMessages in ``messages``.

    ``tools`` shape (Responses API):
        ``[{"type": "function", "name": "...", "description": "...",
            "parameters": <JSON schema dict>}]``

    ``tool_choice`` accepts:
        * ``"auto"`` — model decides whether to call (default behavior)
        * ``"none"`` — disable tool calls for this turn
        * ``"required"`` — force a tool call
        * ``{"type": "function", "name": "..."}`` — force a specific one

    Only included in the body when non-None / non-empty, so callers
    without tools don't pay for the extra fields.

    ``extra`` is merged last — escape hatch for caller-supplied fields
    the protocol may add over time (``service_tier``, ``temperature``,
    etc.).
    """
    messages_list = list(messages)
    instructions = (
        instructions_override
        if instructions_override is not None
        else _extract_instructions(messages_list)
    )
    input_entries: list[dict[str, Any]] = []
    for m in messages_list:
        input_entries.extend(_message_to_input_entries(m))
    body: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": input_entries,
        "stream": True,
        "store": store,
        "reasoning": {"effort": reasoning_effort},
    }
    if tools:
        body["tools"] = list(tools)
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    if extra:
        body.update(extra)
    return body


# ─── SSE parsing ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SseEvent:
    """One parsed SSE event: an ``event:`` line + a ``data:`` line.

    Codex sends one event-data pair separated by a blank line. We
    expose both fields raw; downstream helpers extract specific
    payloads.
    """

    event: str
    data: dict[str, Any] = field(default_factory=dict)


def parse_sse_stream(lines: Iterable[str]) -> Iterator[SseEvent]:
    """Parse Codex's SSE byte stream into :class:`SseEvent` objects.

    Input: iterable of decoded lines (one per ``\\n`` boundary). The
    chat model decodes the response stream and passes the lines here.

    Robust to:

    * Blank lines (event separators) — used as boundary markers.
    * Multi-line ``data:`` values — accumulated until the blank line.
    * Missing ``event:`` — yields an event with empty ``event`` string
      so callers can detect malformed input.
    * Garbage ``data:`` JSON — yielded with ``data={}`` and the raw
      text dropped (we err on the side of "keep streaming" over
      "crash mid-response").
    """
    current_event: str = ""
    data_buffer: list[str] = []

    def flush() -> SseEvent | None:
        nonlocal current_event, data_buffer
        if not current_event and not data_buffer:
            return None
        raw_data = "\n".join(data_buffer)
        parsed: dict[str, Any]
        if not raw_data:
            parsed = {}
        else:
            try:
                parsed = json.loads(raw_data)
                if not isinstance(parsed, dict):
                    parsed = {"_raw": parsed}
            except json.JSONDecodeError:
                parsed = {}
        evt = SseEvent(event=current_event, data=parsed)
        current_event = ""
        data_buffer = []
        return evt

    for line in lines:
        # SSE lines are LF-separated; if iterable yields with trailing
        # ``\r\n`` strip it.
        line = line.rstrip("\r\n")
        if not line:
            # Empty line = event boundary.
            evt = flush()
            if evt is not None:
                yield evt
            continue
        if line.startswith(":"):
            # SSE comment line — heartbeat / keepalive. Ignore.
            continue
        if line.startswith("event:"):
            current_event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_buffer.append(line[len("data:"):].lstrip())
        # Ignore unknown SSE fields (``id:``, ``retry:``) — Codex
        # doesn't use them today.

    # Trailing event without a final blank line (rare; defensive).
    final = flush()
    if final is not None:
        yield final


def iter_text_deltas(events: Iterable[SseEvent]) -> Iterator[str]:
    """Yield incremental text fragments from a Codex SSE event stream.

    Only emits ``response.output_text.delta`` events' ``delta`` field.
    Skips bookkeeping events (item.added, content_part.added, done,
    completed).
    """
    for evt in events:
        if evt.event == "response.output_text.delta":
            delta = evt.data.get("delta")
            if isinstance(delta, str) and delta:
                yield delta


@dataclass(frozen=True)
class CodexToolCall:
    """One tool call collected from a Codex SSE stream.

    Codex emits ``function_call`` items in two places:

    * As ``response.output_item.added`` with ``item.type == "function_call"``
      — carries ``id``, ``call_id``, ``name``, initial ``arguments``.
    * As ``response.function_call_arguments.delta`` events keyed by
      ``item_id`` — each carries an ``arguments`` JSON fragment to
      append.

    ``id`` is Codex's internal item id (``fc-...``). ``call_id`` is
    the id we echo back in ``function_call_output`` to tie the
    result to the call.
    """

    id: str
    call_id: str
    name: str
    arguments_json: str
    """Raw JSON string. We DON'T parse it here — callers that want
    structured args should ``json.loads`` themselves. Some clients
    pass non-strict JSON (trailing commas, etc.) through to model
    feedback, and we don't want to crash mid-stream on the parse."""


@dataclass(frozen=True)
class CodexCompletion:
    """Terminal result extracted from a ``response.completed`` event."""

    response_id: str | None
    """Codex assigns each response an ID (``resp_...``). Useful for
    chaining with ``previous_response_id`` on a follow-up call."""

    final_text: str
    """Concatenated output text across all message items. For multi-
    item responses (which Codex doesn't seem to do for text-only
    today, but could in future) all output_text blocks are joined."""

    tool_calls: list[CodexToolCall]
    """Tool calls collected during the stream. Empty list if the
    model didn't call any tools."""

    usage: dict[str, Any] | None
    """Token usage as reported by Codex on the final event. Shape is
    OpenAI Responses API style (``input_tokens``, ``output_tokens``,
    ``total_tokens``, ...). May be ``None`` if the gateway omits it."""

    raw_response: dict[str, Any]
    """The full ``data.response`` object from the ``response.completed``
    event. Exposed so callers can introspect fields we haven't
    promoted to typed accessors (``service_tier``, ``model``, etc.)."""


def first_stop_match(text: str, stops: list[str]) -> int | None:
    """Return the lowest index in ``text`` where any of ``stops``
    first appears, or ``None`` if none match.

    Empty strings in ``stops`` are ignored — they'd otherwise match
    at index 0 and break the caller's expectation that ``None`` ==
    "no stop seen yet".
    """
    best: int | None = None
    for s in stops:
        if not s:
            continue
        idx = text.find(s)
        if idx >= 0 and (best is None or idx < best):
            best = idx
    return best


def consume_events(
    events: Iterable[SseEvent],
    *,
    stop_sequences: list[str] | None = None,
) -> CodexCompletion:
    """Drain the SSE event stream and return the final completion.

    Tracks text deltas to build a final concatenated text and collects
    tool calls from ``response.output_item.added`` (initial frame) +
    ``response.function_call_arguments.delta`` (argument fragments).

    Raises :class:`CodexResponseError` if a ``response.error`` event
    arrives or if the stream ends without a ``response.completed``.

    **Client-side stop sequences**: when ``stop_sequences`` is set, the
    accumulated output text is checked after each delta. On match, the
    text is truncated at the first match, event consumption stops
    early, and a synthesized :class:`CodexCompletion` is returned with
    ``response_metadata`` indicating an early stop. Note: Codex's
    ``/codex/responses`` endpoint rejects the ``stop`` parameter, so
    this is the *only* way to stop early — Codex may continue
    generating tokens server-side until the SSE connection closes.
    """
    response_id: str | None = None
    text_parts: list[str] = []
    completed_response: dict[str, Any] | None = None
    early_stop_text: str | None = None
    # In-progress tool calls, keyed by Codex's item_id (``fc-...``).
    # Each entry tracks call_id + name (from output_item.added) and
    # accumulates argument fragments (from arguments.delta events).
    pending_tool_calls: dict[str, dict[str, Any]] = {}

    for evt in events:
        if evt.event == "response.created":
            resp = evt.data.get("response", {})
            if isinstance(resp, dict):
                response_id = resp.get("id") or response_id
        elif evt.event == "response.output_text.delta":
            delta = evt.data.get("delta")
            if isinstance(delta, str):
                text_parts.append(delta)
                if stop_sequences:
                    accumulated = "".join(text_parts)
                    match_idx = first_stop_match(accumulated, stop_sequences)
                    if match_idx is not None:
                        early_stop_text = accumulated[:match_idx]
                        break
        elif evt.event == "response.output_item.added":
            item = evt.data.get("item") or {}
            if isinstance(item, dict) and item.get("type") == "function_call":
                item_id = item.get("id")
                if isinstance(item_id, str):
                    pending_tool_calls[item_id] = {
                        "id": item_id,
                        "call_id": item.get("call_id") or "",
                        "name": item.get("name") or "",
                        "arguments": item.get("arguments") or "",
                    }
        elif evt.event == "response.function_call_arguments.delta":
            item_id = evt.data.get("item_id")
            delta = evt.data.get("delta")
            if (
                isinstance(item_id, str)
                and isinstance(delta, str)
                and item_id in pending_tool_calls
            ):
                pending_tool_calls[item_id]["arguments"] += delta
        elif evt.event == "response.completed":
            resp = evt.data.get("response", {})
            if isinstance(resp, dict):
                completed_response = resp
                response_id = resp.get("id") or response_id
        elif evt.event == "response.error" or evt.event == "error":
            err = evt.data.get("error") or evt.data
            raise CodexResponseError(
                message=(
                    err.get("message") if isinstance(err, dict) else str(err)
                )
                or "Codex returned an error mid-stream",
                code=err.get("code") if isinstance(err, dict) else None,
                type=err.get("type") if isinstance(err, dict) else None,
                raw=evt.data,
            )

    # Early-stop path: client-side stop sequence matched. Return a
    # synthesized completion. ``raw_response`` carries an explicit
    # ``stopped_at_client`` marker so callers can distinguish it from a
    # natural completion.
    if early_stop_text is not None:
        tool_calls = [
            CodexToolCall(
                id=tc["id"],
                call_id=tc["call_id"],
                name=tc["name"],
                arguments_json=tc["arguments"],
            )
            for tc in pending_tool_calls.values()
        ]
        return CodexCompletion(
            response_id=response_id,
            final_text=early_stop_text,
            tool_calls=tool_calls,
            usage=None,
            raw_response={
                "status": "stopped_at_client",
                "stopped_at_client": True,
            },
        )

    if completed_response is None:
        raise CodexResponseError(
            message=(
                "Codex SSE stream ended without a response.completed event"
            ),
            code=None,
            type="stream_terminated_early",
            raw={"partial_text": "".join(text_parts)},
        )

    # Tool calls from the stream first; fall back to walking the
    # ``output`` array on the completed event (defensive — covers
    # responses where the gateway elided per-event signals).
    tool_calls = [
        CodexToolCall(
            id=tc["id"],
            call_id=tc["call_id"],
            name=tc["name"],
            arguments_json=tc["arguments"],
        )
        for tc in pending_tool_calls.values()
    ]
    if not tool_calls:
        tool_calls = _extract_tool_calls_from_output(completed_response)

    final_text = "".join(text_parts) or _extract_final_text(completed_response)
    usage = completed_response.get("usage")
    return CodexCompletion(
        response_id=response_id,
        final_text=final_text,
        tool_calls=tool_calls,
        usage=usage if isinstance(usage, dict) else None,
        raw_response=completed_response,
    )


def _extract_tool_calls_from_output(
    response_obj: dict[str, Any],
) -> list[CodexToolCall]:
    """Fallback for when the SSE stream didn't carry per-event signals.
    Walk the ``output`` array on the completed response and pull out
    any ``function_call`` items."""
    out: list[CodexToolCall] = []
    for item in response_obj.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "function_call":
            continue
        out.append(CodexToolCall(
            id=str(item.get("id") or ""),
            call_id=str(item.get("call_id") or ""),
            name=str(item.get("name") or ""),
            arguments_json=str(item.get("arguments") or ""),
        ))
    return out


def _extract_final_text(response_obj: dict[str, Any]) -> str:
    """Fallback for when we didn't collect deltas. Walk the
    ``output`` array and concatenate all ``output_text`` block texts.
    """
    parts: list[str] = []
    for item in response_obj.get("output") or []:
        if not isinstance(item, dict):
            continue
        for block in item.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "output_text":
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
    return "".join(parts)


# ─── Errors ─────────────────────────────────────────────────────────────


class CodexResponseError(RuntimeError):
    """Codex returned an error — either as an HTTP non-2xx envelope
    or as a ``response.error`` SSE event mid-stream."""

    def __init__(
        self,
        *,
        message: str,
        code: str | None = None,
        type: str | None = None,
        raw: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.type = type
        self.raw = raw

    def __repr__(self) -> str:
        bits = [f"message={self.message!r}"]
        if self.code:
            bits.append(f"code={self.code!r}")
        if self.type:
            bits.append(f"type={self.type!r}")
        return f"CodexResponseError({', '.join(bits)})"


def parse_error_body(body_bytes: bytes) -> CodexResponseError:
    """Parse the JSON body returned with a non-2xx HTTP envelope.

    Codex's two error shapes:

    1. ``{"detail": "Some message"}`` — short validator errors.
    2. ``{"error": {"message": ..., "type": ..., "code": ...}}`` —
       OpenAI-style structured errors.

    Returns a :class:`CodexResponseError` either way.
    """
    try:
        parsed = json.loads(body_bytes.decode("utf-8", errors="replace"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return CodexResponseError(
            message=body_bytes.decode("utf-8", errors="replace")[:500]
            or "<empty response body>",
            raw=body_bytes,
        )
    if not isinstance(parsed, dict):
        return CodexResponseError(message=str(parsed)[:500], raw=parsed)
    if "error" in parsed and isinstance(parsed["error"], dict):
        err = parsed["error"]
        return CodexResponseError(
            message=str(err.get("message") or "Codex error"),
            code=err.get("code"),
            type=err.get("type"),
            raw=parsed,
        )
    if "detail" in parsed:
        return CodexResponseError(
            message=str(parsed["detail"]),
            type="validation_error",
            raw=parsed,
        )
    return CodexResponseError(
        message=json.dumps(parsed)[:500],
        raw=parsed,
    )
