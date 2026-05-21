"""LangChain ``BaseChatModel`` for OpenAI's ChatGPT-account Codex
subscription protocol (``/codex/responses`` SSE).

Construction is zero-config in the common case::

    from langchain_codex_plus import ChatCodexPlus

    llm = ChatCodexPlus()
    llm.invoke("Say ok.")  # reads ~/.codex/auth.json, calls Codex Plus

Auth flow: the model reads ``$CODEX_HOME/auth.json`` (populated by
``codex login``) on each call. If the file is missing, instantiation
raises :class:`CodexAuthNotFoundError`. Token refresh is out of scope
for v0.1 — when the access_token expires, calls 401 and the caller
should run ``codex auth refresh`` or restart ``codex login``.

Streaming: every Codex call uses ``stream: true`` at the wire level
(the API requires it). ``_generate`` collects deltas internally and
returns a single :class:`ChatResult`; ``_stream`` / ``_astream``
yields :class:`ChatGenerationChunk` per delta.

Rate-limit hook: every successful response carries ``x-codex-*``
headers (5h + 7d windows, plan tier, credits state). If
``rate_limit_callback`` is set, the parsed snapshot is passed to it.
Exceptions in the callback are caught and logged — they never break
the response path.

Tool calling: use :meth:`ChatCodexPlus.bind_tools` exactly like
``ChatOpenAI.bind_tools``. Tools are serialized into the Codex
Responses API ``tools`` field. Returned :class:`AIMessage` carries
``tool_calls`` populated from the streamed ``function_call`` items.
Tool results go back as :class:`langchain_core.messages.ToolMessage`
with ``tool_call_id`` matching the original ``call_id`` — the
protocol layer transparently serializes them as ``function_call_output``
input entries.

Multimodal: ``HumanMessage`` content can be a list mixing text and
image blocks; the protocol layer maps them to Codex's ``input_text``
+ ``input_image`` content blocks. Accepted image-block shapes:

* ``{"type": "image_url", "image_url": "https://..."}``
* ``{"type": "image_url", "image_url": {"url": ..., "detail": ...}}``
* ``{"type": "image", "source_type": "url", "url": ...}``
* ``{"type": "image", "source_type": "base64", "data": ...,
  "mime_type": "image/png"}`` (encoded as a ``data:`` URL)

Stop sequences: Codex's ``/codex/responses`` rejects the ``stop``
parameter (400 ``Unsupported parameter: stop``), so this package
implements them client-side. The streaming paths use a buffered
matcher that holds back ``max(len(s) for s in stop) - 1`` trailing
characters until they can be ruled out, so a stop seq split across
chunks (the common tokenization case) still truncates cleanly. On
match, the message is truncated, the SSE connection is closed early,
and ``response_metadata['finish_reason']`` is ``'stopped_at_client'``.
Note: Codex may keep generating tokens server-side until we close —
on a subscription account this is a window-utilization cost only,
not a per-token billing cost.

OAuth refresh: when ``auto_refresh=True`` (the default), a 401
response triggers a refresh against ``auth.openai.com/oauth/token``
using the stored ``refresh_token``. New tokens are written back to
``auth.json`` atomically (``.tmp``-then-rename) and the original
call is retried once. Permanent failures (refresh_token expired /
revoked / already-used) surface as :class:`CodexAuthRefreshError`
with ``permanent=True`` — the operator must re-run ``codex login``.
"""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages.tool import ToolCall, ToolCallChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function
from pydantic import ConfigDict, Field, PrivateAttr

from langchain_codex_plus.codex_auth import (
    CODEX_API_BASE,
    CodexAuth,
    CodexAuthNotFoundError,
    arefresh_codex_auth,
    load_codex_auth,
    refresh_codex_auth,
)
from langchain_codex_plus.codex_protocol import (
    CodexCompletion,
    CodexResponseError,
    SseEvent,
    ToolChoice,
    build_request_body,
    consume_events,
    first_stop_match,
    parse_error_body,
    parse_sse_stream,
)
from langchain_codex_plus.rate_limits import (
    CodexRateLimits,
    parse_codex_rate_limits,
)

logger = logging.getLogger(__name__)


_DEFAULT_USER_AGENT = "langchain-codex-plus/0.0.1"
_DEFAULT_ORIGINATOR = "langchain_codex_plus"
"""Originator header value. Codex's default is ``codex_cli_rs``; we
distinguish ourselves so server-side telemetry can tell our traffic
apart from real Codex CLI traffic. Override via the ``originator``
field if you need a more specific label for your application."""

_DEFAULT_CLIENT_VERSION = "0.99.0"
"""Sent as ``?client_version=<v>`` on every request. The gateway is
lenient about exact value; we send a recent-ish stub. Override if
Codex starts gating features by client version."""

_DEFAULT_TIMEOUT_SECONDS = 120.0


class ChatCodexPlus(BaseChatModel):
    """LangChain chat model wrapping the Codex Plus subscription API.

    See module docstring for protocol details and known limitations.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ─── User-facing fields ────────────────────────────────────────────

    model: str = Field(
        default="gpt-5.4",
        description=(
            "Codex model slug (e.g. 'gpt-5.4'). Available slugs come "
            "from GET /codex/models on a real Plus account."
        ),
    )
    instructions: str | None = Field(
        default=None,
        description=(
            "Override the ``instructions`` (system prompt) field "
            "regardless of any SystemMessages in the input. If unset, "
            "SystemMessages are concatenated into instructions."
        ),
    )
    reasoning_effort: str = Field(
        default="none",
        description=(
            "Reasoning effort level. For gpt-5.4: "
            "'none' | 'low' | 'medium' | 'high' | 'xhigh'. "
            "Cheaper to 'none' for short chat-style calls."
        ),
    )
    auth_file_path: Path | None = Field(
        default=None,
        description=(
            "Path to Codex CLI ``auth.json``. Defaults to "
            "``$CODEX_HOME/auth.json`` (``~/.codex/auth.json``)."
        ),
    )
    api_base: str = Field(
        default=CODEX_API_BASE,
        description=(
            "Codex API base URL. Override only for testing against "
            "a mock server."
        ),
    )
    client_version: str = Field(
        default=_DEFAULT_CLIENT_VERSION,
        description="Value sent as ?client_version=<v> on each request.",
    )
    originator: str = Field(
        default=_DEFAULT_ORIGINATOR,
        description="Value sent as the 'originator' header.",
    )
    user_agent: str = Field(
        default=_DEFAULT_USER_AGENT,
        description="Value sent as the 'User-Agent' header.",
    )
    timeout_seconds: float = Field(
        default=_DEFAULT_TIMEOUT_SECONDS,
        description="Total request timeout (connect + read).",
    )
    extra_request_fields: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Extra JSON fields merged into the request body "
            "(escape hatch for ``service_tier``, ``temperature``, etc.)."
        ),
    )
    extra_headers: dict[str, str] | None = Field(
        default=None,
        description="Extra HTTP headers sent with each request.",
    )
    rate_limit_callback: Callable[[CodexRateLimits], None] | None = Field(
        default=None,
        description=(
            "Called with the parsed rate-limit snapshot after each "
            "successful response. Exceptions are caught and logged."
        ),
    )
    store: bool = Field(
        default=False,
        description=(
            "Codex 'store' flag. If True, the server retains the "
            "response on the user's ChatGPT account for replay/UI. "
            "Default False (don't bloat history)."
        ),
    )
    auto_refresh: bool = Field(
        default=True,
        description=(
            "When True (default), a 401 response triggers an OAuth "
            "refresh against the ChatGPT token endpoint, after which "
            "the call is retried once. Permanent refresh failures "
            "(expired/revoked refresh_token) surface as a "
            "``CodexAuthRefreshError``."
        ),
    )
    tools: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Tools to send in the request body, in Codex Responses "
            "API shape: ``[{'type': 'function', 'name', 'description', "
            "'parameters': <JSON schema>}]``. Prefer :meth:`bind_tools` "
            "over setting this directly — it handles LangChain "
            "``BaseTool`` conversion."
        ),
    )
    tool_choice: ToolChoice | None = Field(
        default=None,
        description=(
            "Optional tool-choice directive. ``'auto'`` (model decides), "
            "``'none'``, ``'required'``, or "
            "``{'type': 'function', 'name': '...'}`` to force a specific "
            "tool. Only included in the request body when non-None."
        ),
    )

    # ─── Private state ──────────────────────────────────────────────────

    _auth: CodexAuth | None = PrivateAttr(default=None)
    """Lazily loaded on first call; re-read after a 401."""

    # ─── Identity ───────────────────────────────────────────────────────

    @property
    def _llm_type(self) -> str:
        return "codex-plus"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "api_base": self.api_base,
        }

    # ─── Tool binding ───────────────────────────────────────────────────

    def bind_tools(
        self,
        tools: Sequence[
            dict[str, Any] | type | Callable[..., Any] | BaseTool
        ],
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Return a copy of this chat model with ``tools`` attached.

        Accepts the same input shapes as ``ChatOpenAI.bind_tools``:

        * LangChain ``BaseTool`` instances
        * Pydantic models (used as the args schema)
        * Plain Python callables (signature inferred via inspect)
        * Pre-formed OpenAI-function dicts ``{"name", "description",
          "parameters"}`` — passed through unchanged

        Each tool is normalized to the Codex Responses API shape
        ``{"type": "function", "name": "...", "description": "...",
        "parameters": <JSON schema>}`` (note: name/description/parameters
        are flattened to the top level — different from the Chat
        Completions API's ``{"type": "function", "function": {...}}``
        nesting).

        ``tool_choice`` directives match the protocol module's
        :class:`ToolChoice` type.
        """
        codex_tools: list[dict[str, Any]] = []
        for t in tools:
            if (
                isinstance(t, dict)
                and t.get("type") == "function"
                and "name" in t
            ):
                # Already Codex shape (flat); pass through.
                codex_tools.append(t)
                continue
            # convert_to_openai_function returns the inner
            # ``{name, description, parameters}`` dict — exactly the
            # shape Codex expects at the top level under ``type:
            # function``.
            fn_dict = convert_to_openai_function(t)
            if (
                isinstance(fn_dict, dict)
                and fn_dict.get("type") == "function"
                and "function" in fn_dict
            ):
                # Some versions of langchain-core return the nested
                # Chat-Completions shape; unwrap it.
                fn_dict = fn_dict["function"]
            codex_tools.append({"type": "function", **fn_dict})
        return self.bind(
            tools=codex_tools,
            tool_choice=tool_choice if tool_choice is not None else self.tool_choice,
            **kwargs,
        )

    # ─── Auth ───────────────────────────────────────────────────────────

    def _refresh_auth_sync(self) -> CodexAuth:
        """Refresh the OAuth tokens via the ChatGPT token endpoint
        and update the cached auth. Raises :class:`CodexAuthRefreshError`
        on failure — permanent failures (expired/reused refresh_token)
        require the operator to re-run ``codex login``."""
        current = self._resolve_auth()
        new_auth = refresh_codex_auth(current, path=self.auth_file_path)
        self._auth = new_auth
        return new_auth

    async def _refresh_auth_async(self) -> CodexAuth:
        """Async sibling of :meth:`_refresh_auth_sync`."""
        current = self._resolve_auth()
        new_auth = await arefresh_codex_auth(
            current, path=self.auth_file_path
        )
        self._auth = new_auth
        return new_auth

    def _resolve_auth(self, *, force_reload: bool = False) -> CodexAuth:
        """Load ``auth.json`` (cached). Raises if no OAuth bundle is
        available — strict mode, since instantiating this model is an
        explicit declaration of intent to call Codex Plus."""
        if self._auth is not None and not force_reload:
            return self._auth
        path = self.auth_file_path
        auth = load_codex_auth(path, strict=True)
        if auth is None:
            # ``strict=True`` should have raised; defensive.
            raise CodexAuthNotFoundError(
                "Codex Plus auth could not be loaded. "
                "Run `codex login` and try again."
            )
        self._auth = auth
        return auth

    # ─── HTTP plumbing ──────────────────────────────────────────────────

    def _request_url(self) -> str:
        return (
            f"{self.api_base.rstrip('/')}/codex/responses"
            f"?client_version={self.client_version}"
        )

    def _request_headers(self, auth: CodexAuth) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {auth.access_token}",
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "User-Agent": self.user_agent,
            "originator": self.originator,
        }
        if self.extra_headers:
            headers.update(self.extra_headers)
        return headers

    def _build_body(
        self,
        messages: Sequence[BaseMessage],
        *,
        tools_override: list[dict[str, Any]] | None = None,
        tool_choice_override: ToolChoice | None = None,
    ) -> dict[str, Any]:
        """Build the request body. Per-call overrides (from
        :meth:`bind_tools`) take precedence over the instance fields.
        Pass ``None`` explicitly to mean "use the instance default";
        the call sites pass through ``kwargs.get("tools")`` so a
        bound runnable wins."""
        return build_request_body(
            messages,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            instructions_override=self.instructions,
            store=self.store,
            tools=tools_override if tools_override is not None else self.tools,
            tool_choice=(
                tool_choice_override
                if tool_choice_override is not None
                else self.tool_choice
            ),
            extra=self.extra_request_fields,
        )

    def _fire_rate_limit_callback(
        self, headers: httpx.Headers | dict[str, str]
    ) -> CodexRateLimits | None:
        """Parse ``x-codex-*`` headers and dispatch to callback. Never
        raises — a broken callback shouldn't break the response."""
        try:
            rl = parse_codex_rate_limits(dict(headers))
        except Exception:
            logger.exception("rate-limit header parsing failed")
            return None
        if rl is None or self.rate_limit_callback is None:
            return rl
        try:
            self.rate_limit_callback(rl)
        except Exception:
            logger.exception("rate_limit_callback raised")
        return rl

    @staticmethod
    def _completion_to_ai_message(completion: CodexCompletion) -> AIMessage:
        usage = completion.usage or {}
        response_metadata: dict[str, Any] = {
            "model_name": completion.raw_response.get("model"),
            "finish_reason": completion.raw_response.get("status"),
            "response_id": completion.response_id,
        }
        # LangChain's standard usage_metadata shape uses
        # ``input_tokens`` / ``output_tokens`` / ``total_tokens`` —
        # the same names Codex uses, so passthrough is direct.
        usage_metadata: dict[str, int] | None = None
        if usage:
            usage_metadata = {
                k: int(v)
                for k, v in usage.items()
                if k in {"input_tokens", "output_tokens", "total_tokens"}
                and isinstance(v, (int, float))
            } or None
        # Convert collected Codex tool calls to LangChain ToolCall
        # shape. ``args`` is parsed from the JSON string; if parsing
        # fails (malformed model output), surface raw text in
        # ``invalid_tool_calls`` so the caller can see what happened
        # rather than silently dropping the call.
        tool_calls: list[ToolCall] = []
        invalid_tool_calls: list[dict[str, Any]] = []
        for tc in completion.tool_calls:
            try:
                parsed_args = (
                    json.loads(tc.arguments_json) if tc.arguments_json else {}
                )
            except json.JSONDecodeError as exc:
                invalid_tool_calls.append({
                    "id": tc.call_id,
                    "name": tc.name,
                    "args": tc.arguments_json,
                    "error": str(exc),
                    "type": "invalid_tool_call",
                })
                continue
            if not isinstance(parsed_args, dict):
                parsed_args = {"_value": parsed_args}
            tool_calls.append(ToolCall(
                name=tc.name,
                args=parsed_args,
                id=tc.call_id,
                type="tool_call",
            ))
        return AIMessage(
            content=completion.final_text,
            response_metadata=response_metadata,
            usage_metadata=usage_metadata,
            id=completion.response_id,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
        )

    # ─── Sync sync path: _generate ──────────────────────────────────────

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if stop:
            logger.debug(
                "Codex /codex/responses doesn't accept ``stop``; matching "
                "client-side on %r. Codex may keep generating tokens until "
                "we close the SSE connection.", stop
            )
        auth = self._resolve_auth()
        body = self._build_body(
            messages,
            tools_override=kwargs.get("tools"),
            tool_choice_override=kwargs.get("tool_choice"),
        )
        max_attempts = 2 if self.auto_refresh else 1
        with httpx.Client(timeout=self.timeout_seconds) as client:
            for attempt in range(max_attempts):
                response = self._post_stream_sync(client, auth, body)
                try:
                    if (
                        response.status_code == 401
                        and attempt < max_attempts - 1
                    ):
                        # Drain + close before triggering refresh so
                        # the connection returns cleanly to the pool.
                        response.read()
                        response.close()
                        auth = self._refresh_auth_sync()
                        continue
                    self._raise_for_http_error(response)
                    self._fire_rate_limit_callback(response.headers)
                    completion = self._consume_sync(
                        response, run_manager, stop=stop
                    )
                    break
                finally:
                    response.close()
        message = self._completion_to_ai_message(completion)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _post_stream_sync(
        self,
        client: httpx.Client,
        auth: CodexAuth,
        body: dict[str, Any],
    ) -> httpx.Response:
        """Open a streaming POST. Caller is responsible for closing.
        Split out so tests can mock the transport layer."""
        req = client.build_request(
            "POST",
            self._request_url(),
            headers=self._request_headers(auth),
            json=body,
        )
        return client.send(req, stream=True)

    def _raise_for_http_error(self, response: httpx.Response) -> None:
        """If the HTTP envelope is non-2xx, read the body, parse the
        error, and raise. Reads the body so the connection can close
        cleanly."""
        if 200 <= response.status_code < 300:
            return
        body_bytes = response.read()
        err = parse_error_body(body_bytes)
        # Re-raise with status code prepended so callers can pattern-
        # match on it.
        raise CodexResponseError(
            message=f"HTTP {response.status_code}: {err.message}",
            code=err.code,
            type=err.type,
            raw=err.raw,
        )

    def _consume_sync(
        self,
        response: httpx.Response,
        run_manager: CallbackManagerForLLMRun | None,
        *,
        stop: list[str] | None = None,
    ) -> CodexCompletion:
        events_iter = parse_sse_stream(response.iter_lines())
        # Tap text deltas for run_manager.on_llm_new_token so LangChain
        # callback handlers (LangSmith, custom loggers) see streaming
        # tokens even in the non-streaming _generate path.
        if run_manager is None:
            return consume_events(events_iter, stop_sequences=stop)
        return consume_events(
            _tap_text_deltas_sync(events_iter, run_manager),
            stop_sequences=stop,
        )

    # ─── Sync streaming: _stream ────────────────────────────────────────

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if stop:
            logger.debug(
                "Codex /codex/responses doesn't accept ``stop``; matching "
                "client-side on %r.", stop
            )
        auth = self._resolve_auth()
        body = self._build_body(
            messages,
            tools_override=kwargs.get("tools"),
            tool_choice_override=kwargs.get("tool_choice"),
        )
        max_attempts = 2 if self.auto_refresh else 1
        with httpx.Client(timeout=self.timeout_seconds) as client:
            for attempt in range(max_attempts):
                response = self._post_stream_sync(client, auth, body)
                try:
                    if (
                        response.status_code == 401
                        and attempt < max_attempts - 1
                    ):
                        response.read()
                        response.close()
                        auth = self._refresh_auth_sync()
                        continue
                    self._raise_for_http_error(response)
                    self._fire_rate_limit_callback(response.headers)
                    yield from _yield_chunks_sync(
                        parse_sse_stream(response.iter_lines()),
                        run_manager=run_manager,
                        stop=stop,
                    )
                    return
                finally:
                    response.close()

    # ─── Async path: _agenerate ─────────────────────────────────────────

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if stop:
            logger.debug(
                "Codex /codex/responses doesn't accept ``stop``; matching "
                "client-side on %r.", stop
            )
        auth = self._resolve_auth()
        body = self._build_body(
            messages,
            tools_override=kwargs.get("tools"),
            tool_choice_override=kwargs.get("tool_choice"),
        )
        max_attempts = 2 if self.auto_refresh else 1
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            for attempt in range(max_attempts):
                response = await self._post_stream_async(client, auth, body)
                try:
                    if (
                        response.status_code == 401
                        and attempt < max_attempts - 1
                    ):
                        await response.aread()
                        await response.aclose()
                        auth = await self._refresh_auth_async()
                        continue
                    await self._araise_for_http_error(response)
                    self._fire_rate_limit_callback(response.headers)
                    completion = await self._consume_async(
                        response, run_manager, stop=stop
                    )
                    break
                finally:
                    await response.aclose()
        message = self._completion_to_ai_message(completion)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _post_stream_async(
        self,
        client: httpx.AsyncClient,
        auth: CodexAuth,
        body: dict[str, Any],
    ) -> httpx.Response:
        req = client.build_request(
            "POST",
            self._request_url(),
            headers=self._request_headers(auth),
            json=body,
        )
        return await client.send(req, stream=True)

    async def _araise_for_http_error(self, response: httpx.Response) -> None:
        if 200 <= response.status_code < 300:
            return
        body_bytes = await response.aread()
        err = parse_error_body(body_bytes)
        raise CodexResponseError(
            message=f"HTTP {response.status_code}: {err.message}",
            code=err.code,
            type=err.type,
            raw=err.raw,
        )

    async def _consume_async(
        self,
        response: httpx.Response,
        run_manager: AsyncCallbackManagerForLLMRun | None,
        *,
        stop: list[str] | None = None,
    ) -> CodexCompletion:
        # parse_sse_stream is sync — buffer lines into a list as they
        # arrive, then feed through. For long streams a fully-async
        # SSE parser would be nicer; deferred since each Codex
        # response is bounded.
        lines: list[str] = []
        async for line in response.aiter_lines():
            lines.append(line)
        events = list(parse_sse_stream(lines))
        if run_manager is not None:
            # Walk events, firing on_llm_new_token for each delta —
            # stopping at the first stop-sequence match if ``stop`` is
            # set, so the callback consumer sees the same truncated
            # token stream the caller will see.
            text_so_far = ""
            for ev in events:
                if ev.event != "response.output_text.delta":
                    continue
                delta = ev.data.get("delta")
                if not isinstance(delta, str) or not delta:
                    continue
                if stop:
                    new_text = text_so_far + delta
                    match_idx = first_stop_match(new_text, stop)
                    if match_idx is not None:
                        emit_len = max(0, match_idx - len(text_so_far))
                        emit = delta[:emit_len]
                        if emit:
                            await run_manager.on_llm_new_token(emit)
                        break
                    text_so_far = new_text
                await run_manager.on_llm_new_token(delta)
        return consume_events(events, stop_sequences=stop)

    # ─── Async streaming: _astream ──────────────────────────────────────

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if stop:
            logger.debug(
                "Codex /codex/responses doesn't accept ``stop``; matching "
                "client-side on %r.", stop
            )
        auth = self._resolve_auth()
        body = self._build_body(
            messages,
            tools_override=kwargs.get("tools"),
            tool_choice_override=kwargs.get("tool_choice"),
        )
        max_attempts = 2 if self.auto_refresh else 1
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            for attempt in range(max_attempts):
                response = await self._post_stream_async(client, auth, body)
                if (
                    response.status_code == 401
                    and attempt < max_attempts - 1
                ):
                    await response.aread()
                    await response.aclose()
                    auth = await self._refresh_auth_async()
                    continue
                # Success path: break out of the retry loop and proceed
                # to the SSE-consumption block below. ``response`` is
                # left open for ``aiter_lines`` use.
                break
            try:
                await self._araise_for_http_error(response)
                self._fire_rate_limit_callback(response.headers)
                lines: list[str] = []
                async for line in response.aiter_lines():
                    lines.append(line)
                events = parse_sse_stream(lines)
                # Mirror of ``_yield_chunks_sync`` for the async path —
                # kept inline so callers can ``await
                # run_manager.on_llm_new_token`` on each text delta.
                # Tool-call chunk emission identical to the sync path
                # (see ``_yield_chunks_sync`` for the design rationale).
                response_id: str | None = None
                last_usage: dict[str, Any] | None = None
                tool_call_index: dict[str, int] = {}
                # Same buffered stop-matching algorithm as the sync
                # streaming path (see ``_yield_chunks_sync`` for the
                # full rationale: hold back the trailing chars that
                # could be a stop prefix until we have enough text to
                # rule it out).
                text_so_far = ""
                hold_buffer = ""
                max_stop_len = (
                    max((len(s) for s in stop), default=0) if stop else 0
                )
                stopped_early = False
                for ev in events:
                    if ev.event == "response.created":
                        resp = ev.data.get("response") or {}
                        if isinstance(resp, dict):
                            response_id = resp.get("id")
                    elif ev.event == "response.output_text.delta":
                        delta = ev.data.get("delta")
                        if not isinstance(delta, str) or not delta:
                            continue
                        if not stop:
                            if run_manager is not None:
                                await run_manager.on_llm_new_token(delta)
                            yield ChatGenerationChunk(
                                message=AIMessageChunk(
                                    content=delta, id=response_id
                                )
                            )
                            continue
                        combined = hold_buffer + delta
                        all_text = text_so_far + combined
                        match_idx = first_stop_match(all_text, stop)
                        if match_idx is not None:
                            to_emit = all_text[len(text_so_far):match_idx]
                            if to_emit:
                                if run_manager is not None:
                                    await run_manager.on_llm_new_token(to_emit)
                                yield ChatGenerationChunk(
                                    message=AIMessageChunk(
                                        content=to_emit, id=response_id
                                    )
                                )
                            stopped_early = True
                            break
                        safe_emit_len = len(combined) - (max_stop_len - 1)
                        if safe_emit_len > 0:
                            to_emit = combined[:safe_emit_len]
                            if run_manager is not None:
                                await run_manager.on_llm_new_token(to_emit)
                            yield ChatGenerationChunk(
                                message=AIMessageChunk(
                                    content=to_emit, id=response_id
                                )
                            )
                            text_so_far += to_emit
                            hold_buffer = combined[safe_emit_len:]
                        else:
                            hold_buffer = combined
                    elif ev.event == "response.output_item.added":
                        item = ev.data.get("item") or {}
                        if (
                            isinstance(item, dict)
                            and item.get("type") == "function_call"
                        ):
                            item_id = item.get("id")
                            if isinstance(item_id, str):
                                idx = len(tool_call_index)
                                tool_call_index[item_id] = idx
                                yield ChatGenerationChunk(
                                    message=AIMessageChunk(
                                        content="",
                                        id=response_id,
                                        tool_call_chunks=[ToolCallChunk(
                                            name=item.get("name"),
                                            args=item.get("arguments") or "",
                                            id=item.get("call_id") or "",
                                            index=idx,
                                            type="tool_call_chunk",
                                        )],
                                    )
                                )
                    elif ev.event == "response.function_call_arguments.delta":
                        item_id = ev.data.get("item_id")
                        delta = ev.data.get("delta")
                        if (
                            isinstance(item_id, str)
                            and item_id in tool_call_index
                            and isinstance(delta, str)
                        ):
                            yield ChatGenerationChunk(
                                message=AIMessageChunk(
                                    content="",
                                    id=response_id,
                                    tool_call_chunks=[ToolCallChunk(
                                        name=None,
                                        args=delta,
                                        id=None,
                                        index=tool_call_index[item_id],
                                        type="tool_call_chunk",
                                    )],
                                )
                            )
                    elif ev.event == "response.completed":
                        resp = ev.data.get("response") or {}
                        if isinstance(resp, dict):
                            last_usage = resp.get("usage") if isinstance(
                                resp.get("usage"), dict
                            ) else None
                            response_id = resp.get("id") or response_id
                    elif ev.event in {"response.error", "error"}:
                        err = ev.data.get("error") or ev.data
                        raise CodexResponseError(
                            message=(
                                err.get("message")
                                if isinstance(err, dict)
                                else str(err)
                            )
                            or "Codex returned an error mid-stream",
                            code=(
                                err.get("code")
                                if isinstance(err, dict)
                                else None
                            ),
                            type=(
                                err.get("type")
                                if isinstance(err, dict)
                                else None
                            ),
                            raw=ev.data,
                        )
                # Natural completion: flush any held-back text (we'd
                # buffered it in case it was a stop prefix; the stream
                # ended without ever matching). Then emit the usage
                # chunk. On client-side stop we ``break`` early, so
                # ``stopped_early`` is True and no terminal chunk
                # emits — the right signal to LangChain that this
                # stream ended without natural completion.
                if hold_buffer and not stopped_early:
                    if run_manager is not None:
                        await run_manager.on_llm_new_token(hold_buffer)
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(
                            content=hold_buffer, id=response_id
                        )
                    )
                if last_usage is not None and not stopped_early:
                    usage_metadata = {
                        k: int(v)
                        for k, v in last_usage.items()
                        if k in {
                            "input_tokens",
                            "output_tokens",
                            "total_tokens",
                        }
                        and isinstance(v, (int, float))
                    } or None
                    if usage_metadata:
                        yield ChatGenerationChunk(
                            message=AIMessageChunk(
                                content="",
                                id=response_id,
                                usage_metadata=usage_metadata,
                            )
                        )
            finally:
                await response.aclose()


# ─── Module-private helpers (kept out of class to ease tap testing) ───


def _tap_text_deltas_sync(
    events: Iterator[SseEvent],
    run_manager: CallbackManagerForLLMRun,
) -> Iterator[SseEvent]:
    """Pass-through events while dispatching delta tokens to the
    LangChain callback manager. Used by sync ``_generate`` so
    callback consumers see streaming tokens even in non-streaming
    invocation paths."""
    for ev in events:
        if ev.event == "response.output_text.delta":
            delta = ev.data.get("delta")
            if isinstance(delta, str) and delta:
                run_manager.on_llm_new_token(delta)
        yield ev


def _yield_chunks_sync(
    events: Iterator[SseEvent],
    *,
    run_manager: CallbackManagerForLLMRun | None,
    stop: list[str] | None = None,
) -> Iterator[ChatGenerationChunk]:
    """Sync version of the chunk-emitting loop used in ``_stream``.

    Maps Codex SSE events to LangChain chunks:

    * ``response.output_text.delta`` → text content chunk
    * ``response.output_item.added`` (function_call) → kickoff
      ``ToolCallChunk`` carrying ``name`` + ``id``
    * ``response.function_call_arguments.delta`` → ``ToolCallChunk``
      with incremental ``args`` JSON fragment
    * ``response.completed`` → usage-bearing terminal chunk

    If ``stop`` is provided, the running text is checked after each
    delta; a match truncates the current delta to just-before the
    stop sequence, yields that, and exits early. Codex may keep
    generating server-side until we close the connection.
    """
    response_id: str | None = None
    last_usage: dict[str, Any] | None = None
    # Map Codex item_id → (chunk_index, call_id, name) so we can
    # stamp the same ``index`` on each delta chunk for a given tool
    # call. LangChain merges chunks with matching index to assemble
    # the final ToolCall on the message reducer side.
    tool_call_index: dict[str, int] = {}
    # Buffered stop-sequence matcher: ``text_so_far`` is what we've
    # already YIELDED to the caller; ``hold_buffer`` is text we've
    # received but held back in case it's the prefix of a stop seq
    # we'd otherwise have to un-emit (which streaming doesn't allow).
    text_so_far = ""
    hold_buffer = ""
    max_stop_len = max((len(s) for s in stop), default=0) if stop else 0
    for ev in events:
        if ev.event == "response.created":
            resp = ev.data.get("response") or {}
            if isinstance(resp, dict):
                response_id = resp.get("id")
        elif ev.event == "response.output_text.delta":
            delta = ev.data.get("delta")
            if not isinstance(delta, str) or not delta:
                continue
            if not stop:
                # Fast path: no stop sequences set. Emit eagerly.
                if run_manager is not None:
                    run_manager.on_llm_new_token(delta)
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content=delta, id=response_id)
                )
                continue
            # Stop-aware path. Combine the held buffer with the new
            # delta; check if a stop now appears in the *total* text.
            combined = hold_buffer + delta
            all_text = text_so_far + combined
            match_idx = first_stop_match(all_text, stop)
            if match_idx is not None:
                # Emit any text up to the match that hasn't been
                # yielded yet, then exit. The slice is relative to
                # ``all_text``; we trim by ``text_so_far`` length.
                to_emit = all_text[len(text_so_far):match_idx]
                if to_emit:
                    if run_manager is not None:
                        run_manager.on_llm_new_token(to_emit)
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(
                            content=to_emit, id=response_id
                        )
                    )
                return
            # No match yet. Emit the prefix of ``combined`` that's
            # *guaranteed* not to be the start of any stop seq —
            # i.e., everything except the trailing (max_stop_len - 1)
            # characters. Those stay in ``hold_buffer`` until the
            # next delta arrives.
            safe_emit_len = len(combined) - (max_stop_len - 1)
            if safe_emit_len > 0:
                to_emit = combined[:safe_emit_len]
                if run_manager is not None:
                    run_manager.on_llm_new_token(to_emit)
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=to_emit, id=response_id
                    )
                )
                text_so_far += to_emit
                hold_buffer = combined[safe_emit_len:]
            else:
                hold_buffer = combined
        elif ev.event == "response.output_item.added":
            item = ev.data.get("item") or {}
            if isinstance(item, dict) and item.get("type") == "function_call":
                item_id = item.get("id")
                if isinstance(item_id, str):
                    idx = len(tool_call_index)
                    tool_call_index[item_id] = idx
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(
                            content="",
                            id=response_id,
                            tool_call_chunks=[ToolCallChunk(
                                name=item.get("name"),
                                args=item.get("arguments") or "",
                                id=item.get("call_id") or "",
                                index=idx,
                                type="tool_call_chunk",
                            )],
                        )
                    )
        elif ev.event == "response.function_call_arguments.delta":
            item_id = ev.data.get("item_id")
            delta = ev.data.get("delta")
            if (
                isinstance(item_id, str)
                and item_id in tool_call_index
                and isinstance(delta, str)
            ):
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        id=response_id,
                        tool_call_chunks=[ToolCallChunk(
                            name=None,
                            args=delta,
                            id=None,
                            index=tool_call_index[item_id],
                            type="tool_call_chunk",
                        )],
                    )
                )
        elif ev.event == "response.completed":
            resp = ev.data.get("response") or {}
            if isinstance(resp, dict):
                last_usage = resp.get("usage") if isinstance(
                    resp.get("usage"), dict
                ) else None
                response_id = resp.get("id") or response_id
        elif ev.event in {"response.error", "error"}:
            err = ev.data.get("error") or ev.data
            raise CodexResponseError(
                message=(
                    err.get("message")
                    if isinstance(err, dict)
                    else str(err)
                )
                or "Codex returned an error mid-stream",
                code=err.get("code") if isinstance(err, dict) else None,
                type=err.get("type") if isinstance(err, dict) else None,
                raw=ev.data,
            )
    # Natural completion: flush any text held back for stop-matching
    # before emitting the terminal usage chunk. If we reach here with
    # ``hold_buffer`` non-empty, the stream completed without ever
    # matching a stop sequence — those held characters are real output
    # and must be delivered.
    if hold_buffer:
        if run_manager is not None:
            run_manager.on_llm_new_token(hold_buffer)
        yield ChatGenerationChunk(
            message=AIMessageChunk(content=hold_buffer, id=response_id)
        )
    if last_usage is not None:
        usage_metadata = {
            k: int(v)
            for k, v in last_usage.items()
            if k in {"input_tokens", "output_tokens", "total_tokens"}
            and isinstance(v, (int, float))
        } or None
        if usage_metadata:
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="", id=response_id, usage_metadata=usage_metadata
                )
            )
