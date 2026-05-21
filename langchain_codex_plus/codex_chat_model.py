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

Not yet supported (planned follow-ups):

* **Tool calling** — Codex Responses API supports tools, but mapping
  LangChain ``BindToolsT`` is deferred.
* **Image / multimodal input** — text only for v0.1.
* **Stop sequences** — Codex Responses API doesn't expose them; the
  ``stop`` argument is currently ignored.
* **Token refresh** — relies on the user re-authing via Codex CLI.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import ConfigDict, Field, PrivateAttr

from langchain_codex_plus.codex_auth import (
    CODEX_API_BASE,
    CodexAuth,
    CodexAuthNotFoundError,
    load_codex_auth,
)
from langchain_codex_plus.codex_protocol import (
    CodexCompletion,
    CodexResponseError,
    SseEvent,
    build_request_body,
    consume_events,
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

    # ─── Auth ───────────────────────────────────────────────────────────

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

    def _build_body(self, messages: Sequence[BaseMessage]) -> dict[str, Any]:
        return build_request_body(
            messages,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            instructions_override=self.instructions,
            store=self.store,
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
        return AIMessage(
            content=completion.final_text,
            response_metadata=response_metadata,
            usage_metadata=usage_metadata,
            id=completion.response_id,
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
            logger.debug("Codex Responses API ignores stop sequences; %r dropped", stop)
        auth = self._resolve_auth()
        body = self._build_body(messages)
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = self._post_stream_sync(client, auth, body)
            try:
                self._raise_for_http_error(response)
                self._fire_rate_limit_callback(response.headers)
                completion = self._consume_sync(response, run_manager)
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
    ) -> CodexCompletion:
        events_iter = parse_sse_stream(response.iter_lines())
        # Tap text deltas for run_manager.on_llm_new_token so LangChain
        # callback handlers (LangSmith, custom loggers) see streaming
        # tokens even in the non-streaming _generate path.
        if run_manager is None:
            return consume_events(events_iter)
        return consume_events(_tap_text_deltas_sync(events_iter, run_manager))

    # ─── Sync streaming: _stream ────────────────────────────────────────

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if stop:
            logger.debug("Codex Responses API ignores stop sequences; %r dropped", stop)
        auth = self._resolve_auth()
        body = self._build_body(messages)
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = self._post_stream_sync(client, auth, body)
            try:
                self._raise_for_http_error(response)
                self._fire_rate_limit_callback(response.headers)
                yield from _yield_chunks_sync(
                    parse_sse_stream(response.iter_lines()),
                    run_manager=run_manager,
                )
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
            logger.debug("Codex Responses API ignores stop sequences; %r dropped", stop)
        auth = self._resolve_auth()
        body = self._build_body(messages)
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await self._post_stream_async(client, auth, body)
            try:
                await self._araise_for_http_error(response)
                self._fire_rate_limit_callback(response.headers)
                completion = await self._consume_async(response, run_manager)
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
            for ev in events:
                if ev.event == "response.output_text.delta":
                    delta = ev.data.get("delta")
                    if isinstance(delta, str) and delta:
                        await run_manager.on_llm_new_token(delta)
        return consume_events(events)

    # ─── Async streaming: _astream ──────────────────────────────────────

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if stop:
            logger.debug("Codex Responses API ignores stop sequences; %r dropped", stop)
        auth = self._resolve_auth()
        body = self._build_body(messages)
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await self._post_stream_async(client, auth, body)
            try:
                await self._araise_for_http_error(response)
                self._fire_rate_limit_callback(response.headers)
                lines: list[str] = []
                async for line in response.aiter_lines():
                    lines.append(line)
                events = parse_sse_stream(lines)
                # Stream chunks; we can't easily interleave SSE
                # parsing with awaiting on_llm_new_token because
                # parse_sse_stream is a sync generator, but we can
                # still call the async callback for each chunk.
                response_id: str | None = None
                last_usage: dict[str, Any] | None = None
                for ev in events:
                    if ev.event == "response.created":
                        resp = ev.data.get("response") or {}
                        if isinstance(resp, dict):
                            response_id = resp.get("id")
                    elif ev.event == "response.output_text.delta":
                        delta = ev.data.get("delta")
                        if isinstance(delta, str) and delta:
                            if run_manager is not None:
                                await run_manager.on_llm_new_token(delta)
                            yield ChatGenerationChunk(
                                message=AIMessageChunk(
                                    content=delta, id=response_id
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
                # Final chunk carries usage metadata.
                if last_usage is not None:
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
) -> Iterator[ChatGenerationChunk]:
    """Sync version of the chunk-emitting loop used in ``_stream``.

    Mirrors the inner body of ``_astream`` — kept separate so the
    sync and async paths stay readable.
    """
    response_id: str | None = None
    last_usage: dict[str, Any] | None = None
    for ev in events:
        if ev.event == "response.created":
            resp = ev.data.get("response") or {}
            if isinstance(resp, dict):
                response_id = resp.get("id")
        elif ev.event == "response.output_text.delta":
            delta = ev.data.get("delta")
            if isinstance(delta, str) and delta:
                if run_manager is not None:
                    run_manager.on_llm_new_token(delta)
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content=delta, id=response_id)
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
