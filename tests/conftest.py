"""Shared pytest fixtures + helpers for the test suite.

Pytest auto-discovers ``conftest.py`` modules at every level — fixtures
defined here are available to all test files without explicit imports.
The transport helpers are plain classes / functions but are colocated
here because every test file that touches the chat model needs them.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import httpx
import pytest

from langchain_codex_plus import ChatCodexPlus

# ─── Auth fixture ──────────────────────────────────────────────────────


def _write_auth_json(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "auth_mode": "chatgpt",
        "tokens": {
            "access_token": "atk-test",
            "id_token": "jwt-test",
            "refresh_token": "rtk-test",
            "account_id": "test-account",
        },
        "last_refresh": "2026-05-21T01:02:18.121084Z",
    }))


@pytest.fixture
def auth_file(tmp_path):
    p = tmp_path / "auth.json"
    _write_auth_json(p)
    return p


# ─── SSE body builder ──────────────────────────────────────────────────


def _sse_bytes(events: Iterable[tuple[str, dict[str, Any]]]) -> bytes:
    """Build an SSE byte body from (event_name, data_dict) pairs."""
    out = []
    for name, data in events:
        out.append(f"event: {name}")
        out.append(f"data: {json.dumps(data)}")
        out.append("")
    return ("\n".join(out) + "\n").encode("utf-8")


# ─── Mock transports ───────────────────────────────────────────────────


class _CaptureTransport(httpx.BaseTransport):
    """Mock httpx transport that returns a canned SSE body. Captures
    the most recent outgoing request for assertions."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.body = body
        self.headers = headers or {}
        self.last_request: httpx.Request | None = None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.last_request = request
        _ = request.read()
        return httpx.Response(
            status_code=self.status_code,
            headers=self.headers,
            content=self.body,
            request=request,
        )


class _AsyncCaptureTransport(httpx.AsyncBaseTransport):
    """Async sibling of ``_CaptureTransport``."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.body = body
        self.headers = headers or {}
        self.last_request: httpx.Request | None = None

    async def handle_async_request(
        self, request: httpx.Request
    ) -> httpx.Response:
        self.last_request = request
        _ = await request.aread()
        return httpx.Response(
            status_code=self.status_code,
            headers=self.headers,
            content=self.body,
            request=request,
        )


# ─── _make_llm: build a ChatCodexPlus with a mock transport ────────────


def _make_llm(
    auth_file_path,
    *,
    transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
    **kwargs: Any,
) -> ChatCodexPlus:
    """Build a ``ChatCodexPlus`` whose HTTP calls go through ``transport``
    instead of the real network.

    Monkey-patches the model's ``_post_stream_sync`` / ``_post_stream_async``
    helpers to construct an ``httpx.Client`` / ``AsyncClient`` bound to
    the given transport. Keeps the chat model unchanged in production
    while letting tests inject canned responses.
    """
    llm = ChatCodexPlus(auth_file_path=auth_file_path, **kwargs)
    if transport is None:
        return llm

    import types

    if isinstance(transport, httpx.BaseTransport):
        def _post_sync(self, client, auth, body, _t=transport):
            c2 = httpx.Client(transport=_t, timeout=self.timeout_seconds)
            req = c2.build_request(
                "POST",
                self._request_url(),
                headers=self._request_headers(auth),
                json=body,
            )
            return c2.send(req, stream=True)

        llm._post_stream_sync = types.MethodType(_post_sync, llm)
    if isinstance(transport, httpx.AsyncBaseTransport):
        async def _post_async(self, client, auth, body, _t=transport):
            c2 = httpx.AsyncClient(
                transport=_t, timeout=self.timeout_seconds
            )
            req = c2.build_request(
                "POST",
                self._request_url(),
                headers=self._request_headers(auth),
                json=body,
            )
            return await c2.send(req, stream=True)

        llm._post_stream_async = types.MethodType(_post_async, llm)
    return llm
