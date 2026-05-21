"""OAuth refresh tests.

Covers:

* The ``refresh_codex_auth`` / ``arefresh_codex_auth`` helpers — POST
  shape, response application, error classification, atomic file write.
* The chat model's auto-refresh path: a 401 from Codex triggers a
  refresh against the OAuth token endpoint, then retries the original
  call once.
"""
from __future__ import annotations

import json

import httpx
import pytest
from langchain_core.messages import HumanMessage

from langchain_codex_plus import (
    CODEX_OAUTH_CLIENT_ID,
    CodexAuth,
    CodexAuthRefreshError,
    arefresh_codex_auth,
    load_codex_auth,
    refresh_codex_auth,
)
from langchain_codex_plus.codex_auth import (
    _classify_refresh_error,
    _refresh_endpoint,
)

# ─── _classify_refresh_error ───────────────────────────────────────────


def test_classify_known_permanent_codes():
    """Per ``codex-rs/login/src/auth/manager.rs``, these codes mean
    the refresh token is unusable — operator must re-login."""
    for code in [
        "expired_token",
        "invalid_grant",
        "invalid_token",
        "refresh_token_reused",
        "refresh_token_invalidated",
    ]:
        err = _classify_refresh_error(400, {"error": code})
        assert err.permanent is True, f"{code} should be permanent"
        assert err.code == code


def test_classify_4xx_is_permanent():
    """Generic 4xx (other than 429) is auth-side → permanent."""
    err = _classify_refresh_error(403, {"error": "forbidden"})
    assert err.permanent is True


def test_classify_429_is_transient():
    """Rate limited → retry may work."""
    err = _classify_refresh_error(429, {"error": "rate_limited"})
    assert err.permanent is False


def test_classify_5xx_is_transient():
    err = _classify_refresh_error(503, None)
    assert err.permanent is False


def test_classify_falls_back_to_status_message():
    """Body without an ``error`` field still produces a usable message."""
    err = _classify_refresh_error(502, {"detail": "Bad Gateway"})
    assert "Bad Gateway" in str(err)
    assert err.permanent is False


# ─── refresh_codex_auth: success ───────────────────────────────────────


def _auth_with_refresh_token(token: str = "rtk-old") -> CodexAuth:
    return CodexAuth(
        auth_mode="chatgpt",
        access_token="atk-old",
        id_token="id-old",
        refresh_token=token,
        account_id="acct-1",
        last_refresh=None,
    )


class _OAuthTransport(httpx.BaseTransport):
    """Mock the OAuth token endpoint. Captures the outgoing request
    so tests can verify the body shape Codex expects."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        body: dict | bytes | None = None,
    ) -> None:
        self.status_code = status_code
        if isinstance(body, dict):
            self.body = json.dumps(body).encode()
        elif body is None:
            self.body = b""
        else:
            self.body = body
        self.last_request: httpx.Request | None = None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.last_request = request
        return httpx.Response(
            status_code=self.status_code,
            content=self.body,
            headers={"Content-Type": "application/json"},
            request=request,
        )


class _AsyncOAuthTransport(httpx.AsyncBaseTransport):
    def __init__(
        self,
        *,
        status_code: int = 200,
        body: dict | bytes | None = None,
    ) -> None:
        self.status_code = status_code
        if isinstance(body, dict):
            self.body = json.dumps(body).encode()
        elif body is None:
            self.body = b""
        else:
            self.body = body
        self.last_request: httpx.Request | None = None

    async def handle_async_request(
        self, request: httpx.Request
    ) -> httpx.Response:
        self.last_request = request
        return httpx.Response(
            status_code=self.status_code,
            content=self.body,
            headers={"Content-Type": "application/json"},
            request=request,
        )


def test_refresh_post_body_matches_codex_cli(tmp_path):
    """The request body shape matches what ``codex-rs/login/src/auth/
    manager.rs`` sends (client_id + grant_type + refresh_token)."""
    transport = _OAuthTransport(body={
        "access_token": "atk-new",
        "id_token": "id-new",
        "refresh_token": "rtk-new",
    })
    client = httpx.Client(transport=transport)
    auth = _auth_with_refresh_token("rtk-old")
    refresh_codex_auth(
        auth,
        path=tmp_path / "auth.json",
        write_back=False,
        http_client=client,
    )
    req = transport.last_request
    assert req is not None
    assert req.url == _refresh_endpoint()
    body = json.loads(req.content)
    assert body == {
        "client_id": CODEX_OAUTH_CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": "rtk-old",
    }


def test_refresh_applies_response_tokens(tmp_path):
    transport = _OAuthTransport(body={
        "access_token": "atk-new",
        "id_token": "id-new",
        "refresh_token": "rtk-new",
    })
    client = httpx.Client(transport=transport)
    auth = _auth_with_refresh_token()
    updated = refresh_codex_auth(
        auth,
        path=tmp_path / "auth.json",
        write_back=False,
        http_client=client,
    )
    assert updated.access_token == "atk-new"
    assert updated.id_token == "id-new"
    assert updated.refresh_token == "rtk-new"
    assert updated.last_refresh is not None


def test_refresh_preserves_unchanged_fields(tmp_path):
    """If the server omits ``id_token`` or ``refresh_token``, keep
    the existing values rather than blowing them away."""
    transport = _OAuthTransport(body={"access_token": "atk-new"})
    client = httpx.Client(transport=transport)
    auth = _auth_with_refresh_token("rtk-keep")
    updated = refresh_codex_auth(
        auth,
        path=tmp_path / "auth.json",
        write_back=False,
        http_client=client,
    )
    assert updated.access_token == "atk-new"
    assert updated.id_token == "id-old"  # unchanged
    assert updated.refresh_token == "rtk-keep"  # unchanged


def test_refresh_writes_back_to_auth_json(tmp_path):
    """Default behavior: rewrite ``auth.json`` with the refreshed
    tokens. Atomically (``.tmp``-then-rename) so a crash mid-write
    can't corrupt."""
    path = tmp_path / "auth.json"
    # Seed an existing file with a stray field to verify we preserve it.
    path.write_text(json.dumps({
        "auth_mode": "chatgpt",
        "OPENAI_API_KEY": None,
        "tokens": {
            "access_token": "atk-old",
            "id_token": "id-old",
            "refresh_token": "rtk-old",
            "account_id": "acct-1",
        },
        "last_refresh": "2026-05-21T01:00:00Z",
        "agent_identity": "preserved-field",  # stray field
    }))
    auth = load_codex_auth(path)
    assert auth is not None
    transport = _OAuthTransport(body={"access_token": "atk-new"})
    client = httpx.Client(transport=transport)
    refresh_codex_auth(auth, path=path, http_client=client)
    new = json.loads(path.read_text())
    assert new["tokens"]["access_token"] == "atk-new"
    # Stray field preserved.
    assert new["agent_identity"] == "preserved-field"


# ─── refresh_codex_auth: error paths ───────────────────────────────────


def test_refresh_with_no_refresh_token_raises_permanent():
    """An auth bundle without a refresh_token can't be refreshed —
    permanent error (operator must re-login)."""
    auth = CodexAuth(
        auth_mode="chatgpt",
        access_token="atk",
        id_token=None,
        refresh_token=None,
        account_id=None,
        last_refresh=None,
    )
    with pytest.raises(CodexAuthRefreshError) as exc:
        refresh_codex_auth(auth, write_back=False)
    assert exc.value.permanent is True


def test_refresh_permanent_failure_propagates(tmp_path):
    """OAuth error ``invalid_grant`` (refresh_token expired) →
    permanent CodexAuthRefreshError."""
    transport = _OAuthTransport(
        status_code=400,
        body={
            "error": "invalid_grant",
            "error_description": "Refresh token expired.",
        },
    )
    client = httpx.Client(transport=transport)
    auth = _auth_with_refresh_token()
    with pytest.raises(CodexAuthRefreshError) as exc:
        refresh_codex_auth(
            auth, path=tmp_path / "auth.json", http_client=client
        )
    assert exc.value.permanent is True
    assert exc.value.code == "invalid_grant"


def test_refresh_transient_failure_marked_retriable(tmp_path):
    transport = _OAuthTransport(status_code=503)
    client = httpx.Client(transport=transport)
    auth = _auth_with_refresh_token()
    with pytest.raises(CodexAuthRefreshError) as exc:
        refresh_codex_auth(
            auth, path=tmp_path / "auth.json", http_client=client
        )
    assert exc.value.permanent is False


# ─── arefresh_codex_auth (async sibling) ───────────────────────────────


@pytest.mark.asyncio
async def test_arefresh_applies_response_tokens(tmp_path):
    transport = _AsyncOAuthTransport(body={
        "access_token": "atk-new",
        "refresh_token": "rtk-new",
    })
    client = httpx.AsyncClient(transport=transport)
    auth = _auth_with_refresh_token()
    updated = await arefresh_codex_auth(
        auth,
        path=tmp_path / "auth.json",
        write_back=False,
        http_client=client,
    )
    assert updated.access_token == "atk-new"
    assert updated.refresh_token == "rtk-new"


# ─── Chat model auto-refresh on 401 ────────────────────────────────────


def test_chat_model_refreshes_on_401_then_retries(tmp_path, monkeypatch):
    """End-to-end: a 401 on the first attempt triggers a refresh, then
    a second attempt succeeds. The auth.json file ends up with the
    new tokens persisted."""
    from tests.conftest import _make_llm, _sse_bytes

    # Seed a working auth file.
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(json.dumps({
        "auth_mode": "chatgpt",
        "tokens": {
            "access_token": "atk-stale",
            "id_token": "id",
            "refresh_token": "rtk-old",
            "account_id": "acct",
        },
        "last_refresh": "2026-05-21T00:00:00Z",
    }))

    # First Codex call returns 401; second returns a valid SSE body.
    class _TwoShotTransport(httpx.BaseTransport):
        def __init__(self):
            self.calls = 0
            self.bodies = [
                (401, b'{"detail":"token expired"}'),
                (200, _sse_bytes([
                    ("response.created", {"response": {"id": "r"}}),
                    ("response.output_text.delta", {"delta": "ok"}),
                    ("response.completed", {
                        "response": {"id": "r", "status": "completed"},
                    }),
                ])),
            ]
            self.last_authorizations: list[str] = []

        def handle_request(self, request):
            self.last_authorizations.append(
                request.headers.get("authorization", "")
            )
            status, body = self.bodies[self.calls]
            self.calls += 1
            return httpx.Response(status, content=body, request=request)

    codex_transport = _TwoShotTransport()
    llm = _make_llm(auth_path, transport=codex_transport)

    # Patch the refresh HTTP call to use an in-process transport.
    # We capture the original before patching so the inner call hits
    # the real implementation (not itself — that's the infinite-
    # recursion trap).
    refresh_transport = _OAuthTransport(body={
        "access_token": "atk-fresh",
        "refresh_token": "rtk-new",
    })
    from langchain_codex_plus import codex_auth as _ca
    real_do_refresh = _ca._do_refresh_sync

    def _patched_do_refresh(_client, endpoint, body):
        c2 = httpx.Client(transport=refresh_transport)
        return real_do_refresh(c2, endpoint, body)

    monkeypatch.setattr(_ca, "_do_refresh_sync", _patched_do_refresh)

    msg = llm.invoke([HumanMessage("hi")])
    assert msg.content == "ok"
    assert codex_transport.calls == 2  # 401 + retry
    # Second call used the refreshed token.
    assert codex_transport.last_authorizations[1] == "Bearer atk-fresh"
    # auth.json has the refreshed tokens persisted.
    persisted = json.loads(auth_path.read_text())
    assert persisted["tokens"]["access_token"] == "atk-fresh"
    assert persisted["tokens"]["refresh_token"] == "rtk-new"


def test_chat_model_auto_refresh_false_propagates_401(tmp_path):
    """With ``auto_refresh=False``, a 401 surfaces directly without
    any refresh attempt — same as pre-refresh behavior."""
    from langchain_codex_plus import CodexResponseError
    from tests.conftest import _CaptureTransport, _make_llm

    auth_path = tmp_path / "auth.json"
    auth_path.write_text(json.dumps({
        "auth_mode": "chatgpt",
        "tokens": {
            "access_token": "atk", "id_token": "id",
            "refresh_token": "rtk", "account_id": "a",
        },
    }))
    transport = _CaptureTransport(
        status_code=401,
        body=b'{"detail":"token expired"}',
    )
    llm = _make_llm(auth_path, transport=transport, auto_refresh=False)
    with pytest.raises(CodexResponseError) as exc:
        llm.invoke([HumanMessage("hi")])
    assert "HTTP 401" in str(exc.value)


def test_chat_model_permanent_refresh_failure_raises(tmp_path, monkeypatch):
    """If the refresh itself returns a permanent error
    (``invalid_grant``), ``CodexAuthRefreshError`` propagates up to
    the caller — they need to re-run ``codex login``."""
    from tests.conftest import _make_llm

    auth_path = tmp_path / "auth.json"
    auth_path.write_text(json.dumps({
        "auth_mode": "chatgpt",
        "tokens": {
            "access_token": "atk", "id_token": "id",
            "refresh_token": "rtk-bad", "account_id": "a",
        },
    }))

    class _Always401(httpx.BaseTransport):
        def handle_request(self, request):
            return httpx.Response(
                401, content=b'{"detail":"token expired"}', request=request
            )

    llm = _make_llm(auth_path, transport=_Always401())

    refresh_transport = _OAuthTransport(
        status_code=400,
        body={"error": "invalid_grant",
              "error_description": "Refresh token expired."},
    )
    from langchain_codex_plus import codex_auth as _ca
    real = _ca._do_refresh_sync

    def _patched(client, endpoint, body):
        c2 = httpx.Client(transport=refresh_transport)
        return real(c2, endpoint, body)

    monkeypatch.setattr(_ca, "_do_refresh_sync", _patched)

    with pytest.raises(CodexAuthRefreshError) as exc:
        llm.invoke([HumanMessage("hi")])
    assert exc.value.permanent is True
    assert exc.value.code == "invalid_grant"
