"""Coordinated refresh: never present a refresh token another consumer of
``auth.json`` has already spent.

Refresh tokens are single-use and the server may revoke the whole token
family when a spent one is presented again, logging out every consumer of
the file. These tests pin the three defences: adopt a rotation already on
disk, refresh from the on-disk tokens under a lock (so concurrent
refreshers refresh once), and re-read after a failed refresh.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time

import httpx
import pytest
from langchain_core.messages import HumanMessage

from langchain_codex_plus import (
    CodexAuth,
    CodexAuthRefreshError,
    arefresh_codex_auth_coordinated,
    load_codex_auth,
    refresh_codex_auth_coordinated,
)


def _write_auth(path, access, refresh):
    path.write_text(json.dumps({
        "auth_mode": "chatgpt",
        "tokens": {
            "access_token": access,
            "id_token": "id",
            "refresh_token": refresh,
            "account_id": "acct",
        },
        "last_refresh": "2026-09-25T00:00:00Z",
    }))


def _stale(access="atk-old", refresh="rtk-old") -> CodexAuth:
    return CodexAuth(
        auth_mode="chatgpt",
        access_token=access,
        id_token="id",
        refresh_token=refresh,
        account_id="acct",
        last_refresh=None,
    )


class _RefreshTransport(httpx.BaseTransport):
    """Token endpoint mock. Counts calls, records refresh tokens sent, and
    can run a side effect (e.g. another consumer rotating the file)."""

    def __init__(self, *, status=200, body=None, delay=0.0, side_effect=None):
        self.status = status
        self.body = body if body is not None else {
            "access_token": "atk-new", "refresh_token": "rtk-new",
        }
        self.delay = delay
        self.side_effect = side_effect
        self.sent_refresh_tokens: list[str] = []
        self._lock = threading.Lock()

    def handle_request(self, request):
        with self._lock:
            self.sent_refresh_tokens.append(json.loads(request.content)["refresh_token"])
        if self.delay:
            time.sleep(self.delay)
        if self.side_effect:
            self.side_effect()
        return httpx.Response(self.status, content=json.dumps(self.body).encode(), request=request)


class _AsyncRefreshTransport(httpx.AsyncBaseTransport):
    def __init__(self, *, delay=0.0):
        self.delay = delay
        self.sent_refresh_tokens: list[str] = []

    async def handle_async_request(self, request):
        self.sent_refresh_tokens.append(json.loads(request.content)["refresh_token"])
        await asyncio.sleep(self.delay)
        body = {"access_token": "atk-new", "refresh_token": "rtk-new"}
        return httpx.Response(200, content=json.dumps(body).encode(), request=request)


# ─── sync ──────────────────────────────────────────────────────────────


def test_adopts_rotation_already_on_disk_without_refreshing(tmp_path):
    path = tmp_path / "auth.json"
    _write_auth(path, "atk-rotated", "rtk-rotated")
    transport = _RefreshTransport()
    got = refresh_codex_auth_coordinated(
        _stale(), path=path, http_client=httpx.Client(transport=transport)
    )
    assert got.access_token == "atk-rotated"
    assert got.refresh_token == "rtk-rotated"
    assert transport.sent_refresh_tokens == []


def test_refreshes_from_on_disk_refresh_token_not_the_cached_one(tmp_path):
    path = tmp_path / "auth.json"
    # Same access token (nobody rotated the bearer), but the file carries
    # the authoritative refresh token.
    _write_auth(path, "atk-old", "rtk-on-disk")
    transport = _RefreshTransport()
    got = refresh_codex_auth_coordinated(
        _stale(refresh="rtk-cached"), path=path,
        http_client=httpx.Client(transport=transport),
    )
    assert transport.sent_refresh_tokens == ["rtk-on-disk"]
    assert got.access_token == "atk-new"
    assert load_codex_auth(path).refresh_token == "rtk-new"


def test_failed_refresh_adopts_rotation_made_meanwhile(tmp_path):
    """A consumer that doesn't take the lock (e.g. the Codex app) rotates
    while our request is in flight; ours comes back refresh_token_reused."""
    path = tmp_path / "auth.json"
    _write_auth(path, "atk-old", "rtk-old")
    transport = _RefreshTransport(
        status=400,
        body={"error": "refresh_token_reused"},
        side_effect=lambda: _write_auth(path, "atk-other", "rtk-other"),
    )
    got = refresh_codex_auth_coordinated(
        _stale(), path=path, http_client=httpx.Client(transport=transport)
    )
    assert got.access_token == "atk-other"


def test_failed_refresh_without_rotation_raises_permanent(tmp_path):
    path = tmp_path / "auth.json"
    _write_auth(path, "atk-old", "rtk-old")
    transport = _RefreshTransport(status=400, body={"error": "invalid_grant"})
    with pytest.raises(CodexAuthRefreshError) as exc:
        refresh_codex_auth_coordinated(
            _stale(), path=path, http_client=httpx.Client(transport=transport)
        )
    assert exc.value.permanent is True


def test_concurrent_refreshers_refresh_once(tmp_path):
    """Two threads holding the same stale auth both hit a 401. Only one may
    spend the refresh token; the other must adopt its result."""
    path = tmp_path / "auth.json"
    _write_auth(path, "atk-old", "rtk-old")
    transport = _RefreshTransport(delay=0.2)
    results: list[str] = []

    def worker():
        got = refresh_codex_auth_coordinated(
            _stale(), path=path, http_client=httpx.Client(transport=transport)
        )
        results.append(got.access_token)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert transport.sent_refresh_tokens == ["rtk-old"]
    assert results == ["atk-new", "atk-new"]


def test_write_leaves_no_temp_files(tmp_path):
    path = tmp_path / "auth.json"
    _write_auth(path, "atk-old", "rtk-old")
    refresh_codex_auth_coordinated(
        _stale(), path=path,
        http_client=httpx.Client(transport=_RefreshTransport()),
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["auth.json", "auth.json.lock"]


# ─── async ─────────────────────────────────────────────────────────────


async def test_async_adopts_rotation_already_on_disk(tmp_path):
    path = tmp_path / "auth.json"
    _write_auth(path, "atk-rotated", "rtk-rotated")
    transport = _AsyncRefreshTransport()
    got = await arefresh_codex_auth_coordinated(
        _stale(), path=path, http_client=httpx.AsyncClient(transport=transport)
    )
    assert got.access_token == "atk-rotated"
    assert transport.sent_refresh_tokens == []


async def test_async_concurrent_refreshers_refresh_once(tmp_path):
    path = tmp_path / "auth.json"
    _write_auth(path, "atk-old", "rtk-old")
    transport = _AsyncRefreshTransport(delay=0.2)

    async def one():
        got = await arefresh_codex_auth_coordinated(
            _stale(), path=path,
            http_client=httpx.AsyncClient(transport=transport),
        )
        return got.access_token

    results = await asyncio.wait_for(asyncio.gather(one(), one()), timeout=10)
    assert transport.sent_refresh_tokens == ["rtk-old"]
    assert results == ["atk-new", "atk-new"]


# ─── chat model ────────────────────────────────────────────────────────


def test_long_lived_model_adopts_external_rotation_on_401(tmp_path, monkeypatch):
    """The mimir shape: a model instance caches auth, another consumer
    rotates ``auth.json``, the cached bearer 401s. The model must retry with
    the rotated token and must NOT refresh with its spent refresh token."""
    from langchain_codex_plus import codex_auth as _ca
    from tests.conftest import _make_llm, _sse_bytes

    path = tmp_path / "auth.json"
    _write_auth(path, "atk-stale", "rtk-spent")

    class _TwoShot(httpx.BaseTransport):
        def __init__(self):
            self.auths: list[str] = []

        def handle_request(self, request):
            self.auths.append(request.headers.get("authorization", ""))
            if len(self.auths) == 1:
                return httpx.Response(401, content=b'{"detail":"expired"}', request=request)
            return httpx.Response(200, content=_sse_bytes([
                ("response.created", {"response": {"id": "r"}}),
                ("response.output_text.delta", {"delta": "ok"}),
                ("response.completed", {"response": {"id": "r", "status": "completed"}}),
            ]), request=request)

    codex = _TwoShot()
    llm = _make_llm(path, transport=codex)
    llm._resolve_auth()  # cache atk-stale / rtk-spent
    _write_auth(path, "atk-rotated", "rtk-rotated")  # another consumer refreshed

    def _no_refresh(*_a, **_k):
        raise AssertionError("must not refresh when the file already rotated")

    monkeypatch.setattr(_ca, "_do_refresh_sync", _no_refresh)
    assert llm.invoke([HumanMessage("hi")]).content == "ok"
    assert codex.auths == ["Bearer atk-stale", "Bearer atk-rotated"]
