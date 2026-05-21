"""Codex Plus OAuth credentials reader + refresher.

Codex CLI (``openai/codex``) persists OAuth state at
``$CODEX_HOME/auth.json`` (defaults to ``~/.codex/auth.json``) after
the user runs ``codex login`` and goes through the ChatGPT-account
browser flow. The file is mode 0600 and has shape::

    {
        "auth_mode": "chatgpt",
        "OPENAI_API_KEY": null | "sk-...",
        "tokens": {
            "access_token": "...",
            "id_token": "...",
            "refresh_token": "...",
            "account_id": "..."
        },
        "last_refresh": "2026-05-21T01:02:18Z"
    }

Refresh: the access token has ~1h TTL. When it expires, this module
can POST to ChatGPT's OAuth token endpoint with the refresh token to
get a new pair. Refresh tokens may rotate, so the response is written
back to ``auth.json`` atomically. The chat model auto-refreshes on 401.

References (verified against ``openai/codex`` source 2026-05-20):

* Storage struct: ``codex-rs/login/src/auth/storage.rs`` (``AuthDotJson``)
* Path resolution: ``codex_home.join("auth.json")``
* API base: ``https://chatgpt.com/backend-api``
* Refresh endpoint: ``https://auth.openai.com/oauth/token``
  (override with ``CODEX_REFRESH_TOKEN_URL_OVERRIDE`` env var)
* OAuth client_id: ``app_EMoamEEZ73f0CkXaXp7hrann``
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import httpx

CODEX_API_BASE = "https://chatgpt.com/backend-api"
"""Codex Plus protocol base. Distinct from ``api.openai.com``."""


REFRESH_TOKEN_URL = "https://auth.openai.com/oauth/token"
"""Default refresh endpoint (matches ``openai/codex`` constant).
Override with the ``CODEX_REFRESH_TOKEN_URL_OVERRIDE`` env var."""


REFRESH_TOKEN_URL_OVERRIDE_ENV_VAR = "CODEX_REFRESH_TOKEN_URL_OVERRIDE"


CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
"""OAuth client_id sent on refresh requests. Public value pulled
from the codex CLI source — same string the CLI uses, so refresh
requests look identical from the server's perspective."""


class CodexAuthNotFoundError(FileNotFoundError):
    """Raised when ``auth.json`` is missing — operator hasn't run
    ``codex login`` yet. The error message includes the expected path
    so callers can surface it cleanly."""


class CodexAuthInvalidError(ValueError):
    """Raised when ``auth.json`` exists but doesn't contain a usable
    OAuth bundle (e.g., API-key-only mode, or partially-written file).
    """


class CodexAuthRefreshError(RuntimeError):
    """Raised when an OAuth refresh request fails. The
    :attr:`permanent` flag distinguishes:

    * **Permanent**: refresh_token expired / revoked / already used.
      Operator must re-run ``codex login`` — retry won't help.
    * **Transient**: network error, 5xx, etc. Retry may succeed.
    """

    def __init__(
        self,
        message: str,
        *,
        permanent: bool = False,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.permanent = permanent
        self.code = code


@dataclass(frozen=True)
class CodexAuth:
    """OAuth bundle from ``~/.codex/auth.json``."""

    auth_mode: str
    """``"chatgpt"`` for OAuth (subscription) mode, ``"apikey"`` for
    API-key fallback (rare — Codex CLI primarily targets subscription)."""

    access_token: str
    """Short-lived bearer for ``chatgpt.com/backend-api/...``. Codex
    CLI refreshes it via the ``refresh_token``; consumers should treat
    this as opaque and re-read the file when calls start returning 401."""

    id_token: str | None
    """JWT identifying the ChatGPT account — used in some
    ``agent-identities/*`` flows. Not needed for ``/codex/models`` or
    ``/codex/responses``."""

    refresh_token: str | None
    """Used to obtain a fresh ``access_token`` when the current one
    expires. Refresh flow not implemented in this module — delegate to
    ``codex auth refresh``."""

    account_id: str | None
    """Stable ChatGPT account identifier."""

    last_refresh: datetime | None
    """When the access_token was last minted. Useful as a coarse
    expiry heuristic (Codex's access tokens have ~1h TTL in practice;
    treat anything older than 55min as suspect)."""


def codex_home() -> Path:
    """Return ``$CODEX_HOME`` or ``~/.codex``. Mirrors codex CLI."""
    env = os.environ.get("CODEX_HOME", "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / ".codex"


def auth_file_path(home: Path | None = None) -> Path:
    return (home or codex_home()) / "auth.json"


def load_codex_auth(
    path: Path | None = None, *, strict: bool = False
) -> CodexAuth | None:
    """Read ``auth.json`` and return a :class:`CodexAuth`.

    Default behavior is **lenient** — returns ``None`` when:

    * the file doesn't exist (operator hasn't run ``codex login``), or
    * the file exists but lacks an OAuth bundle (API-key-only mode).

    Pass ``strict=True`` to raise :class:`CodexAuthNotFoundError` /
    :class:`CodexAuthInvalidError` instead. Use strict in code paths
    where missing auth is a configuration bug (e.g., a chat model
    explicitly constructed with intent to call Codex Plus).

    Always raises on malformed JSON — that's a real bug, not a missing-
    auth case.
    """
    p = path or auth_file_path()
    if not p.is_file():
        if strict:
            raise CodexAuthNotFoundError(
                f"Codex Plus auth not found at {p}. "
                f"Run `codex login` to authenticate, then retry."
            )
        return None
    raw = json.loads(p.read_text(encoding="utf-8"))
    auth_mode = str(raw.get("auth_mode") or "")
    tokens = raw.get("tokens") or {}
    access_token = tokens.get("access_token")
    if not access_token:
        if strict:
            raise CodexAuthInvalidError(
                f"{p} exists but has no OAuth bundle "
                f"(auth_mode={auth_mode!r}). "
                f"Run `codex login` to re-authenticate."
            )
        return None
    last_refresh_raw = raw.get("last_refresh")
    last_refresh: datetime | None = None
    if isinstance(last_refresh_raw, str):
        # Codex writes ISO-8601 with trailing Z; Python <3.11 needs
        # an explicit replace.
        try:
            last_refresh = datetime.fromisoformat(
                last_refresh_raw.replace("Z", "+00:00")
            )
        except ValueError:
            last_refresh = None
    return CodexAuth(
        auth_mode=auth_mode,
        access_token=str(access_token),
        id_token=tokens.get("id_token"),
        refresh_token=tokens.get("refresh_token"),
        account_id=tokens.get("account_id"),
        last_refresh=last_refresh,
    )


def is_likely_expired(
    auth: CodexAuth, *, ttl_minutes: int = 55
) -> bool:
    """Coarse expiry heuristic. ChatGPT access tokens have ~1h TTL in
    practice; default to 55min so we're conservative.

    Returns ``True`` if ``last_refresh`` is older than ``ttl_minutes``
    OR if ``last_refresh`` is missing entirely (can't prove freshness).
    """
    if auth.last_refresh is None:
        return True
    age = datetime.now(tz=UTC) - auth.last_refresh
    return age.total_seconds() > ttl_minutes * 60


# ─── Refresh ────────────────────────────────────────────────────────────


def _refresh_endpoint() -> str:
    return os.environ.get(REFRESH_TOKEN_URL_OVERRIDE_ENV_VAR) or REFRESH_TOKEN_URL


def _refresh_body(refresh_token: str) -> dict[str, str]:
    return {
        "client_id": CODEX_OAUTH_CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }


def _apply_refresh_response(
    auth: CodexAuth, response_data: dict
) -> CodexAuth:
    """Merge a successful refresh response into an existing
    :class:`CodexAuth`. Per the codex CLI source, every field in the
    response is optional — only fields present in the response replace
    their counterpart on the auth bundle. The refresh_token may rotate
    or stay the same.
    """
    new_access = response_data.get("access_token")
    new_id = response_data.get("id_token")
    new_refresh = response_data.get("refresh_token")
    if not isinstance(new_access, str) or not new_access:
        raise CodexAuthRefreshError(
            "Refresh response missing access_token",
            permanent=False,
        )
    return replace(
        auth,
        access_token=new_access,
        id_token=new_id if isinstance(new_id, str) else auth.id_token,
        refresh_token=(
            new_refresh
            if isinstance(new_refresh, str)
            else auth.refresh_token
        ),
        last_refresh=datetime.now(tz=UTC),
    )


def _classify_refresh_error(
    status_code: int, body: dict | None
) -> CodexAuthRefreshError:
    """Map an OAuth token-endpoint error response to a typed error.

    Per the codex CLI's classification (``manager.rs``), these
    error codes mean the user must log in again:

    * ``expired_token`` / ``invalid_grant`` — refresh_token expired
    * ``refresh_token_reused`` — already-used token
    * ``invalid_token`` — token revoked
    * 4xx generally → permanent (auth-side problem)
    * 5xx → transient (server-side, retry may work)
    """
    code = None
    message = f"HTTP {status_code} from {_refresh_endpoint()}"
    if isinstance(body, dict):
        err = body.get("error")
        desc = body.get("error_description") or body.get("detail")
        if isinstance(err, str):
            code = err
            message = f"{err}: {desc}" if desc else err
        elif isinstance(desc, str):
            message = desc
    permanent_codes = {
        "expired_token",
        "invalid_grant",
        "invalid_token",
        "refresh_token_reused",
        "refresh_token_invalidated",
    }
    permanent = (
        (code in permanent_codes)
        or (400 <= status_code < 500 and status_code != 429)
    )
    return CodexAuthRefreshError(message, permanent=permanent, code=code)


def _write_auth_json(path: Path, auth: CodexAuth) -> None:
    """Atomically rewrite ``auth.json`` with refreshed tokens.

    Reads the existing file (if any) to preserve fields we don't
    track in :class:`CodexAuth` (``OPENAI_API_KEY``, ``agent_identity``,
    ``auth_mode``-but-set-via-explicit-API-key etc.); then merges the
    refreshed tokens in and writes via a ``.tmp``-then-rename so
    crashes during write can't leave the file corrupted.
    """
    existing: dict = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Corrupted file — start fresh rather than crash.
            existing = {}
    existing["auth_mode"] = auth.auth_mode or existing.get("auth_mode") or "chatgpt"
    tokens = existing.get("tokens") or {}
    tokens["access_token"] = auth.access_token
    if auth.id_token is not None:
        tokens["id_token"] = auth.id_token
    if auth.refresh_token is not None:
        tokens["refresh_token"] = auth.refresh_token
    if auth.account_id is not None:
        tokens["account_id"] = auth.account_id
    existing["tokens"] = tokens
    if auth.last_refresh is not None:
        existing["last_refresh"] = (
            auth.last_refresh.astimezone(UTC)
            .strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        )
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    try:
        tmp.chmod(0o600)
    except OSError:
        # On filesystems that don't support chmod (FAT, Docker
        # bind-mounts) this is a non-fatal hint; skip.
        pass
    tmp.replace(path)


def refresh_codex_auth(
    auth: CodexAuth,
    *,
    path: Path | None = None,
    write_back: bool = True,
    timeout_seconds: float = 30.0,
    http_client: httpx.Client | None = None,
) -> CodexAuth:
    """POST to the OAuth token endpoint to refresh ``auth``.

    Returns a new :class:`CodexAuth` with the rotated tokens. By
    default also writes the result back to ``auth.json`` (atomic
    rename via ``.tmp``) — pass ``write_back=False`` to skip the file
    update (useful for tests).

    Raises :class:`CodexAuthRefreshError`. The ``permanent`` attribute
    distinguishes errors that the operator must fix (re-run ``codex
    login``) from transient ones worth retrying.
    """
    if not auth.refresh_token:
        raise CodexAuthRefreshError(
            "No refresh_token available — operator must run "
            "`codex login` to authenticate.",
            permanent=True,
        )
    endpoint = _refresh_endpoint()
    body = _refresh_body(auth.refresh_token)
    if http_client is None:
        with httpx.Client(timeout=timeout_seconds) as client:
            response = _do_refresh_sync(client, endpoint, body)
    else:
        response = _do_refresh_sync(http_client, endpoint, body)
    updated = _apply_refresh_response(auth, response)
    if write_back:
        _write_auth_json(path or auth_file_path(), updated)
    return updated


async def arefresh_codex_auth(
    auth: CodexAuth,
    *,
    path: Path | None = None,
    write_back: bool = True,
    timeout_seconds: float = 30.0,
    http_client: httpx.AsyncClient | None = None,
) -> CodexAuth:
    """Async sibling of :func:`refresh_codex_auth`. Same contract."""
    if not auth.refresh_token:
        raise CodexAuthRefreshError(
            "No refresh_token available — operator must run "
            "`codex login` to authenticate.",
            permanent=True,
        )
    endpoint = _refresh_endpoint()
    body = _refresh_body(auth.refresh_token)
    if http_client is None:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await _do_refresh_async(client, endpoint, body)
    else:
        response = await _do_refresh_async(http_client, endpoint, body)
    updated = _apply_refresh_response(auth, response)
    if write_back:
        _write_auth_json(path or auth_file_path(), updated)
    return updated


def _do_refresh_sync(
    client: httpx.Client, endpoint: str, body: dict
) -> dict:
    try:
        response = client.post(
            endpoint, json=body, headers={"Content-Type": "application/json"}
        )
    except httpx.HTTPError as exc:
        raise CodexAuthRefreshError(
            f"Network error during refresh: {exc}", permanent=False
        ) from exc
    return _handle_refresh_response(response)


async def _do_refresh_async(
    client: httpx.AsyncClient, endpoint: str, body: dict
) -> dict:
    try:
        response = await client.post(
            endpoint, json=body, headers={"Content-Type": "application/json"}
        )
    except httpx.HTTPError as exc:
        raise CodexAuthRefreshError(
            f"Network error during refresh: {exc}", permanent=False
        ) from exc
    return _handle_refresh_response(response)


def _handle_refresh_response(response: httpx.Response) -> dict:
    if 200 <= response.status_code < 300:
        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise CodexAuthRefreshError(
                f"Refresh response was not valid JSON: {exc}",
                permanent=False,
            ) from exc
        if not isinstance(data, dict):
            raise CodexAuthRefreshError(
                "Refresh response was not a JSON object",
                permanent=False,
            )
        return data
    body: dict | None = None
    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError):
        body = None
    raise _classify_refresh_error(response.status_code, body)
