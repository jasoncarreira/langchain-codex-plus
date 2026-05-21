"""Codex Plus OAuth credentials reader.

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

This module is read-only. Refresh is out of scope here — the caller is
expected to either (a) trust that the access token is fresh, (b) shell
out to ``codex auth refresh`` when expiry approaches, or (c) implement
the refresh dance against ChatGPT's OAuth token endpoint.

References (verified against ``openai/codex`` source 2026-05-20):

* Storage struct: ``codex-rs/login/src/auth/storage.rs`` (``AuthDotJson``)
* Path resolution: ``codex_home.join("auth.json")``
* API base: ``https://chatgpt.com/backend-api``
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

CODEX_API_BASE = "https://chatgpt.com/backend-api"
"""Codex Plus protocol base. Distinct from ``api.openai.com``."""


class CodexAuthNotFoundError(FileNotFoundError):
    """Raised when ``auth.json`` is missing — operator hasn't run
    ``codex login`` yet. The error message includes the expected path
    so callers can surface it cleanly."""


class CodexAuthInvalidError(ValueError):
    """Raised when ``auth.json`` exists but doesn't contain a usable
    OAuth bundle (e.g., API-key-only mode, or partially-written file).
    """


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
