"""Codex Plus rate-limit header parser.

Every successful ``POST /codex/responses`` response from
``chatgpt.com/backend-api`` carries a family of ``x-codex-*`` headers
that report quota-window utilization for the caller's plan. This
module parses them into a typed :class:`CodexRateLimits` snapshot.

Header reference (verified against ``openai/codex`` source
``codex-rs/codex-api/src/rate_limits.rs``, 2026-05-20):

* ``x-codex-primary-used-percent`` — short-window utilization (0–100)
* ``x-codex-primary-window-minutes`` — short window length (typ. 300 = 5h)
* ``x-codex-primary-reset-at`` — unix-ts when the short window resets
* ``x-codex-primary-reset-after-seconds`` — same, expressed as delta
* ``x-codex-secondary-*`` — same fields, long window (typ. 10080 = 7d)
* ``x-codex-credits-{has-credits,unlimited,balance}`` — credits state
* ``x-codex-plan-type`` — subscription tier (``plus`` / ``pro``)
* ``x-codex-active-limit`` — coarse policy label (``premium`` etc.)

Rate-limit headers come back ONLY on ``/codex/responses`` 200s — NOT
on ``/codex/models`` (metadata) and NOT on validation 400s. The chat
model parses them on every successful response so callers can hook
quota monitoring.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class CodexQuotaWindow:
    """One quota window (primary = 5h, secondary = 7d in practice)."""

    used_percent: float
    """Utilization 0–100. ``50.0`` means 50% used."""

    window_minutes: int | None
    """Window length in minutes (300 for 5h, 10080 for 7d). Optional —
    not every response carries it (e.g., if the gateway omits it for a
    plan that doesn't expose window length)."""

    reset_at: int | None
    """Unix timestamp when this window resets. Optional."""

    reset_after_seconds: int | None
    """Same info as :attr:`reset_at`, expressed as a delta from now.
    Useful when client/server clocks disagree."""


@dataclass(frozen=True)
class CodexCredits:
    """Top-up credits state. Subscription accounts typically have
    ``has_credits=False`` and ``unlimited=False`` — plan windows are
    the load-bearing signal."""

    has_credits: bool
    unlimited: bool
    balance: str | None


@dataclass(frozen=True)
class CodexRateLimits:
    """Parsed snapshot of all rate-limit headers from one response."""

    plan_type: str | None
    """``"plus"``, ``"pro"``, or whatever ChatGPT calls the caller's
    subscription tier. May be ``None`` if the gateway didn't include
    the header (e.g., on a downgraded plan or in test environments)."""

    active_limit: str | None
    """Coarse policy bucket label (e.g., ``"premium"``). Mostly
    informational — quota signal is in :attr:`primary` and
    :attr:`secondary`."""

    primary: CodexQuotaWindow | None
    """Short-window utilization (5h in practice). ``None`` only if the
    gateway didn't include any ``x-codex-primary-*`` headers."""

    secondary: CodexQuotaWindow | None
    """Long-window utilization (7d in practice). ``None`` only if the
    gateway didn't include any ``x-codex-secondary-*`` headers."""

    credits: CodexCredits | None
    """Top-up credits state. ``None`` if no ``x-codex-credits-*``
    headers were present."""

    primary_over_secondary_limit_percent: float | None
    """Ratio of short-window usage to long-window usage, expressed as
    a percentage. The gateway sends this for accounts where the policy
    is asymmetric (e.g., short-window can exceed long-window's
    average). Largely an informational diagnostic."""


def _get(headers: Mapping[str, str], name: str) -> str | None:
    """Case-insensitive header lookup. ``httpx.Headers`` is already
    case-insensitive but raw dict mappings aren't, so we normalize."""
    lower = name.lower()
    for k, v in headers.items():
        if k.lower() == lower:
            return v
    return None


def _parse_float(headers: Mapping[str, str], name: str) -> float | None:
    raw = _get(headers, name)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _parse_int(headers: Mapping[str, str], name: str) -> int | None:
    raw = _get(headers, name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _parse_bool(headers: Mapping[str, str], name: str) -> bool | None:
    raw = _get(headers, name)
    if raw is None:
        return None
    # Codex serializes as ``True`` / ``False`` (Python-style). Accept
    # the lowercase forms too for resilience.
    return raw.strip().lower() in {"true", "1", "yes"}


def _parse_window(
    headers: Mapping[str, str], prefix: str
) -> CodexQuotaWindow | None:
    """Parse one window family (``primary`` or ``secondary``).

    Returns ``None`` if the *required* ``used-percent`` header is
    absent — the gateway sends all-or-nothing per window, so a missing
    ``used-percent`` is the signal that we shouldn't synthesize a
    partial window from leftover headers.
    """
    used = _parse_float(headers, f"x-codex-{prefix}-used-percent")
    if used is None:
        return None
    return CodexQuotaWindow(
        used_percent=used,
        window_minutes=_parse_int(headers, f"x-codex-{prefix}-window-minutes"),
        reset_at=_parse_int(headers, f"x-codex-{prefix}-reset-at"),
        reset_after_seconds=_parse_int(
            headers, f"x-codex-{prefix}-reset-after-seconds"
        ),
    )


def _parse_credits(headers: Mapping[str, str]) -> CodexCredits | None:
    """Parse credits headers — only if at least the boolean flags are
    present, otherwise return ``None``."""
    has = _parse_bool(headers, "x-codex-credits-has-credits")
    unlim = _parse_bool(headers, "x-codex-credits-unlimited")
    if has is None and unlim is None:
        return None
    balance_raw = _get(headers, "x-codex-credits-balance")
    return CodexCredits(
        has_credits=bool(has),
        unlimited=bool(unlim),
        balance=balance_raw or None,
    )


def parse_codex_rate_limits(
    headers: Mapping[str, str],
) -> CodexRateLimits | None:
    """Parse all ``x-codex-*`` rate-limit headers into a snapshot.

    Returns ``None`` when the response carries no Codex rate-limit
    headers at all (e.g., the response was from ``/codex/models`` or a
    400 validation error). Returns a partial snapshot when *some*
    headers are present — the caller's monitoring layer can decide
    whether partial signal is enough.
    """
    primary = _parse_window(headers, "primary")
    secondary = _parse_window(headers, "secondary")
    credits = _parse_credits(headers)
    plan = _get(headers, "x-codex-plan-type")
    active = _get(headers, "x-codex-active-limit")
    over = _parse_float(headers, "x-codex-primary-over-secondary-limit-percent")

    if all(
        x is None for x in (primary, secondary, credits, plan, active, over)
    ):
        return None

    return CodexRateLimits(
        plan_type=plan,
        active_limit=active,
        primary=primary,
        secondary=secondary,
        credits=credits,
        primary_over_secondary_limit_percent=over,
    )
