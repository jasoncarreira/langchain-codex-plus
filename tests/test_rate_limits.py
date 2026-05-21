"""Tests for ``langchain_codex_plus.rate_limits``.

Header fixtures here are taken VERBATIM from a real Codex Plus 200
response captured 2026-05-20 against ``chatgpt.com/backend-api/codex/
responses`` (Plus tier, idle account). Pinning real values guards
against subtle parser drift if the server changes case, formatting,
or omits fields.
"""
from __future__ import annotations

from langchain_codex_plus.rate_limits import (
    CodexCredits,
    CodexQuotaWindow,
    parse_codex_rate_limits,
)

# Real headers from a 200 ``/codex/responses`` response on a Plus
# account, captured 2026-05-20. Field order intentionally jumbled to
# verify parser doesn't depend on it.
REAL_HEADERS = {
    "Date": "Thu, 21 May 2026 01:09:51 GMT",
    "Server": "cloudflare",
    "x-codex-plan-type": "plus",
    "x-codex-active-limit": "premium",
    "x-codex-primary-used-percent": "1",
    "x-codex-primary-window-minutes": "300",
    "x-codex-primary-reset-after-seconds": "18000",
    "x-codex-primary-reset-at": "1779343790",
    "x-codex-primary-over-secondary-limit-percent": "0",
    "x-codex-secondary-used-percent": "0",
    "x-codex-secondary-window-minutes": "10080",
    "x-codex-secondary-reset-after-seconds": "604800",
    "x-codex-secondary-reset-at": "1779930590",
    "x-codex-credits-balance": "",
    "x-codex-credits-has-credits": "False",
    "x-codex-credits-unlimited": "False",
}


def test_parse_real_headers_full_snapshot():
    """The load-bearing test: parse the captured Plus account
    headers and verify every field round-trips correctly."""
    rl = parse_codex_rate_limits(REAL_HEADERS)
    assert rl is not None
    assert rl.plan_type == "plus"
    assert rl.active_limit == "premium"
    assert rl.primary_over_secondary_limit_percent == 0.0

    # 5h window
    assert rl.primary == CodexQuotaWindow(
        used_percent=1.0,
        window_minutes=300,
        reset_at=1779343790,
        reset_after_seconds=18000,
    )

    # 7d window
    assert rl.secondary == CodexQuotaWindow(
        used_percent=0.0,
        window_minutes=10080,
        reset_at=1779930590,
        reset_after_seconds=604800,
    )

    # Credits — empty balance on a sub-only account, "False"/"False"
    assert rl.credits == CodexCredits(
        has_credits=False,
        unlimited=False,
        balance=None,
    )


def test_parse_returns_none_for_headers_without_codex_signals():
    """``/codex/models`` and validation 400s don't carry any of these
    headers — caller should see None and skip the rate-limit hook."""
    assert (
        parse_codex_rate_limits({
            "Date": "Thu, 21 May 2026 01:09:51 GMT",
            "Server": "cloudflare",
            "Content-Type": "application/json",
        })
        is None
    )


def test_parse_handles_case_insensitive_header_names():
    """``httpx.Headers`` lowercases keys; ``requests.Response.headers``
    is case-insensitive; raw dicts from urllib are mixed-case. Parser
    must work with all three."""
    lower = {k.lower(): v for k, v in REAL_HEADERS.items()}
    upper = {k.upper(): v for k, v in REAL_HEADERS.items()}
    a = parse_codex_rate_limits(lower)
    b = parse_codex_rate_limits(upper)
    assert a == b
    assert a is not None and a.plan_type == "plus"


def test_parse_partial_primary_window_only():
    """Server might omit secondary headers on some plans; parser
    should produce a partial snapshot rather than refusing."""
    rl = parse_codex_rate_limits({
        "x-codex-primary-used-percent": "50",
        "x-codex-primary-window-minutes": "300",
    })
    assert rl is not None
    assert rl.primary is not None
    assert rl.primary.used_percent == 50.0
    assert rl.primary.window_minutes == 300
    assert rl.primary.reset_at is None  # not provided
    assert rl.secondary is None
    assert rl.credits is None


def test_parse_window_requires_used_percent():
    """Without ``used-percent`` we don't synthesize a window from
    leftover headers — that prevents misleading 'utilization=0' reads
    when the gateway actually said nothing about utilization."""
    rl = parse_codex_rate_limits({
        # Has window-minutes but no used-percent — should produce
        # primary=None, but still carry plan-type from the response.
        "x-codex-primary-window-minutes": "300",
        "x-codex-plan-type": "plus",
    })
    assert rl is not None
    assert rl.plan_type == "plus"
    assert rl.primary is None


def test_parse_credits_unlimited_true():
    """Pro tier reportedly returns ``unlimited=True`` on credits."""
    rl = parse_codex_rate_limits({
        "x-codex-credits-has-credits": "True",
        "x-codex-credits-unlimited": "True",
        "x-codex-credits-balance": "999.50",
    })
    assert rl is not None
    assert rl.credits == CodexCredits(
        has_credits=True,
        unlimited=True,
        balance="999.50",
    )


def test_parse_credits_only_when_signaled():
    """If none of the credits booleans appear, credits is None even
    if balance is a stray header (rare, but defensive)."""
    rl = parse_codex_rate_limits({
        "x-codex-primary-used-percent": "10",
        "x-codex-credits-balance": "stray",
    })
    assert rl is not None
    assert rl.credits is None


def test_parse_bad_numeric_field_falls_back_to_none():
    """If the gateway sends a malformed number, treat as missing
    rather than crashing the response handler."""
    rl = parse_codex_rate_limits({
        "x-codex-primary-used-percent": "not-a-number",
    })
    # used-percent malformed → window won't construct → None primary,
    # but if other headers exist they'd still surface. With ONLY this
    # bad header, no signal at all → snapshot is None.
    assert rl is None


def test_parse_returns_partial_when_only_plan_type_present():
    """A response that carries just ``x-codex-plan-type`` (e.g., a
    cached probe) should still surface the plan label."""
    rl = parse_codex_rate_limits({"x-codex-plan-type": "pro"})
    assert rl is not None
    assert rl.plan_type == "pro"
    assert rl.primary is None
    assert rl.secondary is None
    assert rl.credits is None


def test_codex_rate_limits_is_frozen_dataclass():
    """Snapshots are immutable — callers can't mutate the snapshot
    they receive from the chat model's rate_limit_callback."""
    import dataclasses
    rl = parse_codex_rate_limits(REAL_HEADERS)
    assert rl is not None
    assert dataclasses.is_dataclass(rl)
    # Frozen dataclass raises on attribute set.
    try:
        rl.plan_type = "mutated"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("CodexRateLimits should be frozen")
