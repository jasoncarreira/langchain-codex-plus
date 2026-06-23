"""langchain-codex-plus: LangChain ChatModel for OpenAI Codex Plus.

Wraps OpenAI's ChatGPT-account-backed Codex subscription protocol
(``chatgpt.com/backend-api/codex/responses``) as a LangChain
``BaseChatModel``. See README for what this is and isn't.
"""
from __future__ import annotations

from langchain_codex_plus.codex_auth import (
    CODEX_API_BASE,
    CODEX_OAUTH_CLIENT_ID,
    REFRESH_TOKEN_URL,
    CodexAuth,
    CodexAuthInvalidError,
    CodexAuthNotFoundError,
    CodexAuthRefreshError,
    arefresh_codex_auth,
    auth_file_path,
    codex_home,
    is_likely_expired,
    load_codex_auth,
    refresh_codex_auth,
)
from langchain_codex_plus.codex_chat_model import ChatCodexPlus
from langchain_codex_plus.codex_protocol import (
    CodexCompletion,
    CodexResponseError,
    CodexToolCall,
    SseEvent,
    ToolChoice,
    aparse_sse_stream,
    build_request_body,
    consume_events,
    parse_error_body,
    parse_sse_stream,
    summarize_request_content,
)
from langchain_codex_plus.rate_limits import (
    CodexCredits,
    CodexQuotaWindow,
    CodexRateLimits,
    parse_codex_rate_limits,
)

__version__ = "0.0.5"

__all__ = [
    # codex_auth
    "CODEX_API_BASE",
    "CODEX_OAUTH_CLIENT_ID",
    "REFRESH_TOKEN_URL",
    "CodexAuth",
    "CodexAuthInvalidError",
    "CodexAuthNotFoundError",
    "CodexAuthRefreshError",
    "arefresh_codex_auth",
    "auth_file_path",
    "codex_home",
    "is_likely_expired",
    "load_codex_auth",
    "refresh_codex_auth",
    # codex_protocol
    "CodexCompletion",
    "CodexResponseError",
    "CodexToolCall",
    "SseEvent",
    "ToolChoice",
    "build_request_body",
    "consume_events",
    "parse_error_body",
    "aparse_sse_stream",
    "parse_sse_stream",
    "summarize_request_content",
    # rate_limits
    "CodexCredits",
    "CodexQuotaWindow",
    "CodexRateLimits",
    "parse_codex_rate_limits",
    # chat model
    "ChatCodexPlus",
]
