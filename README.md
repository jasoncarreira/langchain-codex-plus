# langchain-codex-plus

LangChain `ChatModel` for OpenAI's **ChatGPT-account-backed Codex** —
the subscription protocol (Codex Plus / Pro plans), NOT the public
`api.openai.com` API.

## What this is

OpenAI's Codex CLI signs you in with a **ChatGPT account** (browser
OAuth) and routes traffic through:

```
https://chatgpt.com/backend-api/codex/responses
```

— a different protocol than `api.openai.com/v1/chat/completions`. It
has its own request shape, its own auth (OAuth bearer instead of
`OPENAI_API_KEY`), and exposes quota-window utilization via response
headers (`x-codex-primary-*`, `x-codex-secondary-*`).

This package wraps that protocol in a LangChain `BaseChatModel` so
you can use a Codex Plus subscription from any LangChain-built agent
the way you'd use `ChatOpenAI` or `ChatAnthropic`.

## What this is NOT

* Not for `api.openai.com` traffic — use `langchain-openai` for that.
* Not for Claude — use `langchain-anthropic` or `langchain-claude-code`.
* Not a re-implementation of the Codex CLI's agent loop — just the
  chat-model surface.

## Status

Alpha. v0.0.1.

## Auth

Run `codex login` once. The CLI writes OAuth credentials to
`$CODEX_HOME/auth.json` (defaults to `~/.codex/auth.json`). This
package reads the file directly — there's no separate setup.

```python
from langchain_codex_plus import ChatCodexPlus

llm = ChatCodexPlus(model="gpt-5.4")
llm.invoke("Say ok.")
```

If `auth.json` doesn't exist, init raises with a hint to run
`codex login`.

## Rate-limit hook

Every successful `/codex/responses` response carries quota headers
(`x-codex-primary-*` / `-secondary-*`). The chat model parses these
into a `CodexRateLimits` dataclass and (optionally) calls a callback
so your monitoring layer can persist them:

```python
from langchain_codex_plus import ChatCodexPlus, CodexRateLimits

def on_rate_limits(rl: CodexRateLimits) -> None:
    print(f"5h: {rl.primary.used_percent}% / 7d: {rl.secondary.used_percent}%")

llm = ChatCodexPlus(model="gpt-5.4", rate_limit_callback=on_rate_limits)
```

## License

MIT. See `LICENSE`.
