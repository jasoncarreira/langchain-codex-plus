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

Alpha. v0.0.1. 134 tests + a gated real-account smoke test pass.

## Auth

Run `codex login` once. The CLI writes OAuth credentials to
`$CODEX_HOME/auth.json` (defaults to `~/.codex/auth.json`). This
package reads the file directly — there's no separate setup.

```python
from langchain_codex_plus import ChatCodexPlus

llm = ChatCodexPlus(model="gpt-5.4")
llm.invoke("Say ok.")
```

When the access token expires (~1h TTL), a 401 response triggers an
automatic refresh against `auth.openai.com/oauth/token`, then the
call retries once. Permanent refresh failures (expired / revoked /
already-used refresh token) raise `CodexAuthRefreshError` with
`permanent=True` — the operator must re-run `codex login`. Opt out
with `auto_refresh=False` if you want to handle 401s yourself.

## Tool calling

Use `bind_tools` exactly like `ChatOpenAI.bind_tools`:

```python
from langchain_core.tools import tool
from langchain_codex_plus import ChatCodexPlus

@tool
def get_weather(location: str) -> str:
    """Look up the weather."""
    return f"sunny in {location}"

llm = ChatCodexPlus().bind_tools([get_weather])
msg = llm.invoke("Weather in Boston?")
# msg.tool_calls → [{"name": "get_weather", "args": {"location": "Boston"}, "id": "call_..."}]
```

Send tool results back via `ToolMessage(content=..., tool_call_id=...)` —
the protocol layer serializes them as Codex `function_call_output`
entries.

## Multimodal

`HumanMessage` content can be a list mixing text and image blocks:

```python
from langchain_core.messages import HumanMessage

llm.invoke([HumanMessage(content=[
    {"type": "text", "text": "What's in this image?"},
    {"type": "image_url", "image_url": "https://example.com/cat.png"},
])])
```

Both LangChain image-block conventions are accepted (`{type: image_url,
image_url: {url, detail}}` and `{type: image, source_type: "url"|"base64",
...}`). Base64 data is auto-encoded as a `data:` URL.

## Stop sequences

Codex's `/codex/responses` rejects the `stop` parameter, so we match
client-side. Streaming uses a buffered matcher so stop sequences
split across SSE chunks (the common tokenization case) still
truncate cleanly:

```python
llm.invoke("Count from 1 to 100", stop=["50"])
# → "1, 2, 3, ... 49, "
```

## Reasoning effort and client version

`reasoning_effort` defaults to `None`, which omits the `reasoning` field so the
model applies its own default. (Before 0.0.9 the default was `"none"`, which
`gpt-6-astra` rejects; callers who relied on it and want minimal reasoning should
now pass `reasoning_effort="none"` explicitly.) Accepted values are validated
server-side per model, e.g. gpt-5.4 takes `none|low|medium|high|xhigh`,
gpt-6-astra and gpt-6-sol take `low|medium|high|xhigh|max|ultra`, and the luna
models take `low|medium|high|xhigh|max` (no `none`).

The package presents itself as Codex CLI `0.159.1`, which the backend uses to
gate newer models (for example, `gpt-6.1-sol` is only listed for clients
`>= 0.159.0`). Set `CODEX_PLUS_CLIENT_VERSION` to present a newer version
without waiting for a release; an explicit `client_version=` argument wins.

## Sharing auth.json with other consumers

Refresh tokens are single-use, and the server may revoke the whole token
family when a spent one is presented again, logging out every consumer of
`auth.json` at once. So on a 401 this package never refreshes from the token
it cached at startup:

1. Under an exclusive lock on `auth.json.lock`, it re-reads `auth.json`. If
   another consumer (the Codex CLI or app, another process, another model
   instance) already rotated the tokens, it adopts them and skips the refresh.
2. Otherwise it refreshes from the on-disk tokens and writes the result back
   atomically, so concurrent refreshers in any process that uses this package
   refresh exactly once.
3. If the refresh still fails, it re-reads once more before raising, in case a
   consumer that doesn't take the lock rotated in between.

The same behaviour is available directly as `refresh_codex_auth_coordinated` /
`arefresh_codex_auth_coordinated`. The lock is advisory (`flock`); it doesn't
coordinate with the Codex CLI itself, which is why step 3 exists.

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

Callback exceptions are caught and logged — they never break the
response path.

## License

MIT. See `LICENSE`.
