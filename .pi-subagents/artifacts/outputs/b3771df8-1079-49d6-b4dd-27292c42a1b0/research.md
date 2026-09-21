# Research: Does `logprobs`/`top_logprobs` + `chat_template_kwargs` survive a LiteLLM proxy round trip?

Target: client POSTs `{base_url}/chat/completions` with `{"model","messages","temperature":0,"max_tokens":1,"logprobs":true,"top_logprobs":20,"chat_template_kwargs":{"enable_thinking":false,"preserve_thinking":false}}` and reads `choices[0].logprobs.content[0].top_logprobs`. LiteLLM ~1.10x (v1.100.0 released 2026-09-06 per release notes).

## VERDICT

**works-with-config** — logprobs/top_logprobs survive for an `openai/<name>` + `api_base` deployment (non-streaming), and this is the *documented* passthrough case. Two hard conditions: (a) do **not** run with `drop_params: true` (that silently deletes logprobs and you get a 200 with no logprobs, not an error); (b) `chat_template_kwargs` passthrough is the fragile part — it is documented as passed through, but has open bugs, so verify it on the wire.

## CONFIG

```yaml
# config.yaml
model_list:
  - model_name: my-judge
    litellm_params:
      model: openai/qwen3-8b            # openai/ prefix = "call an OpenAI-compatible /chat/completions"
      api_base: http://vllm-host:8000/v1
      api_key: os.environ/UPSTREAM_API_KEY   # the official openai client requires a key; any non-empty string works

litellm_settings:
  drop_params: false          # default — KEEP false. true = silent loss of logprobs (200 OK, no logprobs)
  # additional_drop_params: []   # do NOT list logprobs/top_logprobs here
```

Client side, if `chat_template_kwargs` is stripped, send it inside `extra_body` (documented REST-level passthrough):
`{"model": "...", "messages": [...], "logprobs": true, "top_logprobs": 20, "extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}`
(last-resort deterministic option: a LiteLLM *pass-through* endpoint so the body is not parsed at all — cost tracking is lost).

## Findings

### 1. Yes, the proxy returns `logprobs`/`top_logprobs` for chat completions — for providers that support it

- `logprobs` and `top_logprobs` are first-class *translated* OpenAI params in LiteLLM; the per-provider capability is exposed programmatically, not as a `supports_*` boolean: *"Use this function to get an up-to-date list of supported openai params for any model + provider: `litellm.get_supported_openai_params(model=..., custom_llm_provider=...)`"*. The docs capability table has explicit `logprobs` / `top_logprobs` columns, ✅ for **OpenAI, Azure OpenAI, xAI, OVHCloud** and blank for **Anthropic, Cohere, Replicate, Anyscale, VertexAI(Gemini), Bedrock, Ollama, TogetherAI, Databricks** etc. — [docs.litellm.ai/docs/completion/input](https://docs.litellm.ai/docs/completion/input)
- Capability code lives in the central per-provider list `litellm/litellm_core_utils/get_supported_openai_params.py` (imports `BadRequestError`), plus each provider's own transformation config, e.g. `litellm/llms/hosted_vllm/chat/transformation.py`, `litellm/llms/openai/chat/gpt_transformation.py` — [get_supported_openai_params.py](https://github.com/BerriAI/litellm/blob/main/litellm/litellm_core_utils/get_supported_openai_params.py) · [hosted_vllm/chat/transformation.py](https://github.com/BerriAI/litellm/blob/main/litellm/llms/hosted_vllm/chat/transformation.py)
- **`supports_logprobs` / `supports_top_logprobs` do NOT exist as capability flags in LiteLLM today.** Capability is expressed as *param-name strings* returned by `get_supported_openai_params`. A structured `ProviderCapabilities` declaration is still an **open** PR: [feat(capabilities): Add structured ProviderCapabilities declaration · PR #30032](https://github.com/BerriAI/litellm/pull/30032). (`supports_*` booleans of that shape belong to a different project, `specado`.) Severity: **info** — corrects the premise of the question.
- Direct proof the proxy *does* emit `logprobs` for an OpenAI-compatible upstream: the reported bug is not "logprobs missing" but "logprobs present, `top_logprobs: null`" → [issue #21932](https://github.com/BerriAI/litellm/issues/21932).
- The proxy does **not** forward the provider's raw JSON: it rebuilds the response through LiteLLM's own Pydantic types (`ChatCompletionTokenLogprob`, `litellm/types/utils.py`), which is exactly why the `top_logprobs` null bug existed → [issue #21932](https://github.com/BerriAI/litellm/issues/21932) (points at `litellm/types/utils.py#L840`). Consequence: fields LiteLLM models survive; unmodelled response fields do not. Severity: **medium** (design constraint, not a bug).

### 2. Unsupported param ⇒ **exception (400)**, never a silent drop — but `drop_params: true` makes it a *silent drop*

- *"**Default Behavior**: By default, LiteLLM raises an exception if you send a parameter to a model that doesn't support it. When `drop_params=True` is set, LiteLLM will drop the unsupported parameter instead."* Proxy equivalent: `litellm_settings: drop_params: true`. Per-param opt-out also exists via `additional_drop_params=["..."]`. So **`drop_params` defaults to `False`** → for a provider without logprobs you get `UnsupportedParamsError` surfaced as a 400 OpenAI-shaped error body ([drop_params docs](https://docs.litellm.ai/docs/completion/drop_params), [exception mapping](https://docs.litellm.ai/docs/exception_mapping), [proxy Error Reference](https://docs.litellm.ai/docs/proxy/error_reference)).
- Trap for this workload: `drop_params: true` (very common proxy setting) does **not** 400 — it returns a normal 200 with `logprobs` absent, so `choices[0].logprobs` is `None`/missing. Treat "no logprobs key" as a config regression. Severity: **high**.

### 3. Unknown top-level params (`chat_template_kwargs`): documented as forwarded; several open bugs make it the risky part

- Documented rule, twice: *"LiteLLM assumes any non-openai param is provider specific and passes it in as a kwarg in the request body"* and *"This ONLY DROPS UNSUPPORTED **OPENAI** PARAMS"* — i.e. `drop_params` never removes `chat_template_kwargs`; it is meant to reach the upstream body — [docs/completion/input](https://docs.litellm.ai/docs/completion/input), [docs/completion/provider_specific_params](https://docs.litellm.ai/docs/completion/provider_specific_params).
- REST/proxy form documented for non-OpenAI params is **`extra_body`** (same as OpenAI SDK semantics); `extra_body` contents are merged into the request params — [docs/proxy/clientside_auth](https://docs.litellm.ai/docs/proxy/clientside_auth). **`openai_extra_params` is not a LiteLLM setting** — no doc/code trace found; treat any reference to it as wrong.
- **Bug (open, 2026-04):** *"JSON-configured providers (e.g. Scaleway) not in `openai_compatible_providers` — non-standard params bypass `extra_body` wrapping"* — passthrough behaviour depends on whether the provider is registered in `litellm.openai_compatible_providers`; using the literal `openai/` prefix (rather than a JSON-registered provider) is the safe side of that split — [issue #26443](https://github.com/BerriAI/litellm/issues/26443). Severity: **medium**.
- **Bug (closed-stale ⇒ not fixed):** proxy mishandles client `extra_body` — params inside `extra_body` raise *"provider does not support these parameters, set drop_params=True"*, and only work when **double-nested** (`extra_body.extra_body`) — [issue #18039](https://github.com/BerriAI/litellm/issues/18039) (labels: bug, proxy, llm translation, **stale**). Severity: **high** if you route non-standard params through `extra_body`.
- **Related proxy bug:** `enable_thinking` reaching the OpenAI SDK as a kwarg → `AsyncCompletions.create() got an unexpected keyword argument 'enable_thinking'` — the exact param family you send — [issue #25697](https://github.com/BerriAI/litellm/issues/25697) (closed). Severity: **medium**.
- **Missing knob (open PR, not merged as of Aug 2026):** *"feat(proxy): optionally preserve unknown OpenAI request parameters for Provider mapping"* — confirms unknown top-level params are stripped on at least the proxy "Provider mapping" path, and that no config option exists in a released build yet — [PR #31697](https://github.com/BerriAI/litellm/pull/31697). Severity: **medium**.
- **Wildcard config is not a passthrough mechanism.** `model_name: "openai/*"` / provider wildcard routing only resolves *which deployment* a model string maps to; it has no effect on body params — [docs/wildcard_routing](https://docs.litellm.ai/docs/wildcard_routing), [docs/proxy/configs](https://docs.litellm.ai/docs/proxy/configs). Severity: **info**.
- Active test coverage proving this is live surface: *`test(e2e): cover Together AI reasoning, tool calls, template kwargs, and cost through a live proxy`* — [PR #38286](https://github.com/BerriAI/litellm/pull/38286) (open).

### 4. Known `logprobs` bugs — state and release

| Ref | Symptom | State |
|---|---|---|
| [#21932](https://github.com/BerriAI/litellm/issues/21932) | Proxy emits `top_logprobs: null` (invalid per OpenAI spec) when `logprobs:true`, OpenAI-compatible upstream | **Fixed** by [PR #22245](https://github.com/BerriAI/litellm/pull/22245) — merged **2026-02-28**, commit `d524b795` (≈v1.82.x; included in every 1.10x build) |
| [#18801](https://github.com/BerriAI/litellm/issues/18801) | `stream:true` + `logprobs:true` on vLLM-backed models → `PydanticSerializationError`, stream crashes | **Open** (2026-01-08, updated 2026-03-21) |
| [#4898](https://github.com/BerriAI/litellm/issues/4898) / [#3253](https://github.com/BerriAI/litellm/issues/3253) | `logprobs` + streaming broken (vLLM / OpenAI+Azure) | historical, same streaming class |
| [#8890](https://github.com/BerriAI/litellm/issues/8890) | Gemini 2.0 Flash: logprobs not returned | provider gap |
| [PR #31642](https://github.com/BerriAI/litellm/pull/31642) | Vertex/Gemini 3+ **advertises** `logprobs` in supported params but doesn't honour it | **Open** (2026-06-29) |
| [#17165](https://github.com/BerriAI/litellm/issues/17165) | Ollama logprobs not supported by LiteLLM | feature request |
| [#4031](https://github.com/BerriAI/litellm/issues/4031) | "Fail if `logprobs=True` and provider doesn't support them" | closed 2024 (the loud-failure ask) |

Practical read: **logprobs is reliable only for non-streaming calls on OpenAI/Azure-compatible surfaces (incl. `openai/` + vLLM).** Streaming is where the open breakage concentrates.

### 5. Where logprobs can get stripped

- **Response re-serialisation** (always, for every provider): the proxy returns LiteLLM's own `ModelResponse`/Pydantic dump, not upstream bytes. `logprobs`/`top_logprobs` *are* modelled so they survive; the null-vs-`[]` normalisation bug is the proof and is fixed in ≥ ~1.82 — [#21932](https://github.com/BerriAI/litellm/issues/21932), [PR #22245](https://github.com/BerriAI/litellm/pull/22245).
- **Non-OpenAI providers / streaming**: streaming+logprobs on vLLM raises `PydanticSerializationError` (open [#18801](https://github.com/BerriAI/litellm/issues/18801)); Gemini/Vertex silently return none ([#8890](https://github.com/BerriAI/litellm/issues/8890), [PR #31642](https://github.com/BerriAI/litellm/pull/31642)).
- **Guardrails / logging hooks**: the post-call guardrail + logging pipeline was reworked in [PR #24135](https://github.com/BerriAI/litellm/pull/24135) (merged 2026-03-20) and [PR #23910](https://github.com/BerriAI/litellm/pull/23910); the *logged* payload is the StandardLoggingPayload ([docs/proxy/logging_spec](https://docs.litellm.ai/docs/proxy/logging_spec)), which is separate from the HTTP body. **I found no issue evidencing logprobs being stripped from the client-visible response by a callback or guardrail** — but a guardrail that *rewrites* the response rebuilds `ModelResponse`, so treat any response-modifying guardrail as suspect and re-test with it enabled. Also note `litellm_settings.turn_off_message_logging` affects logging payloads, not the returned body.
- **`drop_params: true` / `additional_drop_params`** — silent removal at request time (see 2).

## Recommended 30-second verification (decisive, better than any doc reading)

```bash
curl -sS "$BASE/v1/chat/completions" -H "authorization: Bearer $KEY" -H "content-type: application/json" \
 -d '{"model":"my-judge","messages":[{"role":"user","content":"hi"}],"temperature":0,"max_tokens":1,
      "logprobs":true,"top_logprobs":20,"chat_template_kwargs":{"enable_thinking":false,"preserve_thinking":false}}' \
 | jq '{n:(.choices[0].logprobs.content[0].top_logprobs|length), first:.choices[0].logprobs.content[0].top_logprobs[0]}'
# expect n == 20 (or <=20 if upstream has a smaller vocab), first == {token, logprob}
# then run the same request with an OpenAI-compatible echo/mock upstream to confirm
# chat_template_kwargs arrived (vLLM: --enable-log-requests, or tcpdump) — logprobs surviving
# does NOT prove chat_template_kwargs was forwarded.
```

## Sources

Kept
- Input Params (capability table + non-openai passthrough note) — https://docs.litellm.ai/docs/completion/input
- Drop Unsupported Params (default = raise; `litellm_settings.drop_params`) — https://docs.litellm.ai/docs/completion/drop_params
- Provider-specific Params — https://docs.litellm.ai/docs/completion/provider_specific_params
- OpenAI-Compatible Endpoints (`openai/` prefix semantics, api_key required) — https://docs.litellm.ai/docs/providers/openai_compatible
- Proxy clientside auth / `extra_body` — https://docs.litellm.ai/docs/proxy/clientside_auth
- Exception mapping / proxy Error Reference (400 shape) — https://docs.litellm.ai/docs/exception_mapping , https://docs.litellm.ai/docs/proxy/error_reference
- get_supported_openai_params.py — https://github.com/BerriAI/litellm/blob/main/litellm/litellm_core_utils/get_supported_openai_params.py
- issue #21932 + PR #22245 (top_logprobs null, fix) — https://github.com/BerriAI/litellm/issues/21932 , https://github.com/BerriAI/litellm/pull/22245
- issue #18801 (streaming+logprobs vLLM, open) — https://github.com/BerriAI/litellm/issues/18801
- issue #26443 / #18039 / #25697 / PR #31697 (chat_template_kwargs-class passthrough)
- PR #31642, PR #30032, issue #8890, #17165, #4031
- Release notes (v1.100.0 = current 1.10x) — https://docs.litellm.ai/docs/release_notes/

Dropped
- SEO/overview posts (telnyx, seaflux, joshuaopolko), supply-chain-attack coverage (unrelated), vLLM logprobs API docs (not LiteLLM behaviour), Cloudflare/workers-ai and onyx/openai-agents issues (client-side, not proxy), openai cookbook (OpenAI, not LiteLLM), `specado` `Capabilities` struct (different project, caused by keyword collision).

## Gaps / next steps

1. Page bodies were read through search-engine extracts, not raw fetches (no fetch tool in this run). Exact `extra_body`/passthrough code path in `litellm/proxy/litellm_pre_call_utils.py` and `litellm/utils.py::get_optional_params` should be read directly (`pip show litellm` → site-packages) before trusting the passthrough for `chat_template_kwargs`.
2. `model_prices_and_context_window.json` is an 1.8 MB binary in GitHub; I could not grep it to 100% confirm absence of `supports_logprobs` keys. Absence of the flags is inferred from the docs capability mechanism + open PR #30032.
3. No evidence found either way for guardrail/callback stripping of logprobs from the client-visible body — test with your guardrails enabled.
4. Run the curl above against the real deployment + a mock upstream; that single test settles both the logprobs and the chat_template_kwargs questions better than further doc reading.

## Severity roll-up (for the implementation ticket)

- **high**: `litellm_settings.drop_params: true` silently removes `logprobs` → 200 with no logprobs. Source: docs/completion/drop_params.
- **high**: `extra_body` double-nesting bug #18039 (closed as `stale`, unfixed) if you pass `chat_template_kwargs` via `extra_body`.
- **medium**: `chat_template_kwargs` top-level passthrough is documented but has open defects (#26443, PR #31697) and a same-family crash (#25697). Verify on the wire.
- **medium**: streaming + logprobs broken for vLLM backends (#18801, open) → keep these classifier calls non-streaming.
- **low/medium**: LiteLLM < ~1.82 emits `top_logprobs: null` (#21932) — pin ≥1.82, fine on 1.10x.
- **info**: `supports_logprobs`/`supports_top_logprobs` are not LiteLLM flags; use `litellm.get_supported_openai_params(model, custom_llm_provider)` to gate.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Findings carry file paths and severities: litellm/litellm_core_utils/get_supported_openai_params.py (capability source of truth), litellm/llms/hosted_vllm/chat/transformation.py, litellm/llms/openai/chat/gpt_transformation.py, litellm/types/utils.py#L840 (ChatCompletionTokenLogprob, response re-serialisation), litellm/proxy/litellm_pre_call_utils.py + litellm/utils.py::get_optional_params (params passthrough, flagged as needing direct read). Severity roll-up: high = drop_params:true silent logprob loss (docs.litellm.ai/docs/completion/drop_params) and extra_body double-nesting bug #18039; medium = chat_template_kwargs passthrough (#26443, PR #31697, #25697), streaming+logprobs vLLM (#18801 open); low = top_logprobs:null fixed by PR #22245 merged 2026-02-28 commit d524b795."
    }
  ],
  "changedFiles": [
    "/home/amemiya/work/jev-bridge/.pi-subagents/artifacts/outputs/b3771df8-1079-49d6-b4dd-27292c42a1b0/research.md"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "web_search (multiple angles: docs capability table, drop_params default, chat_template_kwargs/extra_body passthrough, GitHub logprobs issues, response-transform stripping, release/version anchoring)",
      "result": "passed",
      "summary": "14 queries across querit/exa; primary docs + GitHub issues/PRs retrieved, incl. PR #22245 fix, #21932/#18801/#26443/#18039/#25697, PRs #31697/#31642/#30032/#38286, and v1.100.0 release anchoring."
    }
  ],
  "validationOutput": [
    "VERDICT: works-with-config — logprobs/top_logprobs survive an openai/-prefix + api_base round trip (non-streaming); capability is gated by litellm.get_supported_openai_params(), not by supports_logprobs flags (which do not exist in LiteLLM).",
    "drop_params default = False -> unsupported param raises exception mapped to 400; drop_params: true converts that into a silent drop (200 OK, no logprobs).",
    "Current LiteLLM line confirmed: v1.100.0 (release notes, Sept 2026); top_logprobs null-normalisation fix PR #22245 merged 2026-02-28 so present in 1.10x."
  ],
  "residualRisks": [
    "chat_template_kwargs top-level passthrough is documented but backed by open defects (#26443, PR #31697) and a same-family crash (#25697); must be confirmed on the wire against the real upstream, ideally with an echo/mock so logprobs success is not mistaken for param-forwarding success.",
    "No evidence either way on whether guardrails or custom callbacks strip logprobs from the client-visible response; re-test with guardrails enabled.",
    "Streaming + logprobs on vLLM backends remains broken (#18801, open) — keep the classifier call non-streaming.",
    "Evidence was gathered from search-engine extracts of docs/code rather than raw page fetches (no fetch tool available in this run); model_prices_and_context_window.json (1.8MB, binary on GitHub) could not be grepped to prove absence of supports_logprobs keys.",
    "No litellm installation or repo file was inspected locally; the exact passthrough code path in litellm/proxy/litellm_pre_call_utils.py and litellm/utils.py::get_optional_params is inferred from docs + issues."
  ],
  "noStagedFiles": true,
  "diffSummary": "Added one research artifact (research.md) with VERDICT/CONFIG/CAVEATS, numbered findings, source list, gaps, and severity roll-up. No source code changes.",
  "reviewFindings": [
    "no blockers",
    "advisory: docs.litellm.ai/docs/completion/drop_params — do not ship drop_params: true on a deployment used by this classifier; it silently deletes logprobs instead of 400ing.",
    "advisory: github.com/BerriAI/litellm/issues/18039 (closed-as-stale, unfixed) — avoid relying on extra_body for chat_template_kwargs without a wire-level check.",
    "advisory: github.com/BerriAI/litellm/issues/18801 (open) — streaming+logprobs fails on vLLM backends."
  ],
  "manualNotes": "The question's premise about supports_logprobs/supports_top_logprobs capability flags is inaccurate for LiteLLM: capability is expressed as OpenAI param-name strings via litellm.get_supported_openai_params(); the structured ProviderCapabilities PR (#30032) is still open. Suggested next step is the curl probe in the artifact against the live deployment plus an echo upstream."
}
```
