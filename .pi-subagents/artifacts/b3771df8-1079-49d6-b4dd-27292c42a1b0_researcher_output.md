

Wrote findings to `/home/amemiya/work/jev-bridge/.pi-subagents/artifacts/outputs/b3771df8-1079-49d6-b4dd-27292c42a1b0/research.md`.

---

**VERDICT: works-with-config**

`logprobs`/`top_logprobs` survive an `openai/<name>` + `api_base` round trip on non-streaming calls. Two conditions: never run this deployment with `drop_params: true`, and verify `chat_template_kwargs` on the wire (that part is documented-but-fragile). Current line is v1.100.0 ([release notes](https://docs.litellm.ai/docs/release_notes/)).

**CONFIG**
```yaml
model_list:
  - model_name: my-judge
    litellm_params:
      model: openai/qwen3-8b            # openai/ prefix = call an OpenAI-compatible /chat/completions
      api_base: http://vllm-host:8000/v1
      api_key: os.environ/UPSTREAM_API_KEY   # official openai client requires a key; any non-empty string
litellm_settings:
  drop_params: false          # default — keep false; true = silent loss of logprobs (200 OK, no logprobs)
```
If `chat_template_kwargs` is stripped, send it inside `extra_body` (documented REST passthrough form); last resort is a LiteLLM [pass-through endpoint](https://docs.litellm.ai/docs/proxy/pass_through) (cost tracking lost).

**CAVEATS**
- `supports_logprobs`/`supports_top_logprobs` are **not** LiteLLM flags. Capability = param-name strings from `litellm.get_supported_openai_params(model, custom_llm_provider)`; code at [`litellm/litellm_core_utils/get_supported_openai_params.py`](https://github.com/BerriAI/litellm/blob/main/litellm/litellm_core_utils/get_supported_openai_params.py). Structured capabilities are still an open PR: [#30032](https://github.com/BerriAI/litellm/pull/30032). Docs ✅ only for OpenAI/Azure/xAI/OVHCloud; blank for Anthropic, Cohere, Bedrock, Vertex, Ollama, TogetherAI — [Input Params](https://docs.litellm.ai/docs/completion/input).
- Unsupported param ⇒ **exception → 400**, never silent drop; `drop_params` defaults to **False**. With `drop_params: true` you instead get a 200 with no `logprobs` key at all — [drop_params](https://docs.litellm.ai/docs/completion/drop_params).
- The proxy rebuilds the response through LiteLLM's own Pydantic types (`ChatCompletionTokenLogprob`, `litellm/types/utils.py#L840`), not raw upstream bytes — that's why `top_logprobs` was once emitted as `null`: [#21932](https://github.com/BerriAI/litellm/issues/21932), fixed by [PR #22245](https://github.com/BerriAI/litellm/pull/22245) merged 2026-02-28 (`d524b795`, ≈v1.82.x → present in all 1.10x).
- Non-OpenAI top-level params are documented as passed through (`chat_template_kwargs` included), and `drop_params` **only drops unsupported OpenAI params**: [input params](https://docs.litellm.ai/docs/completion/input), [provider-specific params](https://docs.litellm.ai/docs/completion/provider_specific_params).
- But `extra_body` handling is buggy and was closed as `stale` (unfixed): non-standard params raise "provider does not support these parameters… set drop_params=True" and only work **double-nested** — [#18039](https://github.com/BerriAI/litellm/issues/18039). Same-family crash for `enable_thinking` via proxy: [#25697](https://github.com/BerriAI/litellm/issues/25697).
- Behaviour depends on provider registration; JSON-configured providers aren't in `openai_compatible_providers` and take a different path — open [#26443](https://github.com/BerriAI/litellm/issues/26443). Unknown-param preservation is still an unmerged opt-in PR: [#31697](https://github.com/BerriAI/litellm/pull/31697).
- `openai_extra_params` is **not a LiteLLM setting** (no doc/code trace). Wildcard config (`openai/*`) affects routing only, not body params — [wildcard routing](https://docs.litellm.ai/docs/wildcard_routing).
- Streaming + `logprobs` on vLLM backends crashes with `PydanticSerializationError` — **open** [#18801](https://github.com/BerriAI/litellm/issues/18801). Gemini/Vertex return no logprobs and Vertex 3+ falsely advertises them: [#8890](https://github.com/BerriAI/litellm/issues/8890), open [PR #31642](https://github.com/BerriAI/litellm/pull/31642).
- Guardrails/logging: the post-call guardrail/logging pipeline was reworked ([#24135](https://github.com/BerriAI/litellm/pull/24135), [#23910](https://github.com/BerriAI/litellm/pull/23910)); no evidence of logprob stripping from the client-visible body found either way — re-test with guardrails on.
- Decisive check beats doc reading: curl the proxy and count `top_logprobs` (expect 20), then hit an echo/mock upstream to confirm `chat_template_kwargs` actually arrived — logprobs surviving does not prove the param was forwarded.