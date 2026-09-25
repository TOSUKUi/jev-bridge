# jev-bridge

**A Jev-style `/v1/systemone` API in front of any OpenAI-compatible LLM server.**

[日本語版 README はこちら](README.ja.md)

[TypeSafe AI's Jev](https://typesafe.ai) is a "System One" model: you send it a
piece of *state* plus typed *questions*, and it returns typed answers with
probabilities — no generated prose. Jev itself is a closed, hosted model.
`jev-bridge` reproduces its **API surface and decision semantics** on top of
any OpenAI-compatible backend (SGLang, vLLM, llama.cpp, LMDeploy, TGI, …) by
reading the model's next-token distribution for the answer labels instead of
generating JSON.

```
client ──POST /v1/systemone──▶ jev-bridge ──POST /chat/completions──▶ SGLang / vLLM / llama.cpp
         (Jev wire format)                  (max_tokens=1, logprobs)
```

## Why this works

Classification with an instruct LLM is usually done by prompting for JSON and
praying. `jev-bridge` instead:

1. Puts the **state in the first user turn** (the system message is one fixed
   string with no state in it) and each question in the same turn after the
   state, so all questions of one request share an identical byte prefix —
   backends with prefix caching (SGLang radix cache, vLLM APC) reuse the state
   prefill. See [Prompt caching](#prompt-caching) for what is measured.
2. Appends the answer options as single-token labels (`A`/`B`/`C…`, `true`/`false`,
   `0`/`1`/`2…`) and asks for exactly one label.
3. Requests `max_tokens=1` with `logprobs` + `top_logprobs` and computes a
   **restricted softmax over the labels** — the probabilities sum to 1 over the
   declared options, like Jev's answers do.
4. Derives `choice`/`score`/`noul` and a `confidence` value per answer.

No fine-tuning, no calibration training, no output-token decoding.

## Quick start

```bash
pip install git+https://github.com/TOSUKUi/jev-bridge.git   # or: pip install -e ".[dev]" from a checkout

export JEVB_BACKEND_BASE_URL="http://127.0.0.1:30010/v1"   # any OpenAI-compatible server
export JEVB_BACKEND_MODEL="qwen3.8-flash-next"
jev-bridge serve --host 0.0.0.0 --port 8900
```

No server of your own? Point it at the official OpenAI API instead (add
`JEVB_BACKEND_API_KEY`, `JEVB_DISABLE_THINKING=0`, `JEVB_PREFILL_ASSISTANT=0`) —
[Docker](#docker) works through OpenAI / llama.cpp / vLLM.

Ask a question:

```bash
curl -s http://127.0.0.1:8900/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "jev-latest",
  "state": "My card was charged twice for order A-104 and I need this fixed today.",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing":  "Charges, invoices, payment problems",
        "shipping": "Delivery status, delays, lost packages",
        "returns":  "Exchanges, refunds, damaged items"
      }
    },
    "is_urgent": {"type": "noul", "instructions": "Does this message express urgency?"},
    "severity": {
      "type": "score",
      "instructions": "How severe is the issue?",
      "criteria": ["Cosmetic", "Workaround exists", "Blocking, no workaround"]
    }
  }
}'
```

Response:

```json
{
  "model": "jev-latest",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "confidence": 0.9813,
      "probabilities": {"billing": 0.9934, "shipping": 0.0033, "returns": 0.0033}
    },
    "is_urgent": {"type": "noul", "noul": 0.9987},
    "severity": {
      "type": "score",
      "score": 2.0,
      "confidence": 0.9971,
      "probabilities": {"0": 0.0008, "1": 0.0014, "2": 0.9978},
      "legend": {"0": "Cosmetic", "1": "Workaround exists", "2": "Blocking, no workaround"}
    }
  },
  "usage": {"prompt_tokens": 1234, "completion_tokens": 3, "questions": 3, "elapsed_ms": 187}
}
```

> `choice` probabilities are keyed by the option names from `criteria`, and
> `score` probabilities by level index (`"0"`, `"1"`, …), matching Jev.
> `noul` has no separate confidence field — its probability is the answer.

## Wire format

`POST /v1/systemone`

| field | type | notes |
|---|---|---|
| `model` | string, optional | echoed back; the backend model is chosen by `JEVB_BACKEND_MODEL` |
| `state` | string or JSON | anything the model should judge |
| `questions` | object | map of question name → question |

Question types (matching Jev's primitives):

| type | fields | answer |
|---|---|---|
| `choice` | `instructions`, `criteria` = map of option → description (≤ 255) | `choice`, `confidence`, `probabilities` |
| `noul` | `instructions`, optional `criteria` = `{true, false}` descriptions | `noul` = P(true) |
| `score` | `instructions`, `criteria` = ordered list of 2–10 levels (lowest first) | `score` = Σ i·pᵢ (may fall between rungs), `confidence`, `probabilities`, `legend` |

## Configuration

| env | default | meaning |
|---|---|---|
| `JEVB_BACKEND_BASE_URL` | — (required) | OpenAI-compatible base URL, e.g. `http://host:8000/v1` |
| `JEVB_BACKEND_MODEL` | `default` | model name sent to the backend |
| `JEVB_BACKEND_API_KEY` | `dummy` | bearer token for the backend |
| `JEVB_BACKEND_TIMEOUT` | `60` | per-call timeout (seconds) |
| `JEVB_MAX_CONCURRENCY` | `8` | in-flight question calls per request |
| `JEVB_TOP_K` | `20` | minimum `top_logprobs` requested (raised to cover all options) |
| `JEVB_TEMPERATURE` | `0.0` | `temperature` sent to the backend; rescales every reported probability (see below) |
| `JEVB_LOGIT_BIAS` | — | `{"<token>": <bias>}`, additive bias on the logits themselves, e.g. `{"A": -0.5}` |
| `JEVB_CONFIDENCE_METHOD` | `linear` | `linear` \| `max_prob` \| `entropy` |
| `JEVB_PREFILL_ASSISTANT` | `1` | append `<think></think>` to suppress thinking-model preamble |
| `JEVB_DISABLE_THINKING` | `1` | send `chat_template_kwargs: {enable_thinking: false, preserve_thinking: false}`; auto-retries without it if the backend rejects it |
| `JEVB_BACKEND_EXTRA_BODY` | — | JSON merged into each chat-completions body, e.g. `{"chat_template_kwargs":{"enable_thinking":false}}`. Naming `max_completion_tokens` here also selects the field that carries the token budget |
| `JEVB_MODEL_NAME` | `jev-bridge-1` | reported model name when the request has no `model` |
| `JEVB_ALLOW_LOCAL_IMAGES` | `0` | allow local file paths in `images` (off = data URI / URL / base64 only) |
| `JEVB_MAX_IMAGE_BYTES` | `20971520` (20 MiB) | per-image size cap after decode |
| `JEVB_HOST` / `JEVB_PORT` | `0.0.0.0` / `8900` | listen address for `jev-bridge serve` |

### Confidence

`linear` is the default: `(K·p_max − 1)/(K − 1)`. It maps a uniform
distribution to 0 and a one-hot distribution to 1, and it matches the
confidence values observable in Jev's published examples (e.g. `{0.0, 0.7,
0.3}` → ≈0.54–0.55). `max_prob` and `entropy` (`1 − H/lnK`) are also available.

**Confidence here is a distribution-shape statistic, not Jev's RLCD-calibrated
confidence.** Treat thresholds as drift-prone and validate them on your task.

### `JEVB_TEMPERATURE` / `JEVB_LOGIT_BIAS`

`temperature` is applied by the backend **before** it computes the returned
logprobs, so it rescales every reported probability: at `0.5` the payload matches
the p² renormalisation to three decimals (0.053/0.503/0.444 → 0.006/0.559/0.435),
and `2.0` turns 0.996/0.003/0.001 into 0.921/0.052/0.027. `logit_bias` is added to
the logits. Both are echoed by `/health`.

The defaults (`0.0`, no bias) are the best measured configuration on the JevBench
public tiers: hard accuracy is 88/111 on every rep at `0.0`, and `1.3` drops to
86–87/111 without buying calibration (ECE 0.065–0.087 → 0.075–0.093). A bias term
has nothing to correct either — the option positions the model picks track the gold
positions (17.9/28.4/25.4/23.9/4.5 % vs 22.4/22.4/31.3/17.9/6.0 % on hard).

## Prompt caching

The prompt is laid out for prefix caching. Every question of a request sends the
same bytes up to the question block:

```
[system]  fixed instruction text, no state
[user]    [images…] + "STATE:\n" + <serialized state> + "\n\n" + <question block>
[assistant] "<|im_start|></think>"   (JEVB_PREFILL_ASSISTANT)
```

The serialized state is produced once per request, there are no ids, timestamps
or seeds in the body, and httpx serialises with a fixed key order — so the same
input produces the same bytes, and `tests/test_api.py` asserts that the N
per-question calls share one prefix. One consequence worth knowing:
`serialize_state` re-dumps a dict state with `indent=2` and **wire key order**,
so a client that sends the same state with reordered keys gets a different
prefix (and a different prompt).

Measured on the Qwen3.8-Flash-Next endpoint described in
[Performance](#performance), one token out:

| prompt | cold | warm |
|---|---|---|
| ~110 tokens (typical template) | 106 ms | 106 ms |
| ~1.4k tokens of state | 190 ms | 135 ms |
| ~2.8k tokens of state | 293 ms | — |

So a cached prefix is close to free (~+30 ms for 2k tokens) while a cold prefill
costs roughly +40 ms per extra 1k tokens. Concurrency is what makes the warm path
worth having: four questions over a **warm** prefix took **215 ms concurrent** vs
**468 ms sequential**, so keep the fan-out once the state is warm (and the warm
hit is served from a brand-new connection pool too — 225 ms — so it is not pinned
to a connection). The cold case is the one to know about: the same four questions
over the *same cold* 2.7k-token prefix took **665 ms sequential** (per request
114 / 117 / 119 / 287 — one prefill, three hits) and **880 ms concurrent**. That
concurrent figure is consistent with the burst being admitted before any prefill
is committable, i.e. several prefills of the same prefix. Note the epistemic
status: this is inferred from wall-clock. The server was not started with
`--enable-cache-report`, so `usage` carries no `cached_tokens` and no measurement
here observes the cache directly.

The one lever is `JEVB_MAX_CONCURRENCY=1` — every question sequential, so each one
hits the prefix the previous one warmed. Good for long cold states, bad for warm
throughput (468 ms vs 215 ms above).

A first-question-then-fan-out primer was implemented, measured, and **removed**. On
a synthetic prompt (2.7k-token state, one-line questions) it won — 501 ms including
the primer vs 841 ms unprimed — but on the prompts this bridge actually sends it
lost at every size tested: +79 ms at 4 questions, +92 ms at 6, and +92 ms replaying
the captured bodies (699 → 791 ms). The primer is one of the N calls doing the same
prefill, so it saves at most (N−1)/N of that work and costs a full sequential round
trip up front; with the bridge's question blocks being a few hundred tokens rather
than one line, that trade is negative. Fan-out is unconditional.

If you want to warm a prefix yourself, the cheap call is `max_tokens: 0` — SGLang
runs it prefill-only (`completion_tokens: 0`, no decode step scheduled), llama.cpp
documents the same as `n_predict: 0`. The cost of such a call here was 234–448 ms
cold for a 4.1k-token prefix and 117 ms once warm. Whether it buys you anything
depends on what follows it: on the 2.7k-token single-line-question prompt the
following 4-question burst went 841 → 217 ms, while on the two-tier shape (4.1k
judgment JSON + a ~500-token varying context) the burst went 849 → 490 ms but the
primer's own 448 ms made the combined 938 ms *worse* than not warming at all.
Warm ahead of the request, when a spec changes or on idle, and treat it as a GPU
expense rather than a free one — it competes with live traffic. Two traps: it must
be spelled `max_tokens` (see the SGLang note below), and `cached_tokens` stays
invisible in `usage` unless the server was started with `--enable-cache-report` —
a missing field is not a cache miss.

The match is a plain forward match on the **rendered token sequence from token 0**
— not per message. Reuse length is the longest common prefix with something
already cached, cut off at a block/page boundary (vLLM caches full blocks only,
SGLang inserts at `page_size`), and a block is only insertable once its prefill
has run. Consequences, measured on this endpoint with a ~1.9k-token seed:

| prompt | ms | prompt tokens |
|---|---|---|
| seed, first touch | 233 | 1941 |
| same prefix, 0 appended | 126 | 1941 |
| same prefix, +910 tokens appended | 152 | 2851 |
| same prefix, +1860 tokens appended | 216 | 3801 |
| same prefix, only the question changed | 128–131 | 1946 |
| one nonce inserted at the front of the state | 214–219 | 1945 |

Appending is nearly free (you pay the new tail); moving the divergence point
costs everything after it. The nonce case re-prefilled the whole prompt because
the divergence sat at the start of the user turn and the shared system block is
only a few dozen tokens — a change further down would have kept everything above
it. Watch for nonces, timestamps, turn counters and reordered JSON keys. Eviction
is documented to drop the deepest tail first (SGLang evicts radix leaves, vLLM
frees in reverse order), which is why the long context is what you tend to lose
under memory pressure; we did not measure retention here.

If the stable part of your prompt is the **question spec** and the state is what
changes (a game loop, say), pass one string as `state` with the spec first:
`state = "<judgment JSON>\n\n<current context>"`. `serialize_state` returns a
string untouched, so the cached prefix becomes system+spec, and the spec half is
reused across every request while only the context tail re-prefills — measured on
a 4.1k-token spec: first request 519 ms, same spec with a fresh context 173 ms,
and a `max_tokens: 0` primer on the spec alone takes a 4-question concurrent burst
from 849 ms to 490 ms.

Treat that as a **prompt change, not a config flag**: it needs its own scoring
regression before you ship it. The spec lands inside the untrusted `STATE` region
(the only separator is a newline, and nothing enforces precedence between your
instructions and the payload); a spec duplicated into `state` gives you two
sources of truth that can disagree while the wire format still looks Jev-valid;
images are rendered *before* the state, so the claim is void the moment you attach
one; and a spec long enough to disturb the answer still returns HTTP 200 with tidy
probabilities, because labels absent from the top-K get a floor and are
normalised. Measure accuracy on labelled cases before and after, not just latency.

What the bridge does **not** send: `prompt_cache_key`, a session id, or any
custom header (there is no header configuration; a constant body field can be
injected with `JEVB_BACKEND_EXTRA_BODY`). With OpenAI that is usually moot —
prompt caching is automatic for prompts of **1024 tokens or more**
(`usage.prompt_tokens_details.cached_tokens` reports the hit, eviction after ~5–10
min idle), and a typical jev-bridge template is ~110–200 tokens, i.e. below the
threshold and never cached. OpenAI's own guidance to set `prompt_cache_key` is
about pinning requests that share a prefix to the same cache — relevant as soon
as a gateway fans one state's questions out over several replicas.

## Thinking models (Qwen3.x, …)

**The principle: the score comes from the first sampled token.** Anything the
model emits before the answer label costs the request its signal, so reasoning
has to be switched off at the source. `jev-bridge` does this with **two layers,
on by default**:

1. **`chat_template_kwargs: {"enable_thinking": false, "preserve_thinking": false}`**
   makes the backend's own chat template open the assistant turn with an empty
   `<|im_start|>\n\n</think>\n\n` block — this is the switch that actually works on
   Qwen3-style templates (SGLang, vLLM, current llama.cpp). If the backend
   *rejects* the field (HTTP 400/422) the bridge automatically retries without it
   and remembers that; if it silently *ignores* it, no retry fires and you get
   the flat answers described below.
2. **Assistant prefill `<|im_start|>assistant` `<|im_start|>\n\n</think>\n\n`**
   is appended to the messages. This only helps where the server honours
   continuation prefill (llama.cpp does) — measured through a LiteLLM proxy, an
   endpoint that reads the assistant turn as plain context gained nothing from it.

Backends that speak `reasoning_effort` instead of template kwargs take it through
`JEVB_BACKEND_EXTRA_BODY`:

```bash
export JEVB_BACKEND_EXTRA_BODY='{"reasoning_effort":"none"}'
```

Two measured caveats: an effort *level* is not an off-switch (`low` still opened
with reasoning text on a Qwen3 reasoner), and a value the server does not accept
comes back as HTTP 400 surfaced as 502 — the backend's own message names the
values it accepts.

**What "reasoning is not off" looks like in the answers:** every option lands on
the same probability and confidence collapses —

```
choice → probabilities {"a": 0.3333, "b": 0.3333, "c": 0.3333}, confidence 0.0      noul → 0.5
```

A flat distribution on a model you expect to be confident means this, not a bad
model. Behind a proxy, verify it on the wire: `logprobs` coming back does not
prove `chat_template_kwargs` arrived (and LiteLLM with `drop_params: true` drops
`logprobs` while still answering 200).

Tune with `JEVB_DISABLE_THINKING=0` and/or `JEVB_PREFILL_ASSISTANT=0`. To pass
custom template kwargs instead, use `JEVB_BACKEND_EXTRA_BODY` — it takes
precedence over the built-in defaults.

## Docker

`docker-compose.yml` at the repo root is the default deployment: the **official
OpenAI API**.

```bash
export OPENAI_API_KEY=sk-...
docker compose up
curl -s http://127.0.0.1:8900/health
```

The compose file builds the image from this checkout (`build: .` tagged
`jev-bridge:latest`) — replace those two lines with a published image once you
have one. Switching backend is only ever a matter of
`JEVB_BACKEND_BASE_URL` + `JEVB_BACKEND_MODEL` (+ `JEVB_BACKEND_API_KEY` if the
server checks a token) — by editing the compose file, dropping a
`docker-compose.override.yml` next to it, or using the `docker run` form below.
The three examples below are the same bridge and the same questions; what
differs is what each backend wants to be told.

### Official OpenAI

This is exactly what the compose file ships. `chat_template_kwargs` is a
vLLM/SGLang extension and the Chat API has no assistant prefill, so both
thinking-suppression layers are switched off; the model must support `logprobs`,
and `top_logprobs` is capped at **20** — which is already the `JEVB_TOP_K`
default, so there is nothing to raise. Equivalent one-liner:

```bash
docker run --rm -p 8900:8900 \
  -e JEVB_BACKEND_BASE_URL=https://api.openai.com/v1 \
  -e JEVB_BACKEND_MODEL=gpt-4o-mini \
  -e JEVB_BACKEND_API_KEY="$OPENAI_API_KEY" \
  -e JEVB_DISABLE_THINKING=0 \
  -e JEVB_PREFILL_ASSISTANT=0 \
  jev-bridge
```

#### Which OpenAI models work

Scoring needs the **first sampled token's logprobs**, and it needs more than one
candidate to compare. Measured against the Chat Completions API:

| models | verdict |
|---|---|
| `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano` | work as shipped; `top_logprobs` honoured up to 20 |
| `gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-5.6-luna`, `gpt-5.6-terra`, `gpt-5.6-sol` | work with the recipe below (reasoning family) |
| `o4-mini`, `gpt-5-nano` | unusable: `You are not allowed to request logprobs from this model` |

The GPT-5.x reasoning models need three things that are model policy, not bridge
settings:

```bash
export JEVB_BACKEND_EXTRA_BODY='{"reasoning_effort":"none","max_completion_tokens":8}'
export JEVB_TOP_K=5
```

* `logprobs` is refused on a reasoning model until reasoning is off:
  `Unsupported parameter: 'logprobs' is not supported with this model`.
  `reasoning_effort: "none"` unlocks it; a *level* does not (`low` still refuses).
* `max_tokens` is rejected in favour of `max_completion_tokens` — naming the
  latter in `JEVB_BACKEND_EXTRA_BODY` makes the first request valid (a backend that
  only tells you at runtime is handled by an automatic retry). A budget of 1 token
  is refused outright (`Could not finish the message`); 4 or more works.
* `top_logprobs` is capped at **5**, so keep `JEVB_TOP_K=5` — that is 5 options max.

**The catch: the candidate list comes back truncated.** OpenAI returns only the
tokens carrying real probability mass, so a confident answer arrives as a single
entry (`{"A": -0.0}`) and every other label falls to the bridge's floor value:
`{"billing": 0.9993, "shipping": 0.0003, "returns": 0.0003}` — the same numbers
however the model leaned. `gpt-4o-mini` on the same ambiguous state returns
`{"billing": 0.148, "shipping": 0.0, "returns": 0.852}`. It is truncation, not a
one-token policy (a coin-flip prompt returns two entries, `-0.34 / -1.25`), but
the consequence stands: on the 5.x models the **argmax** is trustworthy while
`probabilities`, `confidence`, thresholds and `default_when` are endpoints rather
than measurements.

Switching to the **Responses API** buys nothing here: the refusal is a model
policy and reads the same there (`logprobs are not supported with reasoning
models`). The bridge speaks `chat/completions`, which is also what llama.cpp,
vLLM and SGLang speak.

### llama.cpp (`llama-server`)

Nothing to configure. Current builds read `chat_template_kwargs` and map
`enable_thinking` onto the chat template; older builds ignore the field, and the
assistant prefill (on by default) covers those. `top_logprobs` is translated to
llama.cpp's `n_probs` (default 20) with no OpenAI-style cap, but a large top-K
per token costs bandwidth — raise `JEVB_TOP_K` only when a question really needs
it. Give the server a slot per concurrent question (`--parallel` ≥
`JEVB_MAX_CONCURRENCY`).

```bash
llama-server -m Qwen3-8B-Instruct-Q4_K_M.gguf -c 16384 --port 8080 --parallel 8

docker run --rm -p 8900:8900 \
  -e JEVB_BACKEND_BASE_URL=http://host.docker.internal:8080/v1 \
  -e JEVB_BACKEND_MODEL=qwen3-8b-instruct \
  --add-host=host.docker.internal:host-gateway \
  jev-bridge
```

### vLLM

`chat_template_kwargs` is understood, so thinking models are handled by the
defaults. `--max-logprobs` caps `top_logprobs` (default **20**, the same as the
bridge's `JEVB_TOP_K`) — raise it when you need more than 20 options, and keep
prefix caching on so the shared state is prefilled once.

```bash
vllm serve Qwen/Qwen3-8B --max-logprobs 64 --enable-prefix-caching

docker run --rm -p 8900:8900 \
  -e JEVB_BACKEND_BASE_URL=http://host.docker.internal:8000/v1 \
  -e JEVB_BACKEND_MODEL=Qwen/Qwen3-8B \
  -e JEVB_TOP_K=64 \
  --add-host=host.docker.internal:host-gateway \
  jev-bridge
```

The same shape covers SGLang (`python -m sglang.launch_server --model …
--enable-cache-report`) — that is the configuration the
[Performance](#performance) numbers were measured on, behind a LiteLLM gateway.
Two SGLang specifics worth knowing:

* `max_tokens: 0` is a genuine prefill-only request (SGLang's `is_prefill_only`
  path schedules no decode step, and the response carries `completion_tokens: 0`),
  and it warms the radix cache. `max_completion_tokens: 0` is **not** the same
  request: the field is read as `max_completion_tokens or max_tokens`, so the 0
  vanishes and the budget becomes `1 << 30` — measured, it generated 292 tokens.
  When you mean zero, send `max_tokens`. (Speculative decoding disables the
  prefill-only fast path.)
* `usage.prompt_tokens_details.cached_tokens` appears only with
  `--enable-cache-report`; otherwise a cache hit leaves no trace in `usage`.

## CLI

```bash
jev-bridge serve [--host H] [--port P]        # run the proxy
jev-bridge probe http://host:8000/v1          # score 3 sample questions directly against a backend
jev-bridge probe http://localhost:8900/v1/systemone   # end-to-end check of a running bridge
```

## Image input (extension)

Jev itself is text-only, so `images` is a jev-bridge extension. Add a
top-level `images` array next to `state`; images and state text share the
same user turn (templates reject images in system messages), and the question
stays at the end so prefix caching still reuses the image prefill:

```json
{
  "state": "What is shown in this photo?",
  "images": ["data:image/png;base64,iVBOR…"],
  "questions": {
    "scene":     {"type": "choice", "instructions": "Where is this?", "criteria": {"indoor": "…", "outdoor": "…"}},
    "has_people": {"type": "noul", "instructions": "Are people visible?"}
  }
}
```

Accepted image references:

| form | example |
|---|---|
| data URI | `data:image/png;base64,iVBOR…` |
| URL | `https://example.com/photo.jpg` (backend fetches it) |
| bare base64 | `iVBORw0KGgo…` (format sniffed from magic bytes) |
| local path | `/path/to.png` — **only with `JEVB_ALLOW_LOCAL_IMAGES=1`** |

Notes:

* The format is re-sniffed from the payload; a mislabeled data URI is corrected,
  and a data URI whose bytes are not a recognized image is rejected (400).
* Native formats: PNG, JPEG, GIF, WEBP, BMP. Other formats must be converted
  by the caller; per-image decode cap `JEVB_MAX_IMAGE_BYTES` (default 20 MiB).
* The backend model must be vision-capable — on a text-only model the backend
  rejects the request and the bridge returns 502 with the backend's message.
* Image tokens are part of the shared prefix, so N questions about one image
  still prefill the image once.

## Performance

Measured end to end, warm, on a single **RTX PRO 6000** serving
**Qwen3.8-Flash-Next** with **SGLang**, through the bridge, 10 runs each:

```
                       direct SGLang     via LiteLLM gateway
1 question  (noul)           81 ms              91 ms
3 questions                 158 ms             169 ms
6 questions                 320 ms             340 ms
```

The second column is the same SGLang server reached through a **LiteLLM** gateway
(`chat_template_kwargs` is honoured on both routes, so thinking is off). The hop
is worth ~10 ms at one question and ~20 ms at six — real, but not where the time
goes. These figures replace an earlier 111 / 201 / 366 ms set taken on another
day; treat single-digit percentages as day-to-day noise and re-measure on your own
hardware.

### Reproducibility

At `temperature: 0` the endpoint is **not bit-reproducible** on this machine. The
same request body, twelve times in a row:

| questions per request | p(the top option) over 12 runs | peak-to-peak |
|---|---|---|
| 1 | 0.667 … 0.738 | 0.071 |
| 6 | 0.6985 every time | 0.000 |

Greedy decoding fixes the *sampling*, not the arithmetic: continuous batching
changes the batch a sequence is folded into, and reduction order moves logits by
just enough to matter near a tie. Do not treat the fourth decimal as a
measurement, and give thresholds real margins. (The narrow reading of the table:
under this shared load, a six-question request came back identical across twelve
runs while a single-question one drifted. It is not evidence that concurrency
improves determinism, and the 6-question row is six independent calls sharing a
prefix — not one joint prompt.)

### What is left on the table

Four measured dead ends, so you don't spend a week on them:

* **Per-question cost.** ~47 ms per extra question when fired back to back, ~63 ms
  when fired with 3 s gaps between requests — i.e. leaving it idle makes the slope
  *worse*, so this is not burst queueing. Attribution is unmeasured (no per-request
  scheduler or GPU instrumentation from the client side).
* **Compact state.** `json.dumps(..., separators=(",", ":"))` instead of `indent=2`
  cuts the prompt 2789 → 1707 tokens/call (-39%) and cold latency 965 → 839 ms
  (paired, `/flush_cache` before every shot). It did **not** win warm (one run said
  +3 ms, another -58 ms) and it moved probabilities by up to **0.27** and confidence
  by **0.41** on the same six questions with zero argmax changes. Whitespace is
  model input: pass a pre-serialized compact string if you want the tokens, but
  re-run your own scoring regression first.
* **`logit_bias`.** Honoured, direct and through the gateway alike, with the same
  returned bytes (`/tokenize {"prompt": "B"}` → id 33 on this tokenizer; `true`/`false`
  are single tokens 1802/3721). But the returned logprobs are conditioned on *not
  being a banned token*, renormalised over the rest of the vocabulary — not over
  your label set — so it can suppress an option, it cannot hand you measured
  probabilities for a set of labels. It does not replace the floor heuristic.
* **`top_logprobs` beyond 20.** This backend accepts 32/64/100 (their API is not
  capped like OpenAI's), yet across 3-option and 10-option probes every label was
  already inside the top 20, so the floor never fired and raising K moved
  probabilities by <= 0.11 — inside the run-to-run wobble above. Leave `JEVB_TOP_K`
  alone unless your labels genuinely fall outside the top 20.

Fan-in — every question in one prompt, read the label at each position — is the one
measured win (6 questions 137 ms vs 337 ms in the same run), and it is not offered: it scores
`P(label_i | state, all questions, earlier answers)` instead of
`P(label_i | state, question_i)`, which flipped the argmax on 2 of 6 questions and
moved a probability by as much as 0.55. A decision engine whose product is a probability
does not get to redefine it for a latency number.

The questions of one request are issued concurrently (`JEVB_MAX_CONCURRENCY`,
default 8), and *N questions at once* is a different quantity from *N questions in
a row*. Four simultaneous 3-question requests against the same warm endpoint:

```
                       wall     per-request p50   slowest    wall/N
direct SGLang          498 ms        370 ms         491 ms     124 ms
via LiteLLM            526 ms        397 ms         520 ms     132 ms
```

What a caller feels is the **per-request p50 (~370–400 ms)**, not the 124–132 ms:
wall/N is throughput written as if it were a latency. (An earlier revision of this
README published exactly that amortised figure as "median ms/request", which
understated user-visible latency by roughly 3×; `examples/bench.py` now prints
wall, p50, slowest and wall/N separately.) 24 backend calls in ~990 ms is ~24
calls/s, while the bridge's own share of a ~85 ms single-question budget stays
around 4 ms. If your gateway puts a `rpm` or `max_parallel_requests` cap on the
model group, that cap — not this number — is your ceiling; check it before
planning capacity around 24 calls/s.

Reproduce with `python examples/bench.py http://127.0.0.1:8900 20`. For
comparison, TypeSafe reports 70–500 ms for hosted Jev and a community
measurement found 3 batched questions at ~212 ms.

Same bridge against the **official OpenAI API** (`JEVB_DISABLE_THINKING=0`,
`JEVB_PREFILL_ASSISTANT=0`, median of 6 warm runs, network included):

```
gpt-4.1-mini   1q ~415 ms  3q ~560 ms     gpt-5.6-luna   1q ~608 ms  3q ~629 ms
gpt-4.1-nano   1q ~438 ms  3q ~523 ms     gpt-5.4-mini   1q ~662 ms  3q ~729 ms
gpt-4o-mini    1q ~583 ms  3q ~570 ms     gpt-5.6-terra  1q ~813 ms  3q ~926 ms
gpt-4o         1q ~597 ms  3q ~617 ms     gpt-5.6-sol    1q ~812 ms  3q ~1077 ms
```

The 5.x numbers need the [model recipe](#which-openai-models-work). Two things a
hosted API changes: latency is dominated by the round trip (~0.4 s here, so the
local numbers above are the ones to beat), and at `temperature: 0` an
unambiguous state comes back one-hot (`{billing: 1.0, shipping: 0.0, returns:
0.0}`), which leaves a threshold nothing to act on. Genuinely ambiguous states do
stay graded (`0.85 / 0.15 / 0.0`), so this is about where the operating point
lands, not about the model being unable to express doubt.

## Known differences from Jev

* Answers come from one LLM forward pass; there is no RLCD training. Quality
  tracks the backend model, and confidence is not calibrated.
* No streaming (Jev has none either) and no image input.
* `choice` supports up to 255 options, but scoring fidelity degrades beyond
  the backend's `top_logprobs` limit — options missing from the returned
  top-K receive a floor probability. Raise `JEVB_TOP_K` where the backend
  allows it (vLLM: `--max-logprobs`, SGLang: `top_logprobs_num`).
* Latency scales with question count only through backend batching: all
  questions of one request run concurrently, and prefix caching makes the shared
  state nearly free — except on the *first* request for a state, where fan-out
  pays the prefill more than once (measured below).

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

## License

MIT — see [LICENSE](LICENSE). `jev-bridge` is an independent project and is
not affiliated with TypeSafe AI.
