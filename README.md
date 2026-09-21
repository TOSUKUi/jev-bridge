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

1. Puts the **state in a shared system message** and each question in a user
   turn, so all questions of one request share an identical prefix — backends
   with prefix caching (SGLang radix cache, vLLM APC) prefill the state once.
2. Appends the answer options as single-token labels (`A`/`B`/`C…`, `true`/`false`,
   `0`/`1`/`2…`) and asks for exactly one label.
3. Requests `max_tokens=1` with `logprobs` + `top_logprobs` and computes a
   **restricted softmax over the labels** — the probabilities sum to 1 over the
   declared options, like Jev's answers do.
4. Derives `choice`/`score`/`noul` and a `confidence` value per answer.

No fine-tuning, no calibration training, no output-token decoding.

## Quick start

```bash
pip install jev-bridge            # or: pip install -e ".[dev]" from a checkout

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

The same shape covers SGLang (`python -m sglang.launch_server --model …`) — that
is the configuration the [Performance](#performance) numbers were measured on.

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

Measured end to end, warm, on a single **RTX PRO 6000** running
**Qwen3.8-Flash-Next** on an OpenAI-compatible endpoint (vLLM/SGLang served, a
LiteLLM gateway in front, `chat_template_kwargs` honoured so thinking is off),
median of 20 runs through the bridge:

```
1 question  (noul)               111 ms   (p10 105 / p90 138)
3 questions (choice+noul+score)  201 ms   (p10 193 / p90 208)
6 questions                      366 ms   (p10 357 / p90 383)

4x 3q at once                    138 ms/request   (550 ms wall for four)
8x 3q at once                    135 ms/request   (1079 ms wall for eight)
```

The questions of one request are issued concurrently (`JEVB_MAX_CONCURRENCY`,
default 8), and what is left of that growth belongs to the backend rather than to
the bridge: the same endpoint answers a lone first-token call in 106 ms serially
and tops out around 16–22 calls/s under concurrency (3 in flight → 191 ms wall,
12 → 548 ms). Bridge overhead sits inside the noise of that 111 ms.

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
  questions of one request run concurrently; with prefix caching the state is
  prefilled once.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

## License

MIT — see [LICENSE](LICENSE). `jev-bridge` is an independent project and is
not affiliated with TypeSafe AI.
