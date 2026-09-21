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
| `JEVB_BACKEND_EXTRA_BODY` | — | JSON merged into each chat-completions body, e.g. `{"chat_template_kwargs":{"enable_thinking":false}}` |
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

Thinking models emit `<think>` as their first token, which is not an answer
label. `jev-bridge` handles this with **two layers, on by default**:

1. **`enable_thinking: false`** is sent via `chat_template_kwargs` so the
   backend's own chat template opens the assistant turn with an empty
   `<think>\n\n</think>\n\n` block — this is the reliable path on Qwen3-style
   templates (SGLang/vLLM). If the backend rejects the field (HTTP 400/422),
   the bridge automatically retries without it and remembers that.
2. **Assistant prefill `<think></think>`** is appended to the messages, which
   covers backends/templates where the server-side flag is unavailable.

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

Measured end to end (HTTP server, warm) on a single RTX PRO 6000 with
**Qwen3.8-Flash-Next** served by SGLang (OpenAI-compatible endpoint):

```
1 question  (noul)              median ~83–111 ms
3 questions (choice+noul+score) median ~161–228 ms   <- all three in parallel
```

(Range across warm runs / cache states; all three questions of a request run
concurrently, so three questions cost far less than 3x one question.)

Reproduce with `python examples/bench.py http://127.0.0.1:8900`. For
comparison, TypeSafe reports 70–500 ms for hosted Jev and a community
measurement found 3 batched questions at ~212 ms.

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
