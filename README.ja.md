# jev-bridge

**[English README](README.md)**

**任意の OpenAI 互換 LLM サーバの前段に、Jev 形式の `/v1/systemone` API を生やすレイヤー。**

[TypeSafe AI の Jev](https://typesafe.ai) は「System One」モデルと呼ばれる判断特化モデルです。
*state*（判定対象の状況）と型付きの *questions*（質問）を送ると、文章生成ではなく
**型付きの回答と確率**を返します。Jev 本体はクローズドなホスト型モデルですが、
`jev-bridge` はその **API 形式と判断セマンティクス**を、任意の OpenAI 互換バックエンド
（SGLang / vLLM / llama.cpp / LMDeploy / TGI など）の上で再現します。
JSON を生成させるのではなく、**回答ラベルの次トークン分布（logprobs）を読む**方式です。

```
client ──POST /v1/systemone──▶ jev-bridge ──POST /chat/completions──▶ SGLang / vLLM / llama.cpp
         (Jev ワイヤ形式)                    (max_tokens=1, logprobs)
```

## 仕組み

instruct モデルで分類をするとき、普通は「JSON で答えてください」とプロンプトして祈るしかありません。
`jev-bridge` はそうではなく:

1. **state を共通の system メッセージ**に置き、各質問を user ターンにする。
   1リクエスト内の全質問が同一プレフィックスを共有するため、プレフィックスキャッシュを持つ
   バックエンド（SGLang の radix cache、vLLM の APC）では state の prefill が1回で済む。
2. 回答候補を**単一トークンのラベル**（`A`/`B`/`C…`、`true`/`false`、`0`/`1`/`2…`）として提示し、
   ラベルを1つだけ出力させる。
3. `max_tokens=1` + `logprobs` + `top_logprobs` で次トークン分布を取得し、
   **候補ラベル上の制限付き softmax** を計算する。確率は宣言した選択肢の上で和が1になる
   （Jev の回答と同じ契約）。
4. `choice` / `score` / `noul` と `confidence` を組み立てる。

ファインチューニング不要、較正学習不要、出力トークンのデコードも不要。

## クイックスタート

```bash
pip install jev-bridge            # チェックアウトからなら: pip install -e ".[dev]"

export JEVB_BACKEND_BASE_URL="http://127.0.0.1:30010/v1"   # 任意の OpenAI 互換サーバ
export JEVB_BACKEND_MODEL="qwen3.8-flash-next"
jev-bridge serve --host 0.0.0.0 --port 8900
```

質問を投げる:

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

レスポンス:

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

> `choice` の `probabilities` は `criteria` の選択肢名をキーにし、`score` は段階番号
> （`"0"`, `"1"`, …）をキーにします（Jev 準拠）。`noul` には独立した confidence フィールドはなく、
> 確率そのものが答えです。

## ワイヤ形式

`POST /v1/systemone`

| フィールド | 型 | 備考 |
|---|---|---|
| `model` | string, 任意 | レスポンスにエコーされるだけ。バックエンドのモデルは `JEVB_BACKEND_MODEL` で指定 |
| `state` | string または JSON | 判定対象。文字列でもオブジェクトでもよい |
| `questions` | object | 質問名 → 質問 のマップ |

質問タイプ（Jev のプリミティブに対応）:

| type | フィールド | 回答 |
|---|---|---|
| `choice` | `instructions`, `criteria` = 選択肢名 → 説明 のマップ（最大255） | `choice`, `confidence`, `probabilities` |
| `noul` | `instructions`, 任意で `criteria` = `{true, false}` の説明 | `noul` = P(true) |
| `score` | `instructions`, `criteria` = 2〜10段階の順序付きリスト（低い順） | `score` = Σ i·pᵢ（段階の間の値も取り得る）, `confidence`, `probabilities`, `legend` |

## 設定

すべて環境変数で指定します。

| 環境変数 | デフォルト | 意味 |
|---|---|---|
| `JEVB_BACKEND_BASE_URL` | —（必須） | OpenAI 互換の base URL（例 `http://host:8000/v1`） |
| `JEVB_BACKEND_MODEL` | `default` | バックエンドに送るモデル名 |
| `JEVB_BACKEND_API_KEY` | `dummy` | バックエンド用の Bearer トークン |
| `JEVB_BACKEND_TIMEOUT` | `60` | 1呼び出しあたりのタイムアウト秒 |
| `JEVB_MAX_CONCURRENCY` | `8` | リクエスト内の並列質問数 |
| `JEVB_TOP_K` | `20` | 要求する `top_logprobs` の最小値（選択肢数をカバーするよう自動で引き上げ） |
| `JEVB_CONFIDENCE_METHOD` | `linear` | `linear` \| `max_prob` \| `entropy` |
| `JEVB_PREFILL_ASSISTANT` | `1` | `<think></think>` プレフィルを assistant メッセージに付与 |
| `JEVB_DISABLE_THINKING` | `1` | `chat_template_kwargs: {enable_thinking: false, preserve_thinking: false}` を送る。バックエンドに拒否されたら自動で外して再試行 |
| `JEVB_BACKEND_EXTRA_BODY` | — | 毎回の chat-completions ボディにマージする JSON。例 `{"chat_template_kwargs":{"enable_thinking":false}}` |
| `JEVB_MODEL_NAME` | `jev-bridge-1` | リクエストに `model` がない場合に報告するモデル名 |
| `JEVB_HOST` / `JEVB_PORT` | `0.0.0.0` / `8900` | `jev-bridge serve` の待ち受けアドレス |

### confidence について

デフォルトは `linear`: `(K·p_max − 1)/(K − 1)`。一様分布を 0、one-hot を 1 に写し、
Jev の公開サンプルで観測できる confidence 値（例: `{0.0, 0.7, 0.3}` → ≈0.54〜0.55）に一致します。
`max_prob` と `entropy`（`1 − H/lnK`）も選べます。

**ここでの confidence は分布形状の統計量であり、Jev の RLCD 較正済み confidence ではありません。**
閾値はドリフトし得るものとして扱い、自分のタスクで検証してください。

## thinking モデル（Qwen3.x など）の扱い

thinking モデルは最初のトークンに `<think>` を出しますが、これは回答ラベルではありません。
`jev-bridge` はデフォルトで**2段構え**で対処します:

1. **`enable_thinking: false`** を `chat_template_kwargs` で送る。バックエンド側の
   chat template が assistant ターンを空の `<think>\n\n</think>\n\n` で開くため、
   Qwen3 系テンプレート（SGLang / vLLM）ではこれが確実な経路。バックエンドがこの
   フィールドを拒否した場合（HTTP 400/422）は自動で外して再試行し、以後その状態を記憶する。
2. **assistant プレフィル `<think></think>`** をメッセージ末尾に付与。サーバ側フラグが使えない
   バックエンド／テンプレートをカバーする。

`JEVB_DISABLE_THINKING=0` や `JEVB_PREFILL_ASSISTANT=0` で個別に切れます。
テンプレート引数を直接渡したい場合は `JEVB_BACKEND_EXTRA_BODY` を使います（組み込みの既定より優先）。

## 性能

単一の RTX PRO 6000 上で **Qwen3.8-Flash-Next** を SGLang で配信し、
HTTP サーバ経由・ウォーム状態で実測:

```
1問  (noul)              median ~83〜111 ms
3問  (choice+noul+score) median ~161〜228 ms   ← 3問は並列実行
```

再現: `python examples/bench.py http://127.0.0.1:8900`。
比較として、TypeSafe はホスト版 Jev で 70〜500 ms と公表しており、
コミュニティ計測では3問バッチで約212 ms という値もあります。

## 既知の差異（Jev 本体との違い）

* 回答は LLM の1フォワードパスから得たもので、RLCD 学習はありません。品質はバックエンド
  モデルに依存し、confidence は較正されていません。
* ストリーミングなし（Jev にもなし）、画像入力なし。
* `choice` は最大255選択肢に対応しますが、バックエンドの `top_logprobs` 上限を超えると
  スコア精度が落ちます（返却された top-K に現れない選択肢は下限確率を割り当て）。
  バックエンドが許すなら `JEVB_TOP_K` を上げてください
  （vLLM: `--max-logprobs`、SGLang: `top_logprobs_num`）。
* レイテンシは質問数の影響をほぼ受けません。1リクエスト内の質問は並列実行され、
  プレフィックスキャッシュにより state の prefill は1回だけです。

## Docker

```yaml
services:
  jev-bridge:
    image: ghcr.io/yourname/jev-bridge:latest   # または build: .
    environment:
      JEVB_BACKEND_BASE_URL: http://sglang:30010/v1
      JEVB_BACKEND_MODEL: qwen3.8-flash-next
    ports:
      - "127.0.0.1:8900:8900"
```

```bash
docker build -t jev-bridge .
docker run --rm -p 8900:8900 \
  -e JEVB_BACKEND_BASE_URL=http://host.docker.internal:30010/v1 \
  -e JEVB_BACKEND_MODEL=qwen3.8-flash-next \
  jev-bridge
```

## CLI

```bash
jev-bridge serve [--host H] [--port P]        # プロキシを起動
jev-bridge probe http://host:8000/v1 --model my-model   # バックエンドに直接3問を投げて確認
jev-bridge probe http://localhost:8900/v1/systemone     # 起動中のブリッジを E2E 確認
```

## 開発

```bash
pip install -e ".[dev]"
pytest -q
```

## ライセンス

MIT — [LICENSE](LICENSE) を参照。`jev-bridge` は独立したプロジェクトであり、
TypeSafe AI とは無関係です。
