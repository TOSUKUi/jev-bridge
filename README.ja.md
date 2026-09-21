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

自分でサーバーを起動しないなら、公式 OpenAI API を向いてもよい（`JEVB_BACKEND_API_KEY` に加えて
`JEVB_DISABLE_THINKING=0` と `JEVB_PREFILL_ASSISTANT=0` を追加）。
バックエンド別の設定は [Docker](#docker) を参照。

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
| `images` | array, 任意 | 画像入力（拡張）。data URI / http(s) URL / 生base64 / ローカルパス（要許可） |
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
| `JEVB_BACKEND_EXTRA_BODY` | — | 毎回の chat-completions ボディにマージする JSON。例 `{"chat_template_kwargs":{"enable_thinking":false}}`。ここに `max_completion_tokens` を指定すると、トークン予算を入れるフィールド名もこれに切り替わる |
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
**原則は「スコアは最初の1トークンから作る」**。モデルが回答ラベルの前に何か出力した時点で
シグナルを失うので、thinking はソース側で消す必要がある。`jev-bridge` はデフォルトで
**2段構え**で対処します:

1. **`chat_template_kwargs: {"enable_thinking": false, "preserve_thinking": false}`** を
   送る。バックエンド側の chat template が assistant ターンを空の `<think>\n\n</think>\n\n` で開くため、
   Qwen3 系テンプレート（SGLang / vLLM / 現行 llama.cpp）で実際に効くのはこちら。バックエンドが
   このフィールドを**拒否**した場合（HTTP 400/422）は自動で外して再試行し、以後その状態を
   記憶する。**黙って無視**された場合はリトライは発火せず、下記の症状そのままになる。
2. **assistant プレフィル `<think></think>`** をメッセージ末尾に付与。継続プレフィルを尊重するサーバー（llama.cpp など）でしか
   効かず、assistant 発話を単なる文脈として読むサーバーでは効かない（LiteLLM 経由で実測）。

`reasoning_effort` で制御するバックエンド（公式 OpenAI や多くのプロキシ）は
`JEVB_BACKEND_EXTRA_BODY` で送る:

```bash
export JEVB_BACKEND_EXTRA_BODY='{"reasoning_effort":"none"}'
```

実測の注意が2つ。effort の**レベル**は off スイッチではない（Qwen3 の reasoner で `low`
を送っても reasoning で始まった）。サーバーが受け付けない値は HTTP 400 が 502 として
返り、エラー本文に受理される値が書かれている。

**thinking が消えていないときの回答の見た目:** 全選択肢が同じ確率になり confidence が
落ちる —

```
choice → probabilities {"a": 0.3333, "b": 0.3333, "c": 0.3333}, confidence 0.0      noul → 0.5
```

信頼できるはずのモデルで一律分布を見たら、モデルのせいではなくこれを疑う。プロキシ経由なら
実線でも確認すること。`logprobs` が返ることと `chat_template_kwargs` が届いたことは別問題で、
LiteLLM は `drop_params: true` にすると `logprobs` を落として 200 を返す。

個別のオン/オフは `JEVB_DISABLE_THINKING=0` / `JEVB_PREFILL_ASSISTANT=0`。
テンプレート引数を直接渡したい場合は `JEVB_BACKEND_EXTRA_BODY`（組み込みの既定より優先）。

## 画像入力（拡張）

Jev 本体はテキスト専用のため、`images` は jev-bridge の拡張です。`state` の隣に
トップレベル `images` 配列を追加します。画像と state テキストは同じ user ターンに置き
（テンプレートは system メッセージ内の画像を拒否するため）、質問は末尾に残すので
プレフィックスキャッシュは画像の prefill を再利用します:

```json
{
  "state": "What is shown in this photo?",
  "images": ["data:image/png;base64,iVBOR…"],
  "questions": {
    "scene":      {"type": "choice", "instructions": "Where is this?", "criteria": {"indoor": "…", "outdoor": "…"}},
    "has_people": {"type": "noul", "instructions": "Are people visible?"}
  }
}
```

受け付ける画像参照:

| 形式 | 例 |
|---|---|
| data URI | `data:image/png;base64,iVBOR…` |
| URL | `https://example.com/photo.jpg`（バックエンドが取得） |
| 生 base64 | `iVBORw0KGgo…`（マジックバイトから形式を判定） |
| ローカルパス | `/path/to.png` — **`JEVB_ALLOW_LOCAL_IMAGES=1` のときのみ** |

注意点:

* 形式はペイロードから再判定されます。data URI の宣言 MIME が実体と食い違う場合は
  実体を優先し、実体が既知の画像形式でなければ 400 で拒否します。
* 対応形式: PNG / JPEG / GIF / WEBP / BMP。それ以外は呼び出し側で変換してください。
  サイズ上限は `JEVB_MAX_IMAGE_BYTES`（既定 20 MiB）。
* バックエンドモデルが vision 対応である必要があります。テキスト専用モデルでは
  バックエンドが拒否し、ブリッジは 502 とバックエンドのメッセージを返します。
* 画像トークンは共有プレフィックスの一部なので、1枚の画像に対する N 質問でも
  画像の prefill は1回です。

## 性能

**RTX PRO 6000 1枚** で **Qwen3.8-Flash-Next** を OpenAI 互換エンドポイント経由
（配得手前に LiteLLM。`chat_template_kwargs` が効くので thinking はオフ）、
ブリッジ経由・ウォーム状態で20回の中央値:

```
1問  (noul)                111 ms   (p10 105 / p90 138)
3問  (choice+noul+score)   201 ms   (p10 193 / p90 208)
6問                        366 ms   (p10 357 / p90 383)

3問 x 4リクエスト同時      138 ms/件   (4件で 550 ms)
3問 x 8リクエスト同時      135 ms/件   (8件で 1079 ms)
```

1リクエスト内の各質問は並列で発行します（`JEVB_MAX_CONCURRENCY`、既定8）。
上に残る差はブリッジではなくバックエンドのもので、同じエンドポイントを
直接叩くと単一の first-token 呼び出しは 107 ms、同時発行では 15〜21 件/s 付近で
頭打ちです（同時3 → 199 ms、12 → 563 ms）。ブリッジ自身のコストは 4 ms 程度。

再現: `python examples/bench.py http://127.0.0.1:8900 20`。
比較として、TypeSafe はホスト版 Jev で 70〜500 ms と公表しており、
コミュニティ計測では3問バッチで約212 ms という値もあります。

公式 OpenAI API を同じブリッジで叩いた値（`JEVB_DISABLE_THINKING=0`、
`JEVB_PREFILL_ASSISTANT=0`、ウォーム状態6回の中央値、ネットワーク込み）:

```
gpt-4.1-mini   1問 ~415 ms  3問 ~560 ms     gpt-5.6-luna   1問 ~608 ms  3問 ~629 ms
gpt-4.1-nano   1問 ~438 ms  3問 ~523 ms     gpt-5.4-mini   1問 ~662 ms  3問 ~729 ms
gpt-4o-mini    1問 ~583 ms  3問 ~570 ms     gpt-5.6-terra  1問 ~813 ms  3問 ~926 ms
gpt-4o         1問 ~597 ms  3問 ~617 ms     gpt-5.6-sol    1問 ~812 ms  3問 ~1077 ms
```

5.x 系は「使える OpenAI モデル」節の設定例が必要です。ホスト型 API で変わる点は
2つ。レイテンシは往復に支配される（この環境で ~0.4 s。上のローカル数値が目標）。
そして `temperature: 0` では曖昧でない入力が one-hot
（`{billing: 1.0, shipping: 0.0, returns: 0.0}`）で返ることが多く、閾値が動く
余地が残らない。本当に曖昧な入力では勾配は残る（`0.85 / 0.15 / 0.0`）ので、
「モデルが迷えない」のではなく動作点が端に寄る、という話。

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

リポジトリ直下の `docker-compose.yml` が既定の構成で、**公式 OpenAI API** を向きます。

```bash
export OPENAI_API_KEY=sk-...
docker compose up
curl -s http://127.0.0.1:8900/health
```

イメージは compose ファイルがリポジトリからビルドして `jev-bridge:latest` に
タグ付けする（`docker build -t jev-bridge .` と同じ）。公開イメージがあれば
`build:` と `image:` の2行を差し替えてください。バックエンドの切り替えは
`JEVB_BACKEND_BASE_URL` と `JEVB_BACKEND_MODEL`（トークン検証をするサーバーなら
`JEVB_BACKEND_API_KEY` を追加）だけで、compose ファイルを編集するか、同じ場所に
`docker-compose.override.yml` を置くか、下記の `docker run` を使えばよい。
以下の3例は同じブリッジ・同じ質問で、違いはバックエンド側に何を伝えておくか
だけです。

### 公式 OpenAI の場合

docker-compose.yml がすでにこの内容です。`chat_template_kwargs` は
vLLM/SGLang 拡張で、Chat API には assistant prefill がないので、thinking 抑止の
2層はどちらもオフにします。モデルは `logprobs` 対応が必要。`top_logprobs` は
**20** までで、既定の `JEVB_TOP_K` と同じ値（上げても増えては返りません）。
同じ内容の1コマンド版:

```bash
docker run --rm -p 8900:8900 \
  -e JEVB_BACKEND_BASE_URL=https://api.openai.com/v1 \
  -e JEVB_BACKEND_MODEL=gpt-4o-mini \
  -e JEVB_BACKEND_API_KEY="$OPENAI_API_KEY" \
  -e JEVB_DISABLE_THINKING=0 \
  -e JEVB_PREFILL_ASSISTANT=0 \
  jev-bridge
```

#### 使える OpenAI モデル

スコアリングは **最初のサンプルトークンの logprobs** があれば成り立つので、
モデルが `logprobs` を受けつけて、複数の候補を返してくれる必要がある。Chat
Completions API で実測した結果:

| モデル | 判定 |
|---|---|
| `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano` | そのまま使える。`top_logprobs` は 20 まで反映 |
| `gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-5.6-luna`, `gpt-5.6-terra`, `gpt-5.6-sol` | 下の設定例で使える（reasoning 系） |
| `o4-mini`, `gpt-5-nano` | 使用不可: `You are not allowed to request logprobs from this model` |

GPT-5.x の reasoning 系は3つの制約がある。いずれもブリッジの設定ではなく
モデル側のポリシー。

```bash
export JEVB_BACKEND_EXTRA_BODY='{"reasoning_effort":"none","max_completion_tokens":8}'
export JEVB_TOP_K=5
```

* reasoning がオンのあいだ `logprobs` は一切拒否される
  （`Unsupported parameter: 'logprobs' is not supported with this model`）。
  `reasoning_effort: "none"` で解除できるが、**レベル指定では無理**
  （`low` はそのまま拒否）。
* `max_tokens` が拒否され `max_completion_tokens` を求められる。上の例のように
  `JEVB_BACKEND_EXTRA_BODY` に書いておけば最初の一発目で通る（実行時まで
  分からないサーバには自動リトライで追従する）。予算 1 トークンは
  `Could not finish the message` で不可、4 以上なら可。
* `top_logprobs` は **5** まで。`JEVB_TOP_K=5`、つまり選択肢は5個まで。

**注意: 候補リストは切り詰められて返る。** OpenAI は確率質量のあるトークンしか
返さず、確信の高い回答は候補1件（`{"A": -0.0}`）で来るため、他のラベルは
ブリッジの下限値に落ちる。結果はどの方向に傾いても
`{"billing": 0.9993, "shipping": 0.0003, "returns": 0.0003}` という見た目になる
（同じ曖昧な入力で `gpt-4o-mini` なら
`{"billing": 0.148, "shipping": 0.0, "returns": 0.852}`）。1件しか返らないのは
「候補を1つしか返さない仕様」ではなく切り詰め（コイントスをさせたプロンプトでは
候補2件・`-0.34 / -1.25` と実差が返る）だが、結論は変わらない: 5.x 系では
**argmax（一番高い選択肢）は信じられる**が、`probabilities`・`confidence`・
閾値・`default_when` は計測値ではなく端数として出る。

**Responses API に乗り換えても得をしない。** 拒否はエンドポイントではなくモデル
側のポリシーで、そちらでも同じメッセージ
（`logprobs are not supported with reasoning models`）が返る。ブリッジは
`chat/completions` を使う。llama.cpp / vLLM / SGLang と同じ窓口であることも同じ
理由。

### llama.cpp（`llama-server`）の場合

ブリッジ側の設定は不要。現行ビルドなら `chat_template_kwargs` が読まれて
`enable_thinking` がチャットテンプレートに渡され、古いビルドで無視されても
assistant prefill（既定 on）が効きます。`top_logprobs` は llama.cpp の `n_probs`
（既定 20）に変換され、OpenAI のような上限はありませんが、1トークンあたりの
top-K を大きくすると転送量が増えるので、本当に必要なときだけ `JEVB_TOP_K` を
上げてください。質問は並列で飛ぶので、サーバー側のスロット（`--parallel`）は
`JEVB_MAX_CONCURRENCY` 以上を確保します。

```bash
llama-server -m Qwen3-8B-Instruct-Q4_K_M.gguf -c 16384 --port 8080 --parallel 8

docker run --rm -p 8900:8900 \
  -e JEVB_BACKEND_BASE_URL=http://host.docker.internal:8080/v1 \
  -e JEVB_BACKEND_MODEL=qwen3-8b-instruct \
  --add-host=host.docker.internal:host-gateway \
  jev-bridge
```

### vLLM の場合

`chat_template_kwargs` が通るので、thinking モデルは既定値のまま処理できます。
`top_logprobs` の上限はサーバーの `--max-logprobs`（既定 **20**、ブリッジの
`JEVB_TOP_K` と同じ）で、20個以上の選択肢が要るなら両方を上げます。プレフィックス
キャッシュは on のままにすると共有 prefix の prefill が1回になります。

```bash
vllm serve Qwen/Qwen3-8B --max-logprobs 64 --enable-prefix-caching

docker run --rm -p 8900:8900 \
  -e JEVB_BACKEND_BASE_URL=http://host.docker.internal:8000/v1 \
  -e JEVB_BACKEND_MODEL=Qwen/Qwen3-8B \
  -e JEVB_TOP_K=64 \
  --add-host=host.docker.internal:host-gateway \
  jev-bridge
```

SGLang（`python -m sglang.launch_server --model …`）も同じ形です。なお
[性能](#性能) の実測値は SGLang 構成でのものです。

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
