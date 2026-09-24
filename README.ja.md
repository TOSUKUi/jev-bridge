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

1. **state を最初の user ターンに**置きます（system は状態を一切含まない固定文）。
   質問を同じ user ターンの state の後に続けるので、1リクエスト内の全質問は
   質問文ブロックまで同一バイトのプレフィックスを共有します。プレフィックスキャッシュを
   持つバックエンド（SGLang の radix cache、vLLM の APC）では state の prefill を
   再利用できます。実測値は [プロンプトキャッシュ](#プロンプトキャッシュ) 節に。
2. 回答候補を**単一トークンのラベル**（`A`/`B`/`C…`、`true`/`false`、`0`/`1`/`2…`）として提示し、
   ラベルを1つだけ出力させる。
3. `max_tokens=1` + `logprobs` + `top_logprobs` で次トークン分布を取得し、
   **候補ラベル上の制限付き softmax** を計算する。確率は宣言した選択肢の上で和が1になる
   （Jev の回答と同じ契約）。
4. `choice` / `score` / `noul` と `confidence` を組み立てる。

ファインチューニング不要、較正学習不要、出力トークンのデコードも不要。

## クイックスタート

```bash
pip install git+https://github.com/TOSUKUi/jev-bridge.git   # PyPI にはありません
# (チェックアウトからなら: pip install -e ".[dev]")

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

## プロンプトキャッシュ

プロンプトはプレフィックスキャッシュが効く組み方にしています。1リクエストの
全質問は、質問文ブロックまで同じバイト列を送ります:

```
[system]    固定の指示文（状態は一切含まない）
[user]      [画像…] + "STATE:\n" + <state のシリアライズ> + "\n\n" + <質問ブロック>
[assistant] "<|im_start|></think>"   (JEVB_PREFILL_ASSISTANT)
```

state のシリアライズは1リクエストに1回だけ行い、ボディに id・タイムスタンプ・
seed はなく、httpx は固定のキー順で JSON 化するので、同じ入力は同じバイト列に
なります（`tests/test_api.py` が N 質問の共有プレフィックスを実際に検証）。ただし
`serialize_state` は dict の state を `indent=2` かつ**ワイヤー上のキー順**で
再シリアライズするので、同じ内容でもキー順を変えて送ったクライアントは別プレフィックス
になります（＝キャッシュミス）。

[性能](#性能) の Qwen3.8-Flash-Next エンドポイントで実測（出力1トークン）:

| プロンプト | cold | warm |
|---|---|---|
| ~110 トークン（典型的なテンプレート） | 106 ms | 106 ms |
| state ~1.4k トークン | 190 ms | 135 ms |
| state ~2.8k トークン | 293 ms | — |

キャッシュが効いていれば 2k トークンで +30 ms 程度とほぼ無料、cold の prefill は
1k トークンごとに +40 ms ほど。並列にする価値は温まった側にあります：**温まった**
2.7k-token state に4問を投げると **並列 215 ms / 直列 468 ms**、しかも新しい
コネクションプールで投げても 225 ms なので、キャッシュはコネクションに縛られません。
つまり state が温まっている限り fan-out のままで正解です。
気をつけるのは cold のほう：**同じ cold** の state に4問を投げると
**直列 665 ms / 並列 880 ms**でした（直列の内訳は 114 / 117 / 119 / 287 ms ＝
prefill 1回とヒット3回）。並列の 880 ms は、受付時点で誰もコミット済み prefix を
持てずに複数本 prefill した説明と整合します。ただしこれは wall-clock からの推定で、
サーバーは `--enable-cache-report` 無しで起動していたので `usage` に `cached_tokens` は
出ず、キャッシュを直接見た測定はありません。

レバーは `JEVB_MAX_CONCURRENCY=1` の1つだけ。全質問が直列になり、前の問が温めた
prefix を次の問が当たります。長い cold state 向きで、温まった側の処理量は落ちます
（上の 468 ms vs 215 ms）。

1問目を単独送信してから残りを流す primer は、一度実装して**削除しました**。
合成プロンプト（state 2.7k・質問1行）では勝っています（primer 込み 501 ms vs
無primer 841 ms）。しかしこのブリッジが実際に送るプロンプトでは全サイズで負けた：
4問で +79 ms、6問で +92 ms、捕捉ボディの再生でも +92 ms（699 → 791 ms）。
primer とは N本のうち1本を前倒しでやるだけなので節約たて (N−1)/N 個分の prefill、
そのために1往復まるごと先行して払う。ブリッジの質問ブロックは1行でなく数百トークン
あるので、この取引はマイナスになります。fan-out は無条件です。

自分で温めるなら安い撃ち方は `max_tokens: 0`。SGLang は prefill-only で走ります
（`completion_tokens: 0`、デコードはスケジューリングされない）、llama.cpp も
`n_predict: 0` と文書化されています。この呼び出し自体のコストは 4.1k-token の
prefix で cold 234〜448 ms / 温後 117 ms。得をするかは後に何が続くかで、
2.7k・質問1行のプロンプトでは続く4問 burst が 841 → 217 ms に下がりましたが、
2段構えの形（判定JSON 4.1k + 毎回変わる context ~500）では burst が 849 → 490 ms に
下がっても primer の 448 ms が効いて合計 938 ms と、温めないほうが速かったです。
温めるならリクエストの外、spec が切り替わった時かアイドル時に。GPU を使う以上タダではなく、
本番トラフィックと競合します。注意が2つ: `max_tokens` と書くこと（下の SGLang 節）、
そして `usage.prompt_tokens_details.cached_tokens` はサーバー起動時に
`--enable-cache-report` を付けていないと出ません（フィールドが無いのはミスではなく未計測）。

一致判定はメッセージ単位ではなく **レンダリング後のトークン列の 0 番地からの前方一致**
です。再利用できる長さは「既にキャッシュされている列との最長共通プレフィックス」で、
ブロック/ページ境界で切り捨てられ（vLLM は full block のみ、SGLang は `page_size` で挿入）、
そのブロックの prefill が走り終えて初めて挿入されます。このエンドポイントで
~1.9k-token の種を使って実測:

| プロンプト | ms | prompt tokens |
|---|---|---|
| 種、初回 | 233 | 1941 |
| 同じ prefix、末尾に 0 行追加 | 126 | 1941 |
| 同じ prefix、+910 tok 追加 | 152 | 2851 |
| 同じ prefix、+1860 tok 追加 | 216 | 3801 |
| 同じ prefix、質問文だけ変更 | 128〜131 | 1946 |
| state の先頭に nonce を 1 つ | 214〜219 | 1945 |

末尾に積み上がるのはほぼ無料（増えた末尾ぶんだけ払う）。分岐点が前にズレるほど
後ろ全体をやり直すわけで、上の nonce が全 prefill になったのは分岐点が user ターンの
先頭で、共有できる system ブロックが数十トークンしかなかったからです。もっと後ろで
分岐すれば、その上のぶんは残ります。nonce・タイムスタンプ・ターン数カウンタ・
JSON のキー順の変換が典型です。eviction は最も深い末尾から無くなる、と文書上は
なっています（SGLang は radix の leaf、vLLM は逆順で free）。つまりメモリが詰まった
ときに落ちやすいのは長いコンテキスト側で、安定している先頭側ではありません —
ただし保持時間はここで測っていません。

プロンプトの安定部分が state ではなく**判定JSON（質問spec）**側で、state の方が毎回
変わる（ゲームループなど）なら、`state` を1本の文字列にして spec を先に置いてください:
`state = "<判定JSON>\n\n<現在のコンテキスト>"`。`serialize_state` は文字列をそのまま
返すので、キャッシュされる prefix は system+spec になり、spec 側は全リクエストで
再利用されて末尾のコンテキストだけ再 prefill します。4.1k-token の spec で実測:
初回 519 ms、spec を使い回してコンテキストだけ変えて 173 ms、spec 単体に
`max_tokens: 0` の primer を撃つと4問並列の burst が 849 ms → 490 ms。

これは config のスイッチではなく **プロンプト設計の変更**として扱ってください。
別途採点のリグレッションテストが要ります。spec は信頼できない `STATE` の中に
入ります（区切りは改行だけで、指示とデータの優先を強制する機構はありません）、
state に写した spec と実際の questions は二重の正本になるので、ズレると
モデルは片方を読み、ブリッジはもう片方のラベル対応で返します。画像は state より
前にレンダリングされるので、画像を1枚でも付ける時点でこの前提は崩れます。
しかも長い spec が回答形式を崩しても HTTP は 200 で、確率はもっともらしく出ます
（top-K に現れないラベルは下限を入れて正規化するため）。速度ではなく
ラベル付き事例での正答率を前後で測ってください。

ブリッジが**送っていないもの**: `prompt_cache_key`、セッション id、カスタム
ヘッダー（ヘッダー設定は無く、固定値のボディフィールドなら `JEVB_BACKEND_EXTRA_BODY`
で注入可）。OpenAI なら通常は問題になりません — プロンプトキャッシュは
**1024 トークン以上**で自動的に働き（ヒットは `usage.prompt_tokens_details.cached_tokens`、
5〜10 分の非活動で eviction）、jev-bridge の typical テンプレートは 110〜200
トークンなので閾値を下回り、そもそもキャッシュされません。OpenAI が
`prompt_cache_key` を推す理由は「同じプレフィックスの要求を同じキャッシュに寄せる」
ことで、ゲートウェイ越しに1つの state の質問を複数レプリカへ散らす構成なら、
まさにそこを締めるためのキーです。

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

**RTX PRO 6000 1枚** で **Qwen3.8-Flash-Next** を **SGLang** で動かした環境に、
ブリッジ経由・ウォーム状態で10回ずつ叩いた値:

```
                        SGLang 直      LiteLLM 経由
1問  (noul)                81 ms           91 ms
3問                       158 ms          169 ms
6問                       320 ms          340 ms
```

右列は同じ SGLang サーバーを **LiteLLM** ゲートウェイ越しに叩いた値です
（両経路とも `chat_template_kwargs` が効くので thinking はオフ）。
ゲートウェイの分は1問で約10 ms、6問で約20 ms — 無視はできないけど、時間の行き先では
ありません。この表は以前 111 / 201 / 366 ms として載せたものを別の日の再計測に
置き換えたもので、数%差は日頃のノイズだと思って各自の機材で測り直してください。

### 再現性

`temperature: 0` でもこの環境は**ビット単位では再現しません**。同じボディを12回連続:

| 1リクエストあたりの質問数 | 最有力オプションの p（12回） | 全幅 |
|---|---|---|
| 1問 | 0.667 〜 0.738 | 0.071 |
| 6問 | 毎回 0.6985 | 0.000 |

greedy は**サンプリング**を固定するだけで、演算は固定しません。continuous batching で
その系列が入るバッチが変わり、reduction の順が変わるだけで logit が微動、拮抗した
質問ではそれが効きます。4桁目を測定値として扱わないでください。閾値には実質的な
余白を持たせてください。（この表の読み方は狭く: この共有負荷の下では6問リクエストは
12回一致し、1問リクエストは動いた、というだけ。「並列数を増やすと再現性が上がる」の
証拠ではなく、6問の行はプレフィックスを共有する独立6本の呼び出しであって、1本の
合議プロンプトではありません。）

### 検討済みで残っていなかったもの

はまる前に、測定で消えた候補を4つ:

* **1問あたりのコスト。** 連続発射で追加1問あたり約47 ms、3秒間隔を空けて発射すると
  約63 ms — 空けると**悪化**するのでバーストの待ち行列ではありません。内訳は未測定
  （クライアント側から見えるのは壁時計だけ）。
* **state の compact 化。** `indent=2` を `separators=(",", ":")` にすると
  2789 → 1707 tok/件（-39%）、cold は 965 → 839 ms（`/flush_cache` 付きのペア比較）。
  ただし warm では勝たない（片方の run で +3 ms、別の run で -58 ms）うえ、同じ6問で
  確率が最大 **0.27**、confidence が **0.41** 動きました（argmax の変化はゼロ）。
  空白もモデルの入力です。トークンを節約したいなら compact 化した文字列を state に
  渡せばいいけれど、その場合は採点のリグレッションを自分で回してください。
* **`logit_bias`。** direct でもゲートウェイ越しでも同じ返却値で効きます（この
  tokenizer では `/tokenize {"prompt": "B"}` → id 33、`true`/`false` は単一トークンの
  1802/3721）。ただし返ってくる logprob は「禁止トークンではない」という条件付きで、
  語彙全体を再正規化した値 — あなたのラベル集合で正規化した値ではありません。
  選択肢を1つ潰すことはできても、ラベル集合の実測確率は取れないので、floor の
  代わりにはなりません。
* **`top_logprobs` を 20 を超えて。** このバックエンドは 32/64/100 を受け付けます
  （OpenAI のような上限は無い）。ただ 3択でも10択でも全ラベルはすでに top-20 内に
  揃っていて floor は一度も発火せず、K を上げても確率の変化は 0.11 以下 — 上の
  運行間ゆらぎの範囲内でした。ラベルが本当に top-20 の外へ出る構成でなければ
  `JEVB_TOP_K` は触らないでください。

Fan-in（全質問を1つのプロンプトに入れ、位置ごとにラベルを読む）は測定で勝った
唯一の案で（6問 137 ms vs 同一運行の fan-out 337 ms）、それでも実装していません。
`P(label_i | state, question_i)` の代わりに `P(label_i | state, 全質問, それ以前の答え)` を
採点することになり、6問中2問で argmax が反転、確率は最大 0.55 動きました。
確率を製品にしている判定エンジンが、レイテンシのために確率の定義を置き換えるのは
筋が違う、という判断です。

1リクエスト内の各質問は並列で発行します（`JEVB_MAX_CONCURRENCY`、既定8）。
ただし「N問を同時に出す」と「N問を順番に出す」は別の量です。同じウォーム環境で
3問リクエストを4本同時に流すと:

```
                        wall      1本あたりの p50   最遅      wall/N
SGLang 直               498 ms        370 ms        491 ms     124 ms
LiteLLM 経由            526 ms        397 ms        520 ms     132 ms
```

ユーザーが体感するのは **1本あたりの p50（370〜400 ms）** のほうで、124〜132 ms の
ほうはスループットを時間の形に書き換えた数字です。（この README の前の版は
まさにその按分値を「1件あたり中央値」として載せていて、体感レイテンシを約3倍
過小評価していました。`examples/bench.py` は wall / p50 / 最遅 / wall÷N を
まとめて出します。）バックエンド呼び出し24本が約990 ms、つまり約24 件/s で、
1問あたり85 ms弱の予算のうちブリッジ自身の分は4 msのままです。ゲートウェイの
モデルグループに `rpm` や `max_parallel_requests` の上限を付けているなら、
その上限が天井になります（先の 24 件/s は天井になりません）— 容量計画の前に確認してください。

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
  プレフィックスキャッシュで共有 state の prefill はほぼ無料になります。ただし
  その state を叩く**最初**のリクエストだけは並列だと prefill を重複して払います
  （実測値は「プロンプトキャッシュ」節）。

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

SGLang（`python -m sglang.launch_server --model … --enable-cache-report`）も同じ形で、
[性能](#性能) の実測はこの構成（配得手前に LiteLLM）でのものです。SGLang について
知っておくべき点が2つ:

* `max_tokens: 0` は文字どおり prefill-only リクエストです（`is_prefill_only` 経路では
  デコードがスケジューリングされず、`completion_tokens: 0` が返る）。radix cache も温まります。
  一方 `max_completion_tokens: 0` は**同じ意味になりません**：内部で
  `max_completion_tokens or max_tokens` と読まれるので 0 が消え、予算が `1 << 30` に
  なります。実測では 292 トークン生成されました。ゼロを意図するなら `max_tokens` を送ってください
  （speculative decoding を有効にすると prefill-only の高速経路は無効になります）。
* `usage.prompt_tokens_details.cached_tokens` は `--enable-cache-report` を付けたときだけ
  返ります。無いとヒットしても `usage` には何も出ません。

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
