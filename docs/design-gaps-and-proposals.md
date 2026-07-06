# Decision Agent 設計ギャップと改善設計書

本書は、現行実装（`src/decision_agent/`）と仕様書（`decision-agent-spec.md`）・
運用ガイド（`operation-guide.md`）を突き合わせて特定した設計不足と、その改善設計を
まとめたものである。

対象コミット時点の実装: `review` / `learn` / `iterate` / `evaluate` CLI、
決定論的レビューエンジン、JSONL 履歴永続化。

優先度は以下の基準で付けている。

- **P0**: 仕様が約束している動作が現実の入力で成立しない、またはデータ破損につながる
- **P1**: プロジェクトの成功基準（「反復でユーザー判断に近づく」）の達成を妨げる
- **P2**: 拡張時に手戻りを生む未定義領域

---

## ギャップ一覧（サマリ）

| # | ギャップ | 優先度 | 影響範囲 |
|---|---------|--------|----------|
| 1 | テキストマッチング層が日本語で機能しない | P0 | review / evaluate / learn の全判断 |
| 2 | 決定履歴の Source of Truth が二重 | P0 | storage / profile / learn |
| 3 | プロファイル知識のライフサイクル未設計 | P1 | learn / profile |
| 4 | 評価の時系列トラッキングがない | P1 | evaluate / 成功基準の計測 |
| 5 | LLM レビュアーの接続点が未定義 | P1 | agent / 将来の review |
| 6 | 生成エージェントとの往復ループ未設計 | P2 | ワークフロー全体 |
| 7 | task_type の拡張方法が未定義 | P2 | models |
| 8 | 書き込みの原子性・スキーマバージョンがない | P2 | storage |

---

## 1. テキストマッチング層が日本語で機能しない（P0）

### 現状

判断の中核はすべて字句トークンの重なり率で動いている。

- `agent.py` の `_review_tokens` は `\w+` で切り出し、3 文字以下を捨てる
- `_text_similarity` は `len(left ∩ right) / len(left)` のトークン Jaccard 変種
- これに依存する箇所: 既知ミス照合（閾値 0.2）、否定パターン/選好ルール照合
  （0.34）、過去レコード類似（0.2）、評価の core issue / revision direction
  一致判定（`_text_matches_signal`、0.25）

日本語は空白で分かち書きされないため、`\w+` では文全体（または句読点区切りの長い
チャンク）が 1 トークンになる。結果として:

- 日本語のルール・フィードバック・成果物同士の類似度はほぼ常に 0 か、完全一致時のみ 1
- 仕様書の入出力例は日本語だが、その例をそのまま流すと照合が一切効かない
- `evaluate` の `core_issue_accuracy` / `revision_direction_accuracy` は
  日本語ケースでは実質「部分文字列が偶然含まれるか」だけで決まる

### 設計方針

類似度判定を **Matcher インターフェースとして抽象化**し、実装を差し替え可能にする。
これは後述の LLM 統合（ギャップ 5）の受け皿にもなる。

```python
class TextMatcher(Protocol):
    def similarity(self, left: str, right: str) -> float: ...
    def matches(self, pattern: str, text: str, threshold: float) -> bool: ...
```

実装は 3 段階で用意する。

1. **`CharNGramMatcher`（デフォルト、依存なし）**
   文字 bi-gram / tri-gram 集合の Jaccard 係数。言語非依存で日本語・英語の両方に
   最低限機能する。現行のトークン方式は英語専用フォールバックとして残してよい。
2. **`MorphologicalMatcher`（オプション依存）**
   `sudachipy` 等の形態素解析があれば単語ベースに切り替える。extras
   （`pip install decision-agent[ja]`）で導入。
3. **`LLMMatcher`（将来）**
   意味的一致を LLM に問い合わせる。evaluate の再現性のため、判定結果は
   ケース ID + テキストハッシュでローカルキャッシュする。

### 設計上の決定事項

- 閾値（0.2 / 0.25 / 0.34）はマジックナンバーとして散在している。Matcher と
  用途ごとの閾値を `MatchingConfig` に集約し、プロファイルまたは CLI オプションで
  上書き可能にする。閾値は matcher 実装ごとに意味が変わるため、**閾値は matcher に
  属する**（呼び出し側が生の数値を持たない）。
- `_text_similarity` は非対称（分母が left のみ）。意図的な設計（「パターンの語が
  どれだけ本文に現れるか」）なので、インターフェース上は
  `containment(pattern, text)` と `similarity(a, b)` を別メソッドとして明示する。

### 受け入れ基準

- 仕様書の日本語サンプル（`docs/decision-agent-spec.md` の評価ケース例）を
  そのまま `evaluate` に通し、core issue の一致・不一致が意図どおり判定される
- 既存の英語テストが matcher 差し替え後も通る

---

## 2. 決定履歴の Source of Truth が二重（P0）

### 現状

`DecisionRecord` が 2 箇所に保存される。

- `DecisionProfile.decision_records`（プロファイル JSON 内）
- `--records` で指定する JSONL ファイル

`learn` はレコードをプロファイルに**も**追加して `save_profile` で書き出すため、
反復するたびにプロファイル JSON が全履歴を抱えて肥大化する。一方 `review` は
`--records` 指定時にプロファイル内レコードを**無視**する（`history_records` が
非 None なら差し替え）ため、どちらが正かが実行パスで変わる。

仕様書は「プロファイルは現在の判断サマリ、JSONL が生の証拠」と明言しており、
実装がそれに反している。

### 設計方針

**JSONL を唯一の履歴 Source of Truth とし、プロファイルから履歴を外す。**

- `DecisionProfile.decision_records` を廃止（読み込み時は後方互換で無視・警告）
- `learn` は「更新済みプロファイル」と「新規 `DecisionRecord`」を**別々に返す**
  ```python
  def learn(...) -> tuple[DecisionProfile, DecisionRecord]: ...
  ```
  現行の「プロファイル末尾のレコードを取り出して JSONL に書く」という CLI 側の
  暗黙の結合（`learned.decision_records[-1]`）を解消する
- `review` / `evaluate` の履歴は常に明示引数 `history_records` で渡す。
  プロファイル内フォールバックを廃止し、「履歴なし = 空」と単純化する
- レコード数が増えたときの読み込み上限（例: 直近 N 件 + task_type 一致優先）を
  `load_decision_records(path, limit=...)` として storage 層に持たせる

### 移行

- 旧形式プロファイル（`decision_records` 入り）を読んだ場合:
  1 回だけ「`decision-agent migrate <profile> --records <jsonl>`」で JSONL へ
  吐き出すコマンドを提供し、以後プロファイルには書かない。

---

## 3. プロファイル知識のライフサイクル未設計（P1）

### 現状

- `preference_rules` / `negative_patterns` / `positive_examples` は
  `_append_unique`（完全一致での重複排除のみ）で**増える一方**
- 言い回しが少しでも違うルールは別物として蓄積し、レビュー時の照合ノイズになる
- `KnownMistake` の統合キーは `user_feedback.notes` の完全一致。自由記述の notes が
  一致することは稀で、`count` がほぼ 1 のまま増殖する
- 仕様・運用ガイドは「suggested_profile_updates はユーザーが承認したものだけ
  プロファイルに入れる」と定めているが、承認を表現する状態やコマンドがない
- ルールに出所（どのフィードバック・評価から来たか）と適用範囲（task_type）がない

### 設計方針

ルールを素の `str` から構造体に昇格させる。

```json
{
  "text": "start with a concrete failure case before naming the concept",
  "task_types": ["blog_outline"],
  "status": "active",          // proposed | active | retired
  "source": "feedback:20260706T...-blog_outline-...",
  "created_at": "2026-07-06T...",
  "hits": 4,                    // レビューで照合に使われた回数
  "confirmations": 2            // ユーザー判断と一致した回数
}
```

- **status ライフサイクル**: `evaluate` の `suggested_profile_updates` は
  `proposed` として保存し、`decision-agent profile approve <rule-id>` /
  `reject <rule-id>` で `active` / 削除に遷移。review で使うのは `active` のみ。
  これで運用ガイドの「承認したルールだけ反映」がツールとして成立する
- **KnownMistake の統合キー**を notes 完全一致から
  「(agent_verdict, user_verdict) ペア + notes の matcher 類似（閾値付き）」に変更。
  類似ミスの `count` が実際に積み上がるようにする
- **減衰**: `hits` があるのに `confirmations` が伸びないルール
  （照合するがユーザー判断に寄与しない）を evaluate レポートで
  `retire_candidates` として提示する。自動削除はしない（仕様の
  「自動で真実扱いしない」原則に従う）
- 後方互換: 素の文字列ルールは読み込み時に
  `{"text": ..., "status": "active", "task_types": []}` へ正規化

---

## 4. 評価の時系列トラッキングがない（P1）

### 現状

成功基準は「反復によりユーザー判断へ近づくこと」だが、`evaluate` は stdout に
レポートを出すだけで、**前回との比較・推移の記録が存在しない**。改善しているかを
知る手段が運用者の記憶しかない。

### 設計方針

- `evaluate --history evals/blog_outline_evals.jsonl` で評価実行を追記保存する
  ```json
  {
    "run_at": "2026-07-06T...",
    "profile_path": "profiles/default.json",
    "profile_fingerprint": "sha256:...",   // プロファイル内容のハッシュ
    "cases_path": "cases/blog_outline_cases.jsonl",
    "cases_fingerprint": "sha256:...",
    "cases": 12,
    "verdict_accuracy": 0.75,
    "core_issue_accuracy": 0.58,
    "revision_direction_accuracy": 0.5
  }
  ```
- fingerprint を持つ理由: ケースセットが変わった run 同士の精度比較は無意味なので、
  同一 `cases_fingerprint` の run だけを系列として比較する
- レポートに前回同条件 run との差分（`delta_vs_previous`）を含める
- ケース側にも安定 ID を必須化する（現状 `id` 省略時は `case-{index}` で行順に
  依存し、ケース追加で ID がずれる）。`id` 欠落は evaluate 時に警告、
  `--strict` でエラー

---

## 5. LLM レビュアーの接続点が未定義（P1）

### 現状

仕様は「LLM がレビューを行っても、データモデルとワークフローは明示的なまま」と
方向付けているが、`DecisionAgent.review` は決定論ロジックがハードコードされ、
差し替え点がない。

### 設計方針

レビューエンジンを Strategy として分離する。

```python
class Reviewer(Protocol):
    def review(
        self,
        request: ArtifactReviewRequest,
        profile: DecisionProfile,
        history: tuple[DecisionRecord, ...],
    ) -> ArtifactReview: ...

class RuleBasedReviewer:   # 現行ロジックを移設
class LLMReviewer:         # 将来: プロファイルをプロンプトに展開して照会
```

- `DecisionAgent` は Reviewer を注入されるオーケストレータになり、
  learn / evaluate は Reviewer 実装に依存しない
- LLMReviewer の設計上の要点（実装は将来でよいが、契約はいま決める）:
  - **入出力契約は既存の `ArtifactReview` スキーマそのまま**。LLM 出力は
    `ArtifactReview.from_dict` でバリデーションし、不正 verdict は失敗扱い
  - プロンプトへ渡すのは profile の `active` ルール・既知ミス・関連履歴
    （ギャップ 2 の `_relevant_records` 上位 N 件）に限定し、注入内容を
    `learned_signals` に記録して説明可能性を保つ
  - `evaluate` で使う場合は再現性のため temperature=0 とレスポンスキャッシュ
    （request + profile fingerprint がキー）を必須とする
- CLI は `--reviewer rule|llm` で選択。デフォルトは `rule`（API キー不要の
  現行体験を維持）

### 付随して直すこと

- `confidence` の意味論が未定義（現行式は issue が多いほど確信度が上がる、
  という直感に反する挙動を含む）。「confidence = この verdict がユーザー判断と
  一致する主観確率」と定義し、根拠（照合したルール数・類似履歴の有無）から
  組み立てる式に改める。将来は evaluate 履歴（ギャップ 4）でキャリブレーション
  可能になる

---

## 6. 生成エージェントとの往復ループ未設計（P2）

### 現状

仕様のターゲットワークフローは「生成 → レビュー → revision_instruction を
生成側に戻す → 再生成」だが、`revision_instruction` を機械可読に受け渡す
インターフェースがない（人間がコピペする前提）。

### 設計方針（MVP+1 スコープ）

- `review` の出力 JSON をそのまま次の生成プロンプトに埋め込める
  「revision request」形式を定義する
  ```json
  {
    "task_type": "blog_outline",
    "intent": "...",
    "previous_artifact": "...",
    "revision_instruction": "...",
    "issues": [...],
    "iteration": 2,
    "parent_record_id": "20260706T...-blog_outline-..."
  }
  ```
- `DecisionRecord` に `parent_record_id` を追加し、同一成果物の反復系列を
  追跡可能にする（「何回の修正で accept に到達したか」が将来の成功指標になる）
- 生成そのものは仕様どおりスコープ外。本設計は**受け渡しデータ形式の定義のみ**

---

## 7. task_type の拡張方法が未定義（P2）

### 現状

`SUPPORTED_TASK_TYPES` がコードにハードコードされ、新しい成果物種別を足すには
コード変更が要る。仕様は「コアループを変えずに他の主観的成果物へ拡張できること」を
求めている。

### 設計方針

- task_type の許可リストを**プロファイルに移す**: `profile.task_types`
  （省略時は現行 3 種）。バリデーションは「プロファイルに宣言された task_type
  のみ受理」に変更
- task_type は自由文字列にしない（typo で履歴が分裂するため）。未知の task_type は
  エラーにし、メッセージで「プロファイルの task_types への追加」を案内する

---

## 8. 書き込みの原子性・スキーマバージョンがない（P2）

### 現状

- `save_profile` は直接上書き。書き込み途中の中断でプロファイル破損の可能性
- `iterate` は「JSONL 追記 → プロファイル保存」の 2 段書き込みで、間で失敗すると
  履歴とプロファイルが不整合になる
- プロファイル・レコードに `schema_version` がなく、本書の変更
  （ギャップ 2, 3）のような形式変更時に判別手段がない

### 設計方針

- `_save_json` を temp file + `os.replace` による原子的書き込みに変更
- `iterate` の書き込み順序を「JSONL 追記が成功してからプロファイル保存」と定め、
  プロファイル保存失敗時のメッセージに「履歴は追記済み。プロファイルのみ再実行
  可能」であることを明示する（JSONL は追記専用なので二重追記は record id で
  検出できる）
- プロファイル・レコード・評価履歴の各 JSON に `schema_version: 1` を導入。
  読み込み時に未知バージョンはエラー、欠落は v0（現行）として互換読み込み

---

## 実施順序の提案

1. **ギャップ 2（履歴の一本化）** — 他の全設計の前提になるデータモデル整理。
   移行コマンド込みで先に済ませる
2. **ギャップ 1（Matcher 抽象化 + 文字 n-gram 実装）** — 日本語運用を成立させる。
   ギャップ 3 の類似ベース統合もこの Matcher を使う
3. **ギャップ 4（評価履歴）** — 以降の改善が数値で追えるようになる
4. **ギャップ 3（ルールライフサイクル）** — 承認フローと構造化ルール
5. **ギャップ 5（Reviewer 分離）** — リファクタのみ先行し、LLMReviewer は
   契約だけ固定して後続
6. ギャップ 6〜8 は上記に相乗りして順次

各段階は独立してテスト可能であり、既存 CLI の互換性（コマンド名・主要引数）は
維持する。
