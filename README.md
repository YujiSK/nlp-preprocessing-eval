# NLP Preprocessing Evaluation

[English](#english) | [日本語](#日本語)

A reproducible evaluation framework for quantifying how preprocessing decisions
change model performance, explainability, prediction coverage, and computational
cost. The repository contains executable experiments, machine-readable results,
an extended appendix, and a guarded Markdown-to-PDF reporting pipeline.

## English

### What this project evaluates

The project compares four classifiers under controlled before/after preprocessing
conditions:

- Logistic Regression
- Linear SVC
- Random Forest
- k-Nearest Neighbors

The outer evaluation split is shared across models and conditions. Unless an
experiment explicitly documents otherwise, evaluation uses five-fold
`StratifiedKFold` with `shuffle=True` and `random_state=42`. Learned preprocessing
is fitted inside each training fold to prevent leakage.

| Experiment | Data characteristic | Before | After | Primary perspective |
|---|---|---|---|---|
| A | Numeric features with different scales | No scaling | `StandardScaler` | Accuracy and scale sensitivity |
| B | Mixed numeric/categorical data with missing values | Drop incomplete training rows | Median/mode imputation | Accuracy and retained training data |
| C | Strong class imbalance | No imbalance correction | Fold-local SMOTE | Average Precision and minority-class performance |
| D | Japanese text classification | Raw text + MeCab/IPAdic | `neologdn` + Sudachi Mode C | macro-F1, vocabulary size, and processing cost |

### Extended experiments

The extended evaluation separates practical questions that are not fully answered
by a basic before/after comparison:

| Extra experiment | Question | Main output |
|---|---|---|
| A — Permutation importance | Does preprocessing change validation-set feature importance? | Fold-level and aggregated importance |
| B — Prediction coverage | What happens when production inference rows contain missing values? | Coverage, conditional accuracy, and correct fraction over all requests |
| C — Nested threshold selection | Does an inner-CV-selected probability threshold improve outer-fold performance? | Selected thresholds and leakage-free outer-fold metrics |
| D — 2×2 NLP ablation | Can cleaning and tokenizer effects be separated? | D0–D3 performance, vocabulary, and preprocessing cost |

The D ablation conditions are:

- D0: raw text × MeCab/IPAdic
- D1: `neologdn` cleaning × MeCab/IPAdic
- D2: raw text × Sudachi core/Mode C
- D3: `neologdn` cleaning × Sudachi core/Mode C

### Evaluation design

The implementation follows these rules:

1. Before and after conditions reuse the same outer folds.
2. Scalers, imputers, encoders, TF-IDF, sampling, calibration, and threshold
   selection are fitted only from training data.
3. Fold-level observations are retained instead of reporting only one aggregate.
4. Paired fold differences are reported without treating CV folds as independent
   samples for a conventional significance test.
5. Model interpretation is qualified: coefficients depend on scale and correlated
   features can dilute permutation importance.
6. Timing and resource measurements are reported alongside predictive metrics.
7. Environment metadata is saved with core experiment outputs.

See [`docs/execution_plan.md`](docs/execution_plan.md) for the complete experimental
protocol, leakage controls, metric definitions, and interpretation constraints.

### Repository structure

```text
.
├── README.md
├── assets/
│   └── styles/report.css
├── configs/
│   └── layout_overrides.yml
├── data_cache/                    # Local datasets/cache; ignored by Git
├── docs/
│   ├── execution_plan.md
│   └── daily_report_*.md
├── outputs/
│   ├── SUMMARY_REPORT.md
│   ├── SUMMARY_REPORT.pdf
│   ├── SUMMARY_REPORT_extra.md
│   ├── SUMMARY_REPORT_extra.pdf
│   ├── exp_a/ ... exp_d/          # Core experiment artifacts
│   ├── exp_a_extra/ ...           # Extended experiment artifacts
│   ├── discussion_draft/
│   ├── figures/
│   ├── reports/                   # Layout and pipeline audit logs
│   └── renders/                   # Ignored intermediate render artifacts
├── scripts/
│   ├── core/                      # Core experiment CLI entry points
│   ├── extra/                     # Extended experiment and aggregation CLIs
│   └── report/                    # Report/PDF CLIs
├── src/
│   ├── experiments/               # Reusable experiment logic
│   └── reporting/                 # Reusable report/PDF logic
└── tests/
```

The placement rule is simple: reusable functions and classes belong in `src/`;
command-line entry points belong in `scripts/`. Experiment-specific report logic
belongs in `scripts/extra/`, while experiment-independent PDF/report logic belongs
in `scripts/report/`.

### Requirements

The project has been exercised with Python 3.14 in a Linux environment. A
dependency lock file is not currently included, so create an isolated environment
and install the runtime packages explicitly:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  numpy pandas scipy \
  scikit-learn imbalanced-learn \
  matplotlib seaborn \
  mecab-python3 ipadic neologdn \
  SudachiPy sudachidict_core \
  markdown beautifulsoup4 PyYAML pdfplumber \
  pytest
```

PDF generation additionally requires:

- `google-chrome` available on `PATH`
- Poppler utilities (`pdftoppm`) for optional PDF page rendering
- A local loopback socket, used briefly to serve report assets to headless Chrome

For Debian/Ubuntu, Poppler can normally be installed with:

```bash
sudo apt-get install poppler-utils
```

Chrome installation varies by platform. The renderer currently resolves the
binary name as `google-chrome`.

### Data preparation

Most core datasets are generated or loaded by scikit-learn. Two paths require
additional attention:

- Extra experiment B loads Titanic from OpenML and caches it under
  `data_cache/openml/`. The first run requires network access; later runs can reuse
  the cache.
- Core/extra experiment D expects the extracted livedoor News Corpus at
  `data_cache/text/`, with one directory per category.

`data_cache/` is intentionally ignored by Git. Dataset archives and extracted
corpora must not be committed; only `data_cache/.gitkeep` is tracked.

### Quick start

Run commands from the repository root. The scripts add the project root to
`sys.path`, but `PYTHONPATH=.` is shown explicitly for predictable invocation:

```bash
source venv/bin/activate

# Run all core experiments
PYTHONPATH=. python3 scripts/core/run_exp_a.py
PYTHONPATH=. python3 scripts/core/run_exp_b.py
PYTHONPATH=. python3 scripts/core/run_exp_c.py
PYTHONPATH=. python3 scripts/core/run_exp_d.py

# Run all extended experiments
PYTHONPATH=. python3 scripts/extra/run_exp_a_permutation.py
PYTHONPATH=. python3 scripts/extra/run_exp_b_coverage.py
PYTHONPATH=. python3 scripts/extra/run_exp_c_threshold.py
PYTHONPATH=. python3 scripts/extra/run_exp_d_ablation.py

# Regenerate per-model extended figures and integrate the appendix
PYTHONPATH=. python3 scripts/extra/generate_extra_figures.py
PYTHONPATH=. python3 scripts/extra/build_summary.py

# Build and validate both reports
PYTHONPATH=. python3 scripts/report/build_report.py -t both
```

Experiment D is the heaviest path because it tokenizes the Japanese corpus and
evaluates four models over multiple folds. Rebuilding only the PDFs does not rerun
those experiments.

### Report build modes

The report CLI accepts `main`, `extra`, or `both`:

```bash
# Main report only
PYTHONPATH=. python3 scripts/report/build_report.py -t main

# Extended report only
PYTHONPATH=. python3 scripts/report/build_report.py -t extra

# Both reports (default target)
PYTHONPATH=. python3 scripts/report/build_report.py -t both
```

The default mode is intentionally non-destructive. It builds and validates once
without writing `configs/layout_overrides.yml`.

```bash
# Explicitly equivalent to the safe default
PYTHONPATH=. python3 scripts/report/build_report.py -t extra --no-repair

# Build HTML/PDF only; skip validation and repair
PYTHONPATH=. python3 scripts/report/build_report.py -t extra --pdf-only

# Opt in to automatic layout repair
PYTHONPATH=. python3 scripts/report/build_report.py -t extra --auto-repair
```

`--auto-repair`, `--no-repair`, and `--pdf-only` are mutually exclusive.

### Safe layout overrides

`configs/layout_overrides.yml` separates human decisions from generated rules and
separates the main and extended reports:

```yaml
manual:
  main:
    page_break_before: []
    keep_together: []
  extra:
    page_break_before: []
    keep_together: []
generated:
  main:
    page_break_before: []
    keep_together: []
  extra:
    page_break_before: []
    keep_together: []
```

Important guarantees:

- Normal builds never save the YAML file.
- Automatic repair never changes either `manual` section.
- Only the selected target's `generated` section is updated.
- Generated settings are written atomically using a temporary file, `fsync`, and
  `os.replace`.
- The builder receives the merged overrides directly; it does not independently
  reload YAML during the repair loop.
- Unknown `data-source-id` values fail clearly instead of being silently ignored.

### Layout validation and stable source IDs

Markdown headings, figures, tables, and code blocks receive stable
`data-source-id` attributes before PDF rendering. Numbered headings keep readable
IDs; implicit headings use their parent hierarchy, a semantic slug, and a text
hash. Adding an unrelated sibling therefore does not renumber existing fallback
IDs.

The checker detects:

- orphan headings at a page boundary
- figure/caption splits
- unexpected splits of short blocks
- text outside the printable area
- source elements that could not be matched to extracted PDF text

Short Japanese headings such as `方法` and `結果` are resolved with exact word
matching and a page/Y-position cursor. A build is:

- `PASS` when there are no violations or unresolved elements
- `FAIL` when one or more violations exist
- `INDETERMINATE` when no violation is confirmed but unresolved elements remain

Only `PASS` returns a successful validation exit status.

To re-check already generated PDFs:

```bash
PYTHONPATH=. python3 scripts/report/check_pdf_layout.py
```

To render the generated PDFs into page images for manual inspection:

```bash
PYTHONPATH=. python3 scripts/report/render_pdf_pages.py
```

Page images and intermediate HTML live under `outputs/renders/` and are ignored by
Git. The normal report pipeline removes obsolete duplicate HTML and page images;
canonical build HTML and source registries are retained under
`outputs/renders/_build/`.

### Outputs

The repository contains two deliverable reports:

- `outputs/SUMMARY_REPORT.md` / `.pdf`: core experiments only
- `outputs/SUMMARY_REPORT_extra.md` / `.pdf`: core report plus extended Appendix C

Each `outputs/exp_*` directory contains machine-readable CSV data and figures.
Core experiments additionally save environment metadata. Each
`outputs/exp_*_extra` directory contains its standalone
`APPENDIX_EXP_*.md`, aggregate/fold CSVs, the original comparison figure, and
per-model or per-condition figures used by the integrated report.

Audit files under `outputs/reports/` include JSON layout results and design notes.
Development records remain under `docs/`, keeping operational logs separate from
deliverable outputs.

### Testing and validation

Run the full test suite:

```bash
PYTHONPATH=. pytest tests/
```

Run import/bytecode and whitespace checks:

```bash
PYTHONPATH=. python3 -m compileall src scripts
git diff --check
```

The PDF end-to-end test starts a loopback HTTP server and launches headless Chrome.
It can fail in a restricted sandbox that prohibits local socket creation even when
the code is correct; run it in a normal local shell in that case.

### Reproducibility notes

- Shared random seed: `42`
- Default outer CV: five stratified shuffled folds
- Before/after comparisons reuse the same fold indices
- Learned preprocessing remains inside the training fold
- Core runs record Python/platform/package information
- Raw/cached datasets are excluded from version control
- Existing CSV artifacts allow extended figures and reports to be rebuilt without
  rerunning every expensive experiment

### Known limitations

- There is currently no pinned `requirements.txt` or lock file. Record package
  versions for any formal reproduction.
- OpenML data may require network access on the first run.
- The livedoor corpus must be obtained separately and used in accordance with its
  license.
- Timing results depend on hardware, OS load, and tokenizer dictionary versions.
- Logistic Regression can reach `max_iter` in the intentionally unscaled condition;
  convergence warnings are retained as an experimental observation.
- Automatic layout repair handles ID-addressable page-break/keep-together fixes. It
  does not invent arbitrary CSS solutions for every possible overflow.

### Contributing

Keep changes reproducible and scoped:

1. Put reusable logic in `src/experiments/` or `src/reporting/`.
2. Keep CLI wrappers thin and place them in the appropriate `scripts/` subdirectory.
3. Do not commit `data_cache/`, virtual environments, caches, or render intermediates.
4. Preserve fold-local fitting and shared splits in before/after comparisons.
5. Add or update tests for changes to matching, IDs, overrides, or report layout.
6. Run tests, `compileall`, and `git diff --check` before publishing.

---

## 日本語

### 概要

本リポジトリは、前処理の選択が機械学習モデルの性能・説明性・予測可能率・
処理コストへ与える影響を定量評価するための再現可能なフレームワークです。
実行可能な実験コード、CSV・比較図、発展実験の付録原稿、および
Markdownから検査済みPDFを生成する安全なレポートパイプラインを収録しています。

全実験で次の4モデルを比較します。

- ロジスティック回帰
- Linear SVC
- ランダムフォレスト
- k-近傍法

基本的な外側評価は、モデル・条件間で共通の
`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`を使用します。
標準化、補完、エンコーディング、TF-IDF、サンプリング、確率校正、
閾値選択などの学習型処理は、必ず各Foldの学習データだけでfitします。

### 実験構成

| 実験 | データ特性 | Before | After | 主な評価観点 |
|---|---|---|---|---|
| A | スケールの異なる数値特徴量 | 未標準化 | `StandardScaler` | Accuracyとスケール依存性 |
| B | 数値・カテゴリ混在の欠損データ | 学習時の欠損行削除 | 中央値・最頻値補完 | Accuracyと学習データ保持率 |
| C | 強いクラス不均衡 | 補正なし | Fold内SMOTE | Average Precisionと少数クラス性能 |
| D | 日本語テキスト分類 | 生テキスト＋MeCab/IPAdic | `neologdn`＋Sudachi Mode C | macro-F1、語彙数、処理コスト |

発展実験では、次の実務的な問いを検証します。

- 実験A: 検証Fold上のPermutation ImportanceによるBefore/After比較
- 実験B: 推論時欠損に対するCoverage、予測可能行の精度、全要求基準の正答率
- 実験C: 内側CVで選んだ確率閾値を外側Foldだけで評価するNested CV
- 実験D: クレンジング有無×形態素解析器のD0〜D3アブレーション

実験Dの4条件は、D0＝生テキスト×MeCab、D1＝クレンジング×MeCab、
D2＝生テキスト×Sudachi、D3＝クレンジング×Sudachiです。これにより、
精度・語彙数・処理時間への寄与を要因別に確認します。

詳細な実験計画、リーク防止規則、指標定義、解釈上の注意は
[`docs/execution_plan.md`](docs/execution_plan.md)を参照してください。

### セットアップ

Python 3.14のLinux環境で動作確認しています。現在は依存関係の固定ファイルを
含んでいないため、仮想環境を作成して必要パッケージを導入してください。

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  numpy pandas scipy \
  scikit-learn imbalanced-learn \
  matplotlib seaborn \
  mecab-python3 ipadic neologdn \
  SudachiPy sudachidict_core \
  markdown beautifulsoup4 PyYAML pdfplumber \
  pytest
```

PDF生成には`google-chrome`、ページ画像化にはPopplerの`pdftoppm`も必要です。
PDFレンダラーは`google-chrome`というコマンド名を使用します。

```bash
sudo apt-get install poppler-utils
```

### データの準備

- 基本実験A〜Cの主要データはscikit-learnによる読込・生成を使用します。
- 発展実験BはOpenMLのTitanicを`data_cache/openml/`へキャッシュします。
  初回だけネットワーク接続が必要です。
- 実験Dは展開済みlivedoor News Corpusを`data_cache/text/`へ配置します。
  カテゴリごとのディレクトリ構造が必要です。

`data_cache/`はGit管理対象外です。データセット本体やアーカイブをコミットせず、
各データセットのライセンスに従って利用してください。

### 実行手順

リポジトリのルートから実行します。

```bash
source venv/bin/activate

# 基本実験
PYTHONPATH=. python3 scripts/core/run_exp_a.py
PYTHONPATH=. python3 scripts/core/run_exp_b.py
PYTHONPATH=. python3 scripts/core/run_exp_c.py
PYTHONPATH=. python3 scripts/core/run_exp_d.py

# 発展実験
PYTHONPATH=. python3 scripts/extra/run_exp_a_permutation.py
PYTHONPATH=. python3 scripts/extra/run_exp_b_coverage.py
PYTHONPATH=. python3 scripts/extra/run_exp_c_threshold.py
PYTHONPATH=. python3 scripts/extra/run_exp_d_ablation.py

# 保存済みCSVから個別図を再生成し、発展版Markdownを統合
PYTHONPATH=. python3 scripts/extra/generate_extra_figures.py
PYTHONPATH=. python3 scripts/extra/build_summary.py

# 本編・発展版PDFを生成して検査
PYTHONPATH=. python3 scripts/report/build_report.py -t both
```

日本語コーパスを解析する実験Dは特に処理時間がかかります。PDFだけを作り直す場合、
実験を再実行する必要はありません。

### PDFビルド

対象は`main`、`extra`、`both`から選択できます。

```bash
# 本編のみ
PYTHONPATH=. python3 scripts/report/build_report.py -t main

# 発展版のみ
PYTHONPATH=. python3 scripts/report/build_report.py -t extra

# 両方
PYTHONPATH=. python3 scripts/report/build_report.py -t both
```

通常ビルドは非破壊です。PDFを1回生成・検査しますが、
`configs/layout_overrides.yml`を書き換えません。

```bash
# 安全な通常ビルドを明示
PYTHONPATH=. python3 scripts/report/build_report.py -t extra --no-repair

# 検査と修復を省略し、既存成果物からHTML/PDFだけを生成
PYTHONPATH=. python3 scripts/report/build_report.py -t extra --pdf-only

# 自動修復を明示的に有効化
PYTHONPATH=. python3 scripts/report/build_report.py -t extra --auto-repair
```

`--auto-repair`、`--no-repair`、`--pdf-only`は同時指定できません。

### レイアウト設定の安全性

`configs/layout_overrides.yml`は、本編・発展版ごとに手動設定と自動生成設定を
分離しています。

- `manual.main` / `manual.extra`: 人が管理する設定
- `generated.main` / `generated.extra`: 自動修復が管理する設定

通常ビルドはYAMLを保存しません。自動修復時も`manual`は変更せず、選択した
レポートの`generated`だけを一時ファイル、`fsync`、`os.replace`によって
原子的に保存します。存在しない`data-source-id`を指定した場合は、設定漏れを
黙って無視せず明示的なエラーにします。

### 自動レイアウト検査

MarkdownからHTMLへ変換する際、見出し・図・表・コードブロックへ安定した
`data-source-id`を付与します。暗黙見出しのIDは、親階層、意味的slug、
見出し本文のハッシュから生成するため、無関係な兄弟要素の追加で変化しません。

検査対象は次のとおりです。

- ページ末尾の孤立見出し
- 図とキャプションの分断
- 短い表・コードブロック等の不自然な分断
- 印刷可能領域からの文字はみ出し
- PDF抽出テキストと原稿要素を照合できない未解決項目

短い日本語見出しは完全一致とページ内Y座標カーソルを使用して解決します。
判定結果は、問題なしの`PASS`、違反ありの`FAIL`、未解決ありの
`INDETERMINATE`です。未解決項目が残る状態を修復成功として扱いません。

生成済みPDFだけを再検査する場合:

```bash
PYTHONPATH=. python3 scripts/report/check_pdf_layout.py
```

目視確認用のページ画像を生成する場合:

```bash
PYTHONPATH=. python3 scripts/report/render_pdf_pages.py
```

一時HTML・ページPNGは`outputs/renders/`へ集約され、Git管理対象外です。
通常パイプラインは不要な重複HTML・ページPNGを整理し、正規の中間HTMLと
source registryだけを`outputs/renders/_build/`へ保持します。

### 成果物

- `outputs/SUMMARY_REPORT.md` / `.pdf`: 基本実験のみの本編
- `outputs/SUMMARY_REPORT_extra.md` / `.pdf`: 本編＋発展実験の付録C
- `outputs/exp_a/`〜`exp_d/`: 基本実験のCSV、図、環境情報
- `outputs/exp_a_extra/`〜`exp_d_extra/`: 発展実験のFold結果、集約CSV、
  比較図、個別図、Appendix原稿
- `outputs/discussion_draft/`: 実験ごとの考察ドラフト
- `outputs/reports/`: PDFレイアウト検査JSON、監査記録
- `docs/`: 実行計画と開発日報

`outputs/`は成果物・自動生成物、`docs/`は計画・開発記録という役割分担です。

### テストと最終確認

```bash
# 全テスト
PYTHONPATH=. pytest tests/

# import・構文確認
PYTHONPATH=. python3 -m compileall src scripts

# 差分の空白エラー確認
git diff --check
```

PDFのE2EテストはローカルHTTPサーバーとheadless Chromeを使用します。
ローカルソケットを禁止するサンドボックス内では、コードに問題がなくても
権限エラーになる場合があるため、通常のローカルシェルで実行してください。

### 再現性

- 共通乱数シードは`42`
- 外側CVは原則5分割Stratified K-Fold
- Before/Afterで同一Foldを共有
- 学習型前処理は学習Fold内だけでfit
- 基本実験はPython・OS・主要パッケージ情報を保存
- 生データとキャッシュはGit管理対象外
- 保存済みCSVから個別図・発展版レポートを再生成可能

### 既知の制約

- パッケージバージョンを固定するlockファイルは未整備です。
- OpenMLデータは初回取得時にネットワーク接続が必要です。
- livedoor News Corpusは別途取得し、ライセンスに従って利用する必要があります。
- 処理時間はハードウェア、OS負荷、辞書バージョンの影響を受けます。
- 意図的な未標準化条件ではLogistic Regressionが`max_iter`へ到達する場合があり、
  収束警告も実験結果の一部として記録します。
- 自動修復はID単位の改ページ・分割抑止を対象とし、任意のCSS問題を自動解決する
  汎用レイアウトエンジンではありません。

### 開発時の配置規約

1. 再利用する関数・クラスは`src/experiments/`または`src/reporting/`へ配置する。
2. CLIは薄いラッパーとして、用途に応じた`scripts/`配下へ配置する。
3. 基本実験CLIは`scripts/core/`、発展実験CLIは`scripts/extra/`、
   レポート共通CLIは`scripts/report/`を使用する。
4. データ、仮想環境、キャッシュ、レンダリング中間物をコミットしない。
5. Before/After比較ではFold共有とFold内fitを維持する。
6. 照合・ID・設定・レイアウトを変更する場合はテストを追加または更新する。
7. 分類に迷う新規ファイルは、配置先を決めてから作成する。
