# NLP Preprocessing Evaluation

[English](#english) | [日本語](#日本語)

## English

A reproducible evaluation framework for measuring how preprocessing choices affect
machine-learning model performance, explainability, prediction coverage, and
computational cost across four data scenarios and their extended experiments.

### Project structure

- `docs/`: execution plan and development records
- `assets/styles/`: CSS for PDF reports
- `scripts/core/`: CLI entry points for the core experiments
- `scripts/extra/`: extended experiments, result aggregation, and figure generation
- `scripts/report/`: report building, PDF layout checks, and page rendering
- `src/experiments/`: reusable preprocessing, modeling, evaluation, and explainability logic
- `src/reporting/`: reusable Markdown, HTML, and PDF reporting logic
- `outputs/`: experiment results, final reports, PDFs, audit logs, and intermediate renders
- `tests/`: unit and integration tests

### Main commands

Run the commands from a project virtual environment with all dependencies installed.

```bash
PYTHONPATH=. python3 scripts/core/run_exp_a.py
PYTHONPATH=. python3 scripts/extra/run_exp_a_permutation.py
PYTHONPATH=. python3 scripts/extra/build_summary.py
PYTHONPATH=. python3 scripts/extra/generate_extra_figures.py
PYTHONPATH=. python3 scripts/report/build_report.py
PYTHONPATH=. python3 scripts/report/check_pdf_layout.py
pytest tests/
```

### Source layout rules

- Place reusable functions and classes imported by other Python code under `src/`.
- Place core experiment CLI entry points under `scripts/core/`.
- Place extended-experiment-specific CLI entry points under `scripts/extra/`.
- Place experiment-independent report and PDF CLI entry points under `scripts/report/`.
- If a new file does not clearly fit one of these categories, decide its location before creating it.

---

## 日本語

前処理条件が機械学習モデルの性能・説明性・予測可能率・処理コストへ与える影響を、
4種類のデータ特性と発展実験で定量評価する、再現可能な評価フレームワークです。

### ディレクトリ

- `docs/`: 実行計画書・開発記録
- `assets/styles/`: PDFレポート用CSS
- `scripts/core/`: 基本実験のCLIエントリーポイント
- `scripts/extra/`: 発展実験・結果集計・図生成のCLI
- `scripts/report/`: レポート構築・PDF検査・ページ画像生成のCLI
- `src/experiments/`: 前処理・モデル・評価・説明性の再利用ロジック
- `src/reporting/`: Markdown・HTML・PDFレポートの再利用ロジック
- `outputs/`: 実験結果・最終レポート・PDF・監査ログ・中間レンダリング
- `tests/`: 単体・結合テスト

### 主要コマンド

依存パッケージを導入したプロジェクト仮想環境から実行してください。

```bash
PYTHONPATH=. python3 scripts/core/run_exp_a.py
PYTHONPATH=. python3 scripts/extra/run_exp_a_permutation.py
PYTHONPATH=. python3 scripts/extra/build_summary.py
PYTHONPATH=. python3 scripts/extra/generate_extra_figures.py
PYTHONPATH=. python3 scripts/report/build_report.py
PYTHONPATH=. python3 scripts/report/check_pdf_layout.py
pytest tests/
```

### 配置規約

- 他のPythonコードからimportする関数・クラスは`src/`へ置く。
- 基本実験CLIは`scripts/core/`へ置く。
- 発展実験専用CLIは`scripts/extra/`へ置く。
- 特定実験に依存しないレポート・PDF CLIは`scripts/report/`へ置く。
- 分類に迷う新規ファイルは、作成前に配置先を決定する。
