# Task 9 — NLP前処理パイプライン定量評価

前処理条件が機械学習モデルの性能・説明性・処理コストへ与える影響を、
4種類のデータ特性と発展検証で評価するプロジェクトである。

## ディレクトリ

- `docs/`: 実験計画書
- `docs/reports/`: 日報・作業記録
- `assets/styles/`: PDFレポート用CSS
- `scripts/core/`: 基本実験のCLIエントリーポイント
- `scripts/extra/`: 発展実験・発展版集計・発展図生成のCLI
- `scripts/report/`: レポート全体の構築・PDF検査・ページ画像生成CLI
- `src/experiments/`: 実験・評価・前処理の再利用ロジック
- `src/reporting/`: Markdown/HTML/PDF構築とレイアウト検査ロジック
- `outputs/`: 実験結果、最終レポート、PDF、検査ログ、中間レンダリング
- `tests/`: 単体・結合テスト

## 主要コマンド

```bash
PYTHONPATH=. python3 scripts/core/run_exp_a.py
PYTHONPATH=. python3 scripts/extra/run_exp_a_permutation.py
PYTHONPATH=. python3 scripts/extra/build_summary.py
PYTHONPATH=. python3 scripts/extra/generate_extra_figures.py
PYTHONPATH=. python3 scripts/report/build_report.py
PYTHONPATH=. python3 scripts/report/check_pdf_layout.py
pytest tests/
```

依存パッケージを導入したプロジェクト仮想環境から実行すること。

## 配置規約

- 他コードからimportする関数・クラスは`src/`へ置く。
- 基本実験CLIは`scripts/core/`、発展実験専用CLIは`scripts/extra/`へ置く。
- 特定実験に依存しないレポート・PDF CLIは`scripts/report/`へ置く。
- 分類に迷う新規ファイルは配置せず、事前に配置案を確認する。
