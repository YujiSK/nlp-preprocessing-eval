# PDF生成パイプライン監査レポート

作成日: 2026-07-29
対象: `task9/outputs/FINAL_REPORT.md` → `task9/outputs/FINAL_REPORT.pdf`

## 1. 現行パイプラインの構成（監査時点）

`FINAL_REPORT.pdf`は以下の使い捨てスクリプト（実行後に削除済み）によって生成された。再現性の観点から、実際に使用したコマンド・設定を本レポートに書き起こす。

### 1.1 実行コマンド・手順

1. `markdown`（Python実装、John Gruber's Markdown準拠）で`FINAL_REPORT.md`をHTML化する。
   ```python
   import markdown
   html_body = markdown.markdown(
       md_text,
       extensions=["extra", "tables", "fenced_code", "toc", "sane_lists"],
   )
   ```
2. 上記HTMLに固定のインラインCSS（`<style>`タグ）を付与し、`outputs/`直下に一時HTMLファイルとして書き出す（画像の相対パス — `exp_a/expA_cv_score_bar.png`等 — を解決するため、`outputs/`と同一階層に配置する必要があった）。
3. `python3 -m http.server`で`outputs/`をカレントディレクトリとしてローカルHTTPサーバーを起動する（`file://`プロトコルでは、印刷CSSのページ分割自体には影響しないが、別工程のMermaid図レンダリングでCDN経由スクリプトがCORSでブロックされた実績があり、本工程でも一貫してHTTPサーバー経由とした）。
4. Chromium（Google Chrome）のheadlessモードで当該URLを開き、PDFへ印刷する。
   ```
   google-chrome --headless=new --disable-gpu --no-sandbox --hide-scrollbars \
     --virtual-time-budget=8000 --no-pdf-header-footer \
     --print-to-pdf=FINAL_REPORT.pdf \
     http://localhost:8792/_final_report_render.html
   ```
5. 生成後、一時HTMLファイルおよびHTTPサーバーを削除・停止した。

### 1.2 レンダラー・バージョン

| 項目 | 値 |
| :--- | :--- |
| レンダラー | Chromium（Google Chrome, headless） |
| Chromeバージョン | Google Chrome 150.0.7871.186（HeadlessChrome/150.0.0.0、User-Agent文字列より） |
| PDF Producer | Skia/PDF m150（`pdfinfo`出力より） |
| PDFバージョン | 1.4 |
| ページサイズ | A4（594.96 × 841.92 pt、CSSの`@page { size: A4 }`指定通り） |
| Markdown変換 | Python `Markdown` パッケージ 3.10.2（pandoc・WeasyPrint・Playwrightは未使用。監査時点でいずれも環境に未インストール） |
| PDF後処理・ページ画像化 | `poppler-utils`（`pdftoppm` 26.01.0, `pdfinfo`）を目視検証のみに使用。PDF自体の後加工は行っていない |

Pandoc、WeasyPrint、Playwrightは監査時点で本環境に導入されていない（`which pandoc` 等はいずれも未検出）。ヘッドレスChromeの`--print-to-pdf`のみで完結する構成であり、追加のPDFライブラリ依存はない。

## 2. CSSと中間HTMLの再現可能性

### 2.1 使用CSS（監査時点、インライン埋め込み）

CSSはビルドスクリプト内にPython文字列として直接埋め込まれており、独立したファイルとしては存在しなかった。主な指定は以下の通り。

- `@page { size: A4; margin: 18mm 16mm; }`
- 日本語フォント: `"Noto Sans CJK JP", "Noto Sans", sans-serif`（システムにインストール済みのNoto Sans CJK JPパッケージに依存。Webフォント埋め込みは行っていない）
- `h2 { page-break-before: always; }`（章単位で改ページ。ただし最初のh2は`page-break-before: avoid`で例外化）
- テーブル・画像・コードブロックの罫線／背景／`max-width`調整
- **改ページ制御（`break-inside`, `orphans`/`widows`等）は監査時点では未設定** — 図表・見出し直後のブロックがページ境界で分断されないようにする明示的な制御が存在しなかった

### 2.2 中間HTMLの出力・再現可能性

**監査で確認された最大の問題点**: 中間HTML（Markdown変換後、Chromeに渡す直前のHTML）は`outputs/_final_report_render.html`として一時的に書き出されていたが、**PDF生成完了後にスクリプトごと削除しており、リポジトリ上には残存していない**。すなわち：

- 現在の`FINAL_REPORT.pdf`がどのHTML構造から生成されたか、後から検証する手段がない。
- ビルドスクリプト自体もリポジトリに保存されておらず、同一のPDFを再生成するには本監査レポートの記述を元に手動で再構築する必要がある。
- HTML中に見出し・図・表・コードブロックを一意に識別するID（`data-source-id`等）は一切付与されておらず、特定の要素が最終PDFのどのページに配置されたかをプログラム的に追跡する手段が存在しなかった。

### 2.3 監査結論と対応方針

上記の再現性・追跡可能性の欠如を解消するため、本監査に続く実装では以下を行う。

1. ビルドスクリプトを`src/report_build.py`として恒久化し、中間HTMLを`outputs/_build/FINAL_REPORT.render.html`として毎回保存する（使い捨てにしない）。
2. 見出し・図・表・コードブロックに`data-source-id`を安定的に付与し、`outputs/_build/source_registry.json`にID→要素種別・代表テキストのマッピングを出力する。
3. 印刷用CSSを`break-inside` / `orphans` / `widows`等を含む形に強化し、独立したCSS定義として管理する。
4. `configs/layout_overrides.yml`により、個別要素の改ページ・分割抑制を宣言的に上書きできる仕組みを追加する。
5. `src/pdf_layout_checker.py`により、生成後のPDFを`pdfplumber`で解析し、`source_registry.json`と突き合わせてレイアウト不整合を検出する。

これらの詳細な実装結果は`outputs/reports/layout_report.md`に記録する。
