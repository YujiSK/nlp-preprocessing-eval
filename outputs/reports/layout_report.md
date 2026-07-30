# FINAL_REPORT.pdf レイアウト品質検証・修復レポート

作成日: 2026-07-29
対象: `task9/outputs/FINAL_REPORT.md` → `task9/outputs/FINAL_REPORT.pdf`

前提として、本パイプラインは「PDF解析のみで完璧な自動修復ができる」ことを目指すものではない。
中間HTMLに `data-source-id` を付与し、`configs/layout_overrides.yml` による宣言的な上書きと組み合わせることで、
検出・修正・検証を反復可能にする構成としている。検出ロジック自体もテキスト内容に基づくヒューリスティックであり、
その限界は本レポート末尾「4. 既知の限界」に明記する。

---

## 1. パイプライン構成

| コンポーネント | 役割 |
| :--- | :--- |
| `src/report_build.py` | `FINAL_REPORT.md` → HTML変換、見出し/図/表/コードへの`data-source-id`付与、`layout_overrides.yml`反映、`source_registry.json`出力 |
| `src/report_style.css` | 印刷用CSS（`@media print`によるページ分割制御をバージョン管理下に配置） |
| `src/pdf_render.py` | HTML→PDF変換（Chromium headless）、PDF→ページ画像変換（`pdftoppm`） |
| `src/pdf_layout_checker.py` | `pdfplumber`によるPDF解析、`source_registry.json`との突き合わせによる違反検出 |
| `src/pdf_layout_pipeline.py` | build→render→check→修復の反復オーケストレータ（最大3回、ロールバック付き） |
| `configs/layout_overrides.yml` | `page_break_before` / `keep_together` の宣言的オーバーライド定義 |

中間HTMLは `outputs/_final_report_render.html`（画像相対パス解決用）と `outputs/_build/FINAL_REPORT.render.html`（控え）の2箇所に保存し、`outputs/_build/source_registry.json` にID一覧を出力する。これにより、`pdf_pipeline_audit.md` で指摘した「使い捨てHTML・ID未付与による再現性の欠如」を解消した。

---

## 2. 検出ルールの定義

| 違反種別 | 定義 | 除外対象 |
| :--- | :--- | :--- |
| `orphan_heading` | 見出しと直後の本文ブロックのページ番号が異なる、または見出しがページ下端付近（高さ比0.90超）に位置する | 文書末尾で比較対象がない見出し |
| `figure_caption_image_split` | 図のキャプションが存在するページに画像が1枚も存在しない | — |
| `short_block_split` | 表・コードの先頭要素と末尾要素のページ番号が異なる | `size_class`が`long-table` / `long-code`（表15行・コード30行超、分割許可）のもの |
| `overflow` | 文字の座標が`@page`マージン（上下18mm・左右16mm、±3ptの許容誤差）を超えている | — |

スコアは `-( orphan_heading×2 + figure_caption_image_split×3 + short_block_split×2 + overflow×1 )` の重み付き件数の符号反転とし、0が違反なしを表す。

---

## 3. 実行結果

### 3.1 レジストリ集計

`source_registry.json` に登録された要素数: **59件**（見出し40、図11、表5、コード3）。

### 3.2 修復ループの推移

| Iteration | スコア | 違反件数 | 違反種別 | 判定 |
| :---: | :---: | :---: | :--- | :--- |
| 1 | 0 | 0 | なし | `accepted`（0件のため即時終了） |

初回のビルド・レンダリング・検査で違反0件を達成したため、自動修復（`page_break_before` / `keep_together` オーバーライドの追加）は発動しなかった。`configs/layout_overrides.yml` は空のまま（`page_break_before: []`, `keep_together: []`）で確定した。

修復ループ自体（提案生成・スコア比較・改善なし時のロールバック・最大3回での停止）は、`tests/test_pdf_layout_checker.py` の `test_repair_loop_stops_when_zero_violations` および `test_repair_loop_rolls_back_when_no_improvement` にて、スコア推移を制御した模擬実行により、停止条件・ロールバック動作を個別に検証済みである（4章参照）。

### 3.3 最終成果物

| 項目 | 値 |
| :--- | :--- |
| `outputs/FINAL_REPORT.pdf` ページ数 | 24 |
| 検出違反件数 | **0件** |
| 未解決（テキスト照合失敗）件数 | 0件 |
| `outputs/renders/page-001.png` 〜 `page-024.png` | 全24ページを画像化済み |

**結論**: 定義したレイアウト検査規則（`orphan_heading` / `figure_caption_image_split` / `short_block_split` / `overflow`）において、現行の`FINAL_REPORT.pdf`は検出違反0件であることを確認した。ただし、これは本パイプラインが定義する検査規則の範囲内での確認であり、目視で識別しうるあらゆるレイアウト上の問題（フォントの視覚的な詰まり、色のコントラスト、日本語の禁則処理の細部等）を網羅的に保証するものではない。

---

## 4. テスト（`tests/test_pdf_layout_checker.py`）

`tests/fixtures/sample_layout.md`（トレイリング見出し、図とキャプション、長短の表・コードを含む最小サンプル）および合成`PageData`/レジストリを用いて、以下13件のテストを実装・全件合格を確認した。

- 検出ロジックの単体テスト（9件）: 孤立見出しの検出/非検出、文書末尾見出しの除外、図とキャプションの分離検出/非検出、短い表・短いコードの分割検出、長い表の分割除外確認、はみ出し検出/非検出
- 結合テスト（1件）: `tests/fixtures/sample_layout.md`を実際にビルド→PDF化→検査し、違反0件・未解決0件であることを確認
- 修復ループのテスト（2件）: スコアが改善する場合の停止動作、改善しない場合のロールバック動作をモックしたビルド結果で検証
- （上記1件を含め計13件）

```
$ pytest tests/test_pdf_layout_checker.py -v
13 passed
```

---

## 5. 既知の限界

1. **テキスト照合はヒューリスティックである**: Chromium(Skia)がPDF化する際、一部のCJK文字が字形の近い別のUnicodeコードポイント（例: "方"→"⽅"、"言"→"⾔"、"長"→"⻑"）に置換されて埋め込まれる事象が確認された。厳密な文字列一致では大多数のテキストが不一致と誤判定されるため、本パイプラインではNFKC正規化と、NFKCで解決できない既知の置換（現時点で2件: "・"⇔"‧"、"長"⇔"⻑"）の手動マッピングに加え、トライグラム（3文字連続部分列）の含有率（閾値0.6）によるファジーマッチングを採用している。**この置換パターンは網羅的に把握されたものではなく、未知の置換が新たな文書で見つかった場合、該当箇所が「未解決（unresolved）」として報告される可能性がある。**
2. **反復する見出しラベルの位置特定**: 「【事実】」「【メカニズム】」等、章をまたいで同一文言が繰り返される見出しは、テキスト内容だけでは一意に位置を特定できない。本パイプラインでは、レジストリが文書出現順であることを前提に、直前に解決済みの要素のページ番号を下限（カーソル）として順次探索することで対処している。**レジストリの並び順が文書順であるという前提が崩れた場合（将来的な仕様変更等）、この解決方式は機能しない。**
3. **図とキャプションの分離検出は代理指標である**: 実際に特定の画像とキャプションの対応関係をPDF内で直接追跡する手段がないため、「キャプションが存在するページに画像が1枚以上存在するか」を代理の判定基準としている。同一ページに複数の図がある場合や、無関係な画像が同ページに存在する場合、厳密な意味での「その図とキャプションの対応」を保証するものではない。
4. **自動修復の対象は限定的である**: `page_break_before` / `keep_together` はID単位のCSSクラス付与で実現できる違反（`orphan_heading`, `figure_caption_image_split`, `short_block_split`）にのみ適用可能である。`overflow`（はみ出し）はID単位のオーバーライドでは直接解消できないため、自動修復の対象外とし、検出のみ行う。
5. **座標ベースのはみ出し検出はサンプリング的である**: `detect_overflow`は各ページの最初に閾値を超えた語1件のみを違反として記録し、同一ページ内の他の超過箇所を網羅的には列挙しない。
6. **スコアの重み付けは暫定である**: 違反種別ごとの重み（`orphan_heading`=2等）は本パイプライン内で定義した相対的な重要度であり、業務要件に応じた妥当性検証は行っていない。

以上より、本レポートの「検出違反0件」という結論は、**上記の検査規則・照合手法の範囲内で確認されたものであり、PDFレイアウトの完全性を保証するものではない**。将来的にFINAL_REPORT.mdの内容が変更された場合は、`python3 -m src.pdf_layout_pipeline` を再実行し、本レポートを再生成することを前提とする。
