# 日報（進行中）— 課題9：前処理パイプライン定量評価プロジェクト

日付: 2026-07-30

本ファイルは当日分の作業ログである。以降、Claude Codeが実施した作業を実施時刻とともに追記する。

## 作業ログ

### 09:01 — 日報ファイルの整理
- `outputs/reports/daily_report_draft.md` を `outputs/reports/daily_report_20260729.md` にリネームし、確定版の内容（本日の作業内容・気付き・問題点・明日の予定）に更新した。
- 本ファイル（`daily_report_20260730.md`）を新規作成し、以降の作業をタイムスタンプ付きで記録する運用を開始した。

### 09:32 — 現行PDFの同期・自動検査
- `python3 -m src.pdf_layout_pipeline` を実行した。
- 現行PDF（全18ページ）の同期と自動検査が完了し、レイアウト違反は0件だった。

### 09:46 — 実験A（発展）の作業開始
- `task9_plan.md` を確認し、発展検証「Permutation Importanceを算出しBefore/After・モデル間で比較する」に着手した。
- 既存のレポート本文・図表・PDFスタイリングは変更せず、実験成果と考察は新規Appendix向け成果物として分離する方針を確認した。
- 既存の実験Aスクリプト、共通評価・説明性モジュール、成果物を調査し、検証Fold上でのPermutation Importance算出手順を整理する。

### 09:50 — 実験A（発展）の完了
- `scripts/run_experiment_a_permutation.py` を新規作成した（10:26の整理方針修正後の最終パス）。本文と同一の5分割StratifiedKFoldを全条件・全モデルで共用し、各学習FoldでPipelineをfitした後、未使用の検証Fold上でAccuracyベースのPermutation Importanceを各特徴量30反復で算出する。
- システム側の`python3`にはmatplotlibが導入されておらず初回実行はimport時に停止したため、プロジェクト仮想環境`venv/bin/python`で再実行して完了した。
- 既存本文・既存図表・`FINAL_REPORT.md`・`FINAL_REPORT.pdf`・既存PDFスタイルには変更を加えず、成果物を`outputs/exp_a_extra/`へ分離保存した（10:26の整理方針修正後の最終パス）。
- 出力: repeat単位Long CSV（36,000行）、条件×モデル×特徴量の集約CSV（240行、各セル150観測）、Fold平均CSV、Before/Afterペア差CSV、4モデル比較図、Appendix原稿。
- 最大平均PIは、BeforeではLogistic Regression=`worst area` 0.2776、Linear SVC=`worst area` 0.3447、Random Forest=`worst area` 0.0116、k-NN=`worst area` 0.3814だった。Afterでは順に`worst texture` 0.0172、`worst texture` 0.0379、`worst area` 0.0116、`worst texture` 0.0107だった。
- Random ForestのBefore/After差は全特徴量中の最大絶対差でも0.0006で、標準化に対する実質的な不変性が確認された。線形モデルとk-NNでは重要度分布が大きく変化したが、相関特徴量の存在およびモデル性能・収束状態の変化を含むため、独立した因果寄与とは解釈しない。
- 未標準化Logistic Regressionは5 Foldすべてで`max_iter=2000`到達の収束警告が発生した。既存条件を維持してパラメータは変更せず、Appendixに解釈上の制約として明記した。
- 検証: `py_compile`成功、集約CSVの重複0・欠損0・各セル150観測を確認、比較図を目視確認。`tests/test_common_modules.py`は3件すべて成功（既知の未標準化Logistic Regression収束警告6件）。
- `task9_plan.md`の実験A（発展）チェック項目を完了に更新した。

### 09:51 — 終了時の全体確認
- `pytest tests -q`を実行し、16件中15件が成功した。失敗1件はPDFレイアウトのE2E fixtureがlocalhost用socketを作成する箇所で、実行サンドボックスのネットワーク制限により`PermissionError: [Errno 1] Operation not permitted`となったもの。実験Aの計算・集約・描画処理に起因する失敗ではない。
- 既存成果物の更新日時を確認し、`FINAL_REPORT.md`および`src/report_style.css`が未変更であることを確認した。`FINAL_REPORT.pdf`は引き継ぎ記録どおり09:32更新のままで、今回の実験では再生成していない。
- 実験A（発展）の実装、実行、成果物検証、Appendix原稿作成、計画書・日報更新を完了した。

### 10:13 — 実験B（発展）の作業開始
- `task9_plan.md`の副実験要件を再確認し、推論時欠損を含む実データで、Before（完全行のみ予測可能）とAfter（補完により全行予測可能）のCoverage、予測可能サンプル限定性能、全サンプルに対する実用性能を分離評価する作業に着手した。
- 既存レポート本文・図表・PDF・スタイルは変更せず、実装と成果物を新規Appendix専用ディレクトリへ分離する。
- 既存の実験Bコードとローカルデータを調査したところ、実験B本体は合成データで、Titanic等の欠損を含む実データは未保存だったため、出典と再現性を明示できる公開データの取得方法を確認する。

### 10:19 — 実験B（発展）の完了
- OpenML Titanic v1（data_id=40945、1,309件）を取得した。通常サンドボックス内の初回取得はDNS制限で失敗したため、承認済みネットワーク実行で取得し、`data_cache/openml/`へキャッシュした。
- `scripts/run_experiment_b_coverage.py`を新規作成した（10:26の整理方針修正後の最終パス）。予測時に利用可能な7特徴量（pclass, sex, age, sibsp, parch, fare, embarked）を使用し、結果発生後のboat/body、識別性の高いname/ticket、欠損率の高いcabin/home.destを除外した。
- 本文と同一設定の5分割StratifiedKFoldを全条件・全モデルで共用した。Beforeは学習時・推論時とも完全行のみを使用し、欠損を含む推論行は予測不能（abstain）とした。AfterはFold内で数値中央値・カテゴリ最頻値を学習して補完し、全行を予測した。
- Beforeの平均Coverageは0.7968（予測不可率0.2032、Fold範囲0.7824〜0.8130）、Afterは全Fold・全モデルで1.0000だった。自然欠損の中心はage 263件（20.09%）で、fare 1件、embarked 2件だった。
- 全推論要求に対する正解割合はBefore→Afterで、Logistic Regression 0.6257→0.7876（+0.1620）、Linear SVC 0.6211→0.7869（+0.1658）、Random Forest 0.6279→0.7975（+0.1696）、k-NN 0.6287→0.8075（+0.1788）となった。
- Afterの欠損行AccuracyはLogistic Regression 0.7927、Linear SVC 0.7992、Random Forest 0.7990、k-NN 0.8255だった。ただしCoverage 100%は出力可能性を示すだけで信頼性を保証せず、Titanicの自然欠損はMCARとは限らない旨をAppendixに明記した。
- `outputs/exp_b_extra/`にFold別指標CSV（320行）、集約CSV（64行）、欠損プロファイルCSV、データセットメタデータJSON、比較図、`APPENDIX_EXP_B_COVERAGE.md`を保存した（10:26の整理方針修正後の最終パス）。
- 検証: `py_compile`成功、集約キー重複0・欠損0、After Coverage全件1.0、比較図を目視確認。`tests/test_common_modules.py`は3件すべて成功（既知の未標準化Logistic Regression収束警告6件）。
- `task9_plan.md`の実験B（発展）チェック項目を完了に更新した。`FINAL_REPORT.md`、`FINAL_REPORT.pdf`、`src/report_style.css`は今回変更していない。

### 10:22 — Appendix関連ファイル整理の作業開始
- 増加した発展実験成果物と発展実験スクリプトを専用階層へ集約する作業に着手した（この時点の親フォルダ案は10:26のユーザー指示により最終配置へ修正）。
- 移動前に旧出力パス・旧スクリプトパスの参照箇所を全検索し、スクリプト2件、`task9_plan.md`、当日日報に参照があることを確認した。
- 移動後も任意の作業ディレクトリから実行でき、`src`・`data_cache`・出力先を正しく解決できるよう、スクリプトのTask 9ルート解決処理を調整してから動作確認する。

### 10:26 — 整理方針修正への対応開始
- ユーザー指示により、10:22時点の中間方針（`appendices/`親フォルダ）は採用せず、成果物を`outputs/exp_a_extra/`・`outputs/exp_b_extra/`、スクリプトを`scripts/`直下へ置く最終配置へ修正する。
- 中間配置からの全ファイル移動、Task 9ルート解決処理、計画書・日報内の参照、実行・テストを最終配置に合わせて再確認する。

### 10:30 — ファイル整理の完了
- 最終配置を以下に統一した。`appendices/`という親ディレクトリは残していない。

```text
task9/
├── scripts/
│   ├── README.md
│   ├── run_experiment_a_permutation.py
│   └── run_experiment_b_coverage.py
└── outputs/
    ├── exp_a_extra/
    │   ├── APPENDIX_EXP_A_PERMUTATION.md
    │   ├── expA_permutation_importance_long.csv
    │   ├── expA_permutation_importance_summary.csv
    │   ├── expA_permutation_importance_fold_means.csv
    │   ├── expA_permutation_importance_before_after_diff.csv
    │   └── expA_permutation_importance_comparison.png
    └── exp_b_extra/
        ├── APPENDIX_EXP_B_COVERAGE.md
        ├── expB_coverage_fold_metrics.csv
        ├── expB_coverage_summary.csv
        ├── expB_missingness_profile.csv
        ├── expB_dataset_metadata.json
        └── expB_coverage_comparison.png
```

- 両スクリプトの`TASK9_ROOT`を`scripts/`直下の配置に合わせて`Path(__file__).resolve().parents[1]`へ変更し、`src`、`data_cache/openml`、出力先をファイル位置基準で解決するようにした。
- 今後の発展実験は、実験Cを`outputs/exp_c_extra/`、実験Dを`outputs/exp_d_extra/`へ保存する規約を`scripts/README.md`に明記した。
- `task9_plan.md`と当日日報の旧パス参照を最終配置へ更新した。旧パス文字列および旧`appendices/`ディレクトリが残っていないことを全検索で確認した。
- 最終配置のスクリプトから実験A・Bを再実行し、それぞれ`outputs/exp_a_extra/`、`outputs/exp_b_extra/`へ正常出力されることを確認した。実験Aの未標準化Logistic Regression収束警告は既知の挙動で、パス移動によるエラーはなかった。
- サンドボックス外でPDF E2Eを含む全テストを再実行し、16件すべて成功した（既知の収束警告6件のみ）。
- `FINAL_REPORT.md`、`FINAL_REPORT.pdf`、`src/report_style.css`の更新日時は整理前から変わらず、既存レポート・PDF・スタイルを変更していない。

### 10:57 — 実験C（発展）の作業開始
- `task9_plan.md`の「内側CVのみで閾値を選択し外側Foldで評価する」項目に着手した。
- 既存の実験C実装を確認し、同一の不均衡合成データと外側5-Foldを再利用する。閾値効果をサンプリング効果と混在させないため、補正なしC0条件のLogistic RegressionとLinear SVCを対象とする。
- 各外側train内で3-FoldのOOF確率を生成し、F1最大化閾値を選択してから外側testへ一度だけ適用するNested CVとする。Linear SVCは各学習範囲内だけで`CalibratedClassifierCV`をfitし、外側testを閾値選択・校正に使用しない。
- スクリプトは`scripts/run_experiment_c_threshold.py`、成果物は`outputs/exp_c_extra/`へ新規出力し、既存レポート・既存実験C成果物・スタイルは変更しない。

### 11:01 — 実験C（発展）の完了
- `scripts/run_experiment_c_threshold.py`を新規作成し、既存実験Cと同一の2,000件・正例約7%の不均衡合成データ、補正なしC0条件でNested CVを実装・実行した。
- 外側5-Foldの各train内で3-Fold OOF確率を生成し、0.05〜0.95（0.01刻み）の91候補からF1最大化閾値を選択した。外側testは閾値選択および確率校正に使用していない。
- Logistic RegressionはPipelineの`predict_proba`、Linear SVCは各学習範囲内で`CalibratedClassifierCV(method="sigmoid", cv=3)`をfitして確率を取得した。
- 選択閾値はLogistic Regressionが0.26〜0.35（平均0.316±0.038）、Linear SVCが0.26〜0.33（平均0.298±0.029）で、全外側Foldにおいて既定0.5より低かった。
- 外側Fold平均F1はLogistic Regressionが0.5243→0.5746（+0.0502）、Linear SVCが0.5273→0.5840（+0.0567）となった。Fold単位ではLogistic Regressionは5/5 Foldで改善、Linear SVCは4/5 Foldで改善・1 Foldで0.0111悪化した。
- RecallはLogistic Regression 0.3922→0.5278、Linear SVC 0.3853→0.5409へ上昇し、Precisionはそれぞれ0.8190→0.6417、0.8600→0.6377へ低下した。F1最適化による意図したトレードオフとしてAppendixへ記録した。
- Average PrecisionとROC-AUCは同一確率ランキングに対する閾値非依存指標のため、default/tuned間の外側Fold差が全件0であることを確認した。
- `outputs/exp_c_extra/`にOuter Fold指標CSV（160行）、集約CSV（32行）、閾値選択CSV（10行）、Inner探索グリッドCSV（910行）、Foldペア差CSV（80行）、比較図、`APPENDIX_EXP_C_THRESHOLD.md`を保存した。
- 検証: `py_compile`成功、全CSVの欠損0、Outer指標キー重複0、選択閾値が全件Inner候補内に存在すること、比較図の視認性を確認した。
- PDF E2Eを含む全テストを再実行し、16件すべて成功した（既知の未標準化Logistic Regression収束警告6件のみ）。
- `task9_plan.md`の実験C（発展）チェック項目を完了に更新した。既存レポート本文・既存実験C成果物・PDF・スタイルは変更していない。

### 11:10 — 実験D（発展）の作業開始
- `task9_plan.md`のD0〜D3アブレーション項目に着手した。既存のlivedoor News Corpus読込・重複除去・メタデータリーク除去・TF-IDF設定・外側5-Foldを維持する。
- 条件をD0=クレンジングなし×MeCab/IPAdic、D1=neologdnクレンジングあり×MeCab/IPAdic、D2=クレンジングなし×Sudachi core/Mode C、D3=neologdnクレンジングあり×Sudachi core/Mode Cとして2×2要因を分離する。
- クレンジング、各解析条件、TF-IDF語彙数、モデル評価時間を個別計測し、クレンジング効果（D1−D0、D3−D2）と解析器効果（D2−D0、D3−D1）を同一Fold・同一モデルのペア差として集計する。
- スクリプトは`scripts/run_experiment_d_ablation.py`、成果物は`outputs/exp_d_extra/`へ新規出力し、既存レポート・既存実験D成果物・PDFスタイルは変更しない。

### 11:21 — 実験D（発展）の完了
- `scripts/run_experiment_d_ablation.py`を新規作成し、重複除去後のlivedoor News Corpus 7,361記事に対してD0〜D3の2×2アブレーションを実行した。
- 条件はD0=raw×MeCab/IPAdic、D1=neologdn×MeCab/IPAdic、D2=raw×Sudachi core/Mode C、D3=neologdn×Sudachi core/Mode Cとし、同一の外側5-Fold、TF-IDF設定、4モデル、seed=42を共用した。TF-IDFは各学習Fold内のみでfitした。
- D0の全CV指標が既存実験DのBefore、D3の全CV指標が既存Afterと完全一致（最大絶対差0）することを確認し、元実験との再現整合性を検証した。
- macro-F1のクレンジング効果は、MeCab/IPAdicでLogistic Regression −0.0031、Linear SVC +0.0001、Random Forest −0.0037、k-NN −0.0078、Sudachiで順に−0.0026、−0.0007、−0.0033、−0.0028だった。Sudachi条件では全モデルで小幅に低下した。
- 解析器効果はraw条件で順に+0.0002、−0.0013、−0.0012、+0.0002、cleaned条件で+0.0007、−0.0022、−0.0009、+0.0052となり、モデル間で方向が異なり一貫した改善は観測されなかった。
- 語彙数はD0/D1=42,123、D2=47,222、D3=46,936だった。neologdnによる語彙減少はMeCab/IPAdicで0、Sudachiで286語（0.61%）だった。
- 単一実行の決定論的前処理時間はD0=5.94秒、D1=10.23秒、D2=12.79秒、D3=16.44秒だった。高度解析とクレンジングはいずれも本データ・実装で処理コストを増加させた。時間は単一実行の参考値である旨をAppendixに明記した。
- `outputs/exp_d_extra/`にFold指標CSV（240行）、集約CSV（48行）、Fold要因差CSV（80行）、要因差集約CSV（16行）、語彙数・前処理時間・トークン統計CSV、比較図、`APPENDIX_EXP_D_ABLATION.md`を保存した。
- 検証: `py_compile`成功、全CSVの欠損0、Fold指標キー重複0、全モデル×全効果が各5 Fold存在すること、空文書0件、比較図の視認性を確認した。
- PDF E2Eを含む全テストを再実行し、16件すべて成功した（既知の未標準化Logistic Regression収束警告6件のみ）。
- `task9_plan.md`のD0〜D3アブレーション項目を完了に更新した。`FINAL_REPORT.md`、`FINAL_REPORT.pdf`、`src/report_style.css`の更新日時は変わらず、既存成果物は変更していない。

### 11:37 — 最終統合・PDFビルド作業の開始
- 全発展実験A〜Dの完了を受け、本編Markdownの改名・無改変複製、考察ドラフト整備、個別図生成、発展版Markdown統合、2種類のPDF生成・レイアウト検査に着手した。
- `FINAL_REPORT.md`の本文内容は変更せず`SUMMARY_REPORT.md`として保持し、発展版のみ末尾へAppendixを追加する。
- 現行PDFパイプラインが`FINAL_REPORT`固定名であることを確認したため、同じCSS・検査規則を用いて`SUMMARY_REPORT`と`SUMMARY_REPORT_extra`を順に処理できるよう、入出力名だけを一般化する。

### 11:47 — 全工程の完了
- `outputs/FINAL_REPORT.md`を`outputs/SUMMARY_REPORT.md`へ改名し、改名前後のSHA-256が`bb0b088c...e5605c5e25`で一致することを確認した。本編内容は無改変で保持した。
- `SUMMARY_REPORT.md`を基に`SUMMARY_REPORT_extra.md`を生成し、本編全文を完全な接頭部分として保持したうえで、付録C〜Fとして発展実験A〜Dを末尾へ統合した。再生成スクリプトは`scripts/build_summary_extra.py`。
- `outputs/discussion_draft/exp_a_extra.md`〜`exp_d_extra.md`を作成し、収束警告、Coverage、Nested CV、処理強度・語彙数・時間のトレードオフに関する実務的示唆を各Appendixへ反映した。
- `scripts/generate_extra_report_figures.py`を作成し、保存済みCSVからモデル別個別図15枚（A=4、B=4、C=2、D=4＋資源比較1）を300dpiで生成した。各Appendixの参照を個別図へ置換し、一括比較図は従来どおり保持した。
- `src/report_build.py`と`src/pdf_layout_pipeline.py`をレポート名ごとの入出力へ一般化し、同一CSS・オーバーライド・検査規則で本編と発展版を順次生成できるようにした。検査結果は`outputs/reports/layout_summary_report.json`と`layout_summary_report_extra.json`へ保存する。
- 仮想環境をPATH先頭にした`python3 -m src.pdf_layout_pipeline`を最終実行した。`SUMMARY_REPORT.pdf`は18ページ・違反0件・未解決0件、`SUMMARY_REPORT_extra.pdf`は34ページ・違反0件だった。
- 発展版には文字抽出照合不能による未解決見出し12件が記録されたが、違反検出ではない。Appendix開始ページ、個別図ページ、最終ページをレンダリング画像で目視し、画像欠落、見切れ、はみ出し、末尾切断がないことを確認した。全15個別画像のHTTP取得も200応答だった。
- 実験Dの残存チェック項目を閉じるため`scripts/audit_exp_d_structure.py`を実行した。重複URL・重複filenameは0件で明示的group IDもないためGroup分割は不要と判断し、固定コーパス内要因比較では共通StratifiedKFoldを維持した。将来時点への一般化には別途時系列評価が必要と`outputs/exp_d_extra/expD_temporal_group_audit.csv`へ記録した。
- `task9_plan.md`の実施チェックリストに未完了項目が0件であることを確認した。
- 最終状態でPDF E2Eを含む全テストを実行し、16件すべて成功した（既知の未標準化Logistic Regression収束警告6件のみ）。

### 11:49 — 大規模ディレクトリリファクタリングの開始
- 指定された責務分離に従い、基本実験CLIを`scripts/core/`、発展実験CLIを`scripts/extra/`、レポートCLIを`scripts/report/`へ分類し、再利用ロジックを`src/experiments/`と`src/reporting/`へ移す作業に着手した。
- 移動前のPython import、設定、テスト、計画書、レポート内パス参照を全検索した。
- `git`実行ファイルが環境に存在しないことを確認したため、ファイルシステム上の整理と`.gitignore`更新は実施するが、`git rm --cached`は実行不能として完了時に明記する。

### 11:56 — 大規模ディレクトリリファクタリングの完了
- `docs/task9_plan.md`、`assets/styles/report.css`へ静的資料を移し、責務別に`scripts/core/`、`scripts/extra/`、`scripts/report/`、`src/experiments/`、`src/reporting/`を整備した。ルート`README.md`に配置規約と主要CLIを記載した。
- 基本実験CLIを`scripts/core/run_exp_a.py`〜`run_exp_d.py`、発展実験・集計・図生成CLIを`scripts/extra/`、PDF構築・検査・ページ画像生成CLIを`scripts/report/`へ移動・改名した。
- 再利用ロジックを`src/experiments/{evaluation,explainability,models,preprocessing}.py`と`src/reporting/{layout_checker,layout_pipeline,pdf_renderer,report_builder}.py`へ分離した。実験Cの共通不均衡データ生成も`src/experiments/preprocessing.py`へ集約した。
- 全Python import、プロジェクトルート解決、CSS・設定・計画書・レポート内の現行パス参照を新構成へ更新し、全14 CLIファイルのimport path checkに成功した。`SUMMARY_REPORT_extra.md`は新しい集計CLIから再生成した。
- `/home/rb132/Desktop/Sunagawa/nlp_preprocessing/venv/bin/python -m pytest -p no:cacheprovider tests/`を実行し、16件すべて成功した（既知の未標準化Logistic Regression収束警告6件のみ）。
- `scripts/report/build_report.py`で両PDFを再構築した。`SUMMARY_REPORT.pdf`は19ページ・違反0件・未解決0件、`SUMMARY_REPORT_extra.pdf`は35ページ・違反0件（文字抽出照合不能の既知未解決見出し12件）だった。続けて`PYTHONPATH=. python3 scripts/report/check_pdf_layout.py`相当の新CLIでも違反0件を再確認した。
- ルート`.gitignore`へPython/pytestキャッシュ、仮想環境、OpenMLキャッシュ、PDF中間生成物を追加した。全`__pycache__`、`.pytest_cache`、旧コード4件を含む`archive_old/`を削除した（ワークスペース上では復元不能。必要時は外部バックアップを使用）。 
- `git`コマンドが環境にインストールされていないため、指定された`git rm --cached`のみ実行不能だった。追跡解除が必要な場合はGit導入後に`.gitignore`対象を確認して実行する。
- `docs/task9_plan.md`の未完了チェックボックスが0件であることを確認した。

### 12:01 — レポートCLI最終確認の開始
- `scripts/report/`の3エントリーポイントについて、`src/reporting/`への委譲、`__main__`ガード、処理本体の重複有無を確認した。
- 指定されたcompileall、Git状態、キャッシュおよび旧名称重複ファイルの最終確認に着手した。

### 12:03 — レポートCLI最終確認の完了
- `build_report.py`は既に薄いラッパーだった。`check_pdf_layout.py`と`render_pdf_pages.py`に残っていた反復処理を、それぞれ`src/reporting/layout_checker.py`と`src/reporting/pdf_renderer.py`へ移し、3本すべてをimport・`__main__`ガード・`SystemExit`のみの薄いCLIに統一した。
- `PYTHONPATH=. python3 -m compileall src scripts`を実行し、全モジュール・全スクリプトのコンパイルに成功した。
- compileallが生成した`src/`・`scripts/`配下の`__pycache__`を確認後に削除し、旧名称の実験・レポートスクリプトおよび意図しない重複ファイルが0件であることを確認した。
- `git status --short`は実行を試みたが、この環境には引き続き`git`実行ファイルがなく、終了コード127（`git: command not found`）だった。このためGit追跡状態の一覧のみ取得不能である。

### 12:07 — Git導入後の状態再確認
- ユーザー側でGitを導入後、`git --version`が`2.53.0`を返すことを確認した。
- `git status --short`を再実行したが、親ディレクトリ`nlp_preprocessing/.git`は空ディレクトリ（`HEAD`・`config`等なし）で、有効なGitリポジトリではないため`fatal: not a git repository`となった。既存履歴を推測して初期化することは避けた。
- ファイルシステム上では`__pycache__`、`.pytest_cache`、旧名称の実験・レポートスクリプト、意図しない重複ファイルがいずれも0件であることを再確認した。

### 13:11 — 日報・レンダリング中間生成物の再配置開始
- 日報を`docs/reports/`へ移し、`outputs/reports/`をPDFビルド・レイアウト検査ログ専用として維持する整理に着手した。
- `outputs/FINAL_REPORT.pdf`を削除し、直下のレンダリングHTML、ページ画像ディレクトリ、`_build/`を`outputs/renders/`配下へ集約する方針で現行参照を監査した。

### 13:15 — 日報・レンダリング中間生成物の再配置完了
- `daily_report_20260729.md`と`daily_report_20260730.md`を`docs/`へ移し、`outputs/reports/`にはレイアウト検査・PDF監査ログ5件のみを残した。
- 旧成果物`outputs/FINAL_REPORT.pdf`を削除した（ワークスペース上では復元不能。現行成果物は`SUMMARY_REPORT*.pdf`）。
- 直下の3レンダリングHTML、`renders_summary_report/`、`renders_summary_report_extra/`、`_build/`を`outputs/renders/`配下へ集約した。
- `src/reporting/`のビルド、レジストリ参照、ページ画像出力を新配置へ更新した。中間HTML内に`base href="/"`を設定し、`renders/`配下からも実験画像を`outputs/`基準で解決できるようにした。
- `python3 scripts/report/check_pdf_layout.py`はシステムPythonに`pdfplumber`がなく失敗したため、プロジェクト仮想環境をPATH先頭にして同じコマンドを実行した。`SUMMARY_REPORT.pdf`は19ページ・違反0件、発展版は35ページ・違反0件だった。
- 新配置で`build_report.py`を実行し、全画像がHTTP 200で取得され、両PDFが同じページ数・違反0件で再生成されることを確認した。
- 全テスト16件が成功した（既知の未標準化Logistic Regression収束警告6件のみ）。生成された`__pycache__`も削除した。
- `outputs/`直下は`SUMMARY_REPORT*`4ファイル、`exp_*`、`discussion_draft/`、`figures/`、`renders/`、`reports/`のみに整理された。

### 13:36 — 独立Gitリポジトリ初期化の開始
- 親階層`../.git`が引き続き空ディレクトリであり、Gitメタデータや履歴を含まないことを再確認した。
- `task9/`を単独リポジトリとして初期化するため、計画書改名、公開用`.gitignore`整備、除外対象検証、初回コミットに着手した。

### 13:38 — 初回コミット前の作者情報待ち
- `docs/task9_plan.md`を`docs/execution_plan.md`へ改名し、現行コード・計画書・最終レポート内の参照を更新した。
- `task9/.gitignore`と`data_cache/.gitkeep`を作成し、実データ、OpenMLキャッシュ、展開済みコーパス、`outputs/renders/`がGit除外対象になることを`git check-ignore`で確認した。
- `task9/`で`git init`と`git add .`を実行し、145ファイルを初回コミット対象としてステージした。50 MB超の追跡対象ファイルは0件だった。
- コミット実行時にGit作者名・メールが未設定だったため停止した。公開履歴へ残る情報を推測せず、ユーザー指定を待ってリポジトリローカル設定後に再実行する。

### 14:06 — README英語版の追加
- GitHub公開後の国際的な閲覧性を高めるため、既存の日本語READMEを保持しつつ同等内容の英語セクションを追加した。
- リポジトリ名に合わせて表題を`NLP Preprocessing Evaluation`へ変更し、冒頭に英語・日本語のページ内リンクを設置した。
- 現行構成に合わせて、両言語で概要、ディレクトリ責務、主要コマンド、ソース配置規約を整合させた。
