# 課題9：データ特性および前処理ステップ別モデル比較実験計画書

## 1. 目的
単一のベンチマークデータ（トイ・データセット）による評価にとどまらず、データの性質（欠損率・不均衡性・テキストデータ等）や前処理ステップ（標準化・欠損補完・クレンジング・辞書選定）の違いが、機械学習モデルの挙動や精度に与える影響を定量的に検証・考察する。

あわせて、単発の `train_test_split` による一点評価では見落とされやすい以下4つの検証軸を全実験に共通して組み込み、評価の再現性と厳密さを高める。

1. **評価の信頼性向上**: 5分割 `StratifiedKFold` 交差検証による平均精度・標準偏差の算出
2. **モデルの説明性**: `coef_`（Logistic Regression / Linear SVC）、`feature_importances_`（Random Forest）の抽出・可視化
3. **処理コストと精度のトレードオフ**: 学習時間・推論時間・（実験Dのみ）形態素解析処理時間の計測
4. **ハイパーパラメータチューニング**: 主要パラメータ変化に対する過学習/未学習の挙動観察

さらに、Before/After比較の因果解釈を成立させるため、以下を設計上の前提とする。

* **Before/Afterで変更する要因は原則1つに限定する**。やむを得ず複数要因が同時に変わる場合は、個別手法の効果ではなく「前処理パイプライン全体の比較」として扱うことを明記する。
* **学習型の前処理（標準化・補完・エンコーディング・TF-IDF・サンプリング等）は必ずPipeline内に置き、各交差検証Foldの学習データのみでfitする**。データ全体に対して事前にfitしてはならない。
* **すべての実験は「必須（本日中に完了させる範囲）」と「発展（時間が余った場合に追加する範囲）」を区別して計画する**。全項目の完了を前提にすると、未完了時にレポートの整合性が崩れるため。

---

## 2. 実験構成マトリックス（4つのデータセット × 前処理Before/After × 4モデル）

### 適用モデル（全実験共通）
1. **ロジスティック回帰 (Logistic Regression)**
2. **Linear SVC**
3. **ランダムフォレスト (Random Forest)**
4. **k-近傍法 (K-Nearest Neighbors)**

### 実験マトリックス一覧

| 実験 | データセット | データ特性 | Before | After |
| :--- | :--- | :--- | :--- | :--- |
| **A** | Breast Cancer（569件） | 欠損なし／数値データ | 未標準化 | `StandardScaler` 標準化 |
| **B** | 完全データに人工欠損を注入した合成データ（数値＋カテゴリ変数）※3.2参照 | 欠損値／カテゴリ変数混在 | 欠損行の削除（One-Hotは共通適用） | 欠損値の補完（One-Hotは共通適用） |
| **C** | クレジットカード不正検知風データ | 不均衡（正例5〜10%） | 補正なし（4モデル共通） | サンプリング調整（4モデル共通、Fold内限定）※`class_weight='balanced'`はk-NNに適用不可のため3モデルのみの追加実験（3.3参照） |
| **D** | 日本語テキスト（データセットは事前に確定。3.4参照） | 非構造化テキスト | クレンジングなし＋IPA辞書＋TF-IDF | `neologdn`クレンジング＋NEologd/Sudachi Mode C＋TF-IDF（パイプライン全体としての比較） |

---

## 3. 各実験の検証内容とBefore/After評価軸

### 【実験A】標準ベンチマークデータ（ベースライン）
* **データセット**: Breast Cancer（569件 / 欠損なし / 数値データ）
* **前処理**:
  * **Before**: 未標準化データ
  * **After**: `StandardScaler` による標準化（Pipeline内でFoldごとにfit）
* **検証ポイント**: 距離ベースモデル（k-NN）や線形モデル（Logistic Regression / Linear SVC）と決定木系（RF）における、特徴量スケールと正則化の関係への依存度の違い。RFは標準化の影響を受けにくいと想定されるが、結論は実測値のみに基づいて記述する。
* **交差検証**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` を全モデル・全条件で共通の分割として使い回し、Before/Afterそれぞれについて4モデル×平均Accuracy・標準偏差を算出する。単一分割との差分も確認する。
* **説明性（必須）**: Logistic Regression / Linear SVCについて、正の係数上位10・負の係数上位10・絶対値上位10を分けて表示する。未標準化係数は特徴量の単位・スケールの影響を受けるため、Before/Afterの係数ランキングをそのまま「重要特徴量の変化」として比較しない。係数図はモデル内部の判断根拠の提示にとどめる。
* **説明性（発展）**: 検証Fold上のPermutation Importanceを算出し、Before/Afterおよびモデル間で比較可能な重要度として併記する。乳がんデータには相関の強い特徴量が含まれるため、重要度を独立した因果的寄与として解釈しない。
* **コスト計測**: 4モデル×Before/Afterで`fit()`時間・`predict()`時間を計測し、標準化の前処理コスト（`fit_transform`時間）も含めて表に整理する（詳細な計測区分は4.6参照）。Logistic Regression / Linear SVCの収束警告・`max_iter`到達有無も記録する。
* **チューニング**: k-NNの`n_neighbors`（1, 3, 5, 10, 20）、LogisticRegression/Linear SVCの`C`（0.01, 0.1, 1, 10, 100）を変化させ、train/CV精度の乖離幅から過学習/未学習の傾向を確認する。この結果は挙動観察として扱い、最終性能評価には用いない（4.3参照）。

### 【実験B】欠損値・ノイズ含有データ
* **データセット**: 完全データ（欠損なし）に人工的に欠損を注入した合成データ（数値＋カテゴリ変数）
* **前処理**:
  * **Before**: 学習Fold内で欠損行を削除
  * **After**: 学習Fold内で欠損値を補完（数値：中央値、カテゴリ：最頻値）
  * **One-Hot Encodingは両条件で共通**とし、欠損処理（削除 vs 補完）のみを比較対象とする。カテゴリ変数を未エンコードのまま4モデルへ入力することはできないため。
* **主実験の設計（必須）**:
  1. 元データは欠損のない完全データを用意する
  2. 外側CV（4.2の共通分割）でtrain/testに分割する
  3. **外側trainのみ**に人工欠損を注入する
  4. 同一の欠損マスクをBefore/Afterで共有する
  5. 外側testは欠損のない同一サンプルを使用する（Before/Afterで評価対象が一致するため、AccuracyやF1を単純比較できる）
  6. Beforeはtrainの欠損行を削除、Afterはtrainの欠損値を補完する
* **副実験（発展）**: 推論時にも欠損が生じる実データ（Titanic等）を用いる場合は、Beforeを「欠損入力に対して予測不能」として扱い、予測可能率（Coverage）、予測可能サンプルに限定した性能、全サンプルに対する実用上の性能を分けて報告する。
* **検証ポイント**: 学習データ保持率の変化と、欠損処理方法が各種モデルの予測精度に与える影響。
* **交差検証**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`。Beforeは学習Fold内でレコード数が減少するため、各Foldの学習サンプル数も併記し、分散への影響を考察する。
* **前処理構成（ColumnTransformer）**: 数値列とカテゴリ列を分離し、`OneHotEncoder(handle_unknown="ignore", min_frequency=2)`を用いる。`handle_unknown="ignore"`により、学習Foldに存在せず検証Foldにのみ現れるカテゴリでの例外を防ぐ。`min_frequency`により低頻度カテゴリによる次元爆発を抑える。
* **エッジケース確認（必須）**: 学習Foldでは欠損がないが検証Foldで初めて欠損する列、学習Fold内で全欠損の列、1件しかないカテゴリ、高カーディナリティ列、欠損が目的変数と強く関連する場合、目的変数自体の欠損（除外し件数を報告）、数値列への文字列混入。
* **説明性**: Random Forestの`feature_importances_`上位特徴量に補完由来の列（欠損フラグ、One-Hotカテゴリ）が含まれるかを確認し、Logistic Regressionの`coef_`と比較する。
* **コスト計測**: 欠損処理（行削除 vs 補完＋エンコーディング）自体の処理時間、および学習・推論時間をBefore/Afterで比較する。
* **チューニング**: Random Forestの`max_depth`（3, 5, 10, None）を変化させ、欠損補完データにおける過学習挙動（train精度は上昇するがCV精度が頭打ちになる境界）を確認する。

### 【実験C】不均衡（Imbalanced）データ
* **データセット**: クレジットカード不正検知風データ（正例 5〜10% : 負例 90〜95%）
* **前処理・条件設計**:
  * **C0（Before）**: 補正なし、4モデル共通
  * **C1（After・必須）**: サンプリング調整（Over/Undersampling）、4モデル共通。サンプリングは`imblearn.pipeline.Pipeline`内に置き、**学習Foldのみ**に適用する。データ全体へ適用してからCVすると検証Foldの情報が学習に混入するため禁止。SMOTEを用いる場合は数値特徴量のスケーリング後に適用する。
  * **C2（発展・追加実験）**: `class_weight='balanced'`。Logistic Regression / Linear SVC / Random Forestの3モデルのみに適用可能で、`KNeighborsClassifier`には`class_weight`パラメータが存在しないため4モデル共通条件にできない。C0/C1とは別枠の追加実験として扱う。
* **検証ポイント**: 見かけの正解率（Accuracy）の罠と、Recall・Precision・F1・Balanced Accuracy・混同行列の改善度。
* **交差検証**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`。Fold間でのRecall・PR-AUC・F1の平均・標準偏差を算出する（不均衡データはFoldごとの正例数の偏りが精度分散に直結するため必須）。
* **指標（主指標をAverage Precision / PR-AUCに固定）**: Average Precision（PR-AUC）を主指標とし、Recall、Precision、F1、Balanced Accuracy、MCC、ROC-AUC、混同行列、正例数・負例数（Foldごと）を副指標として記録する。Accuracyは参考値にとどめる。
* **閾値（発展）**: `predict()`の既定閾値のみで評価すると、サンプリング/クラス重みの効果と判定閾値の効果が混在する。時間があれば内側CVのみで閾値を選択し、外側Foldで評価する（外側testで閾値を最適化しない）。Linear SVCで確率値が必要な場合は`CalibratedClassifierCV`を内側CVで使用する。
* **説明性**: Logistic Regressionの`coef_`、Random Forestの`feature_importances_`をC0/C1で比較し、補正の有無で重要視される特徴量が変化するかを確認する。
* **コスト計測**: サンプリング処理時間を追加し、学習・推論時間とあわせて計測する。
* **チューニング**: Logistic Regression/Linear SVCの`C`を変化させ、C0とC1でRecall/Precisionのトレードオフがどう変化するかを検証する。

### 【実験D】日本語テキスト分類データ
* **データセット**: 事前に確定する（例：livedoor News Corpus。多クラス分類として計画と整合しやすい）。独自レビューデータを用いる場合はライセンス・ラベル品質・クラス比率を記録する。
* **前処理**:
  * **Before**: クレンジングなし ＋ 標準辞書（IPA辞書）での形態素解析 ＋ TF-IDF
  * **After**: `neologdn` クレンジング ＋ 拡張辞書（NEologd / Sudachi Mode C）＋ TF-IDF
  * クレンジング有無・解析器（MeCab/Sudachi）・辞書（IPAdic/NEologd/Sudachi辞書）・分割モードが同時に変わるため、**個別手法の因果的効果ではなく、前処理パイプライン全体（Before/After）の比較として解釈する**ことを明記する。
* **アブレーション（発展）**: 時間が許す場合、以下4条件でクレンジングと解析器の効果を分離する。

  | 条件 | クレンジング | 形態素解析 |
  | :--- | :--- | :--- |
  | D0 | なし | MeCab＋IPAdic |
  | D1 | NFKC＋neologdn | MeCab＋IPAdic |
  | D2 | なし | Sudachi core＋Mode C |
  | D3 | NFKC＋neologdn | Sudachi core＋Mode C |

* **検証ポイント**: 表記揺れ吸収による語彙数（特徴量次元数）の圧縮効果と、分類精度（F1-score）の向上幅。
* **交差検証**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`。TF-IDFベクトライザは各Foldの学習データのみでfitし、Pipeline化してリークを防止する。
* **リーク確認（必須）**:
  * メタデータリーク: ファイルパス中のカテゴリ名、カテゴリ別ディレクトリ名、URLドメイン、記事末尾の媒体名、固定ヘッダー等を特徴量に含めない。
  * 重複・類似記事: 正規化前後のテキストハッシュによる重複除去を行う。元記事ID・ユーザーID・スレッドIDがある場合はGroup分割（`StratifiedGroupKFold`等）を検討する。削除件数・割合を記録する。
  * 時系列・グループ構造: 記事が時期やイベントで偏っている場合、通常のランダムCVは評価を楽観的にする可能性がある。データの性質に応じて時系列分割やGroup分割の要否を判断する。
* **空文書対策**: クレンジング・品詞除去後の空文字列、1トークンのみの文書、記号/URL/絵文字のみの文書、`min_df`により全語彙が消えるFoldを確認する。空文書数、トークン数の分布（最小・中央値・最大）、語彙数、除外件数を記録する。
* **TF-IDFへの入力**: 形態素解析済み文字列を渡す際、TF-IDF側で英語用トークン化を再適用しないよう`tokenizer=str.split, token_pattern=None, lowercase=False`等を指定する。
* **説明性**: TF-IDF語彙はFoldごとにfitするため、Fold間で列インデックスが一致しない。`get_feature_names_out()`で語彙名に戻し、語彙名単位で選出回数・係数を集計するか、CV終了後に全学習データで説明専用モデルを再学習する（性能評価には使わず、記述的分析としてのみ提示する）。クラスごとの上位寄与語を正負に分けて抽出する。
* **コスト計測**: 形態素解析処理時間（Before: IPA辞書 / After: NEologd・Sudachi）、TF-IDFベクトル化時間、学習・推論時間を分けて計測する。形態素解析器はループ内で毎回初期化せず、1回生成して再利用する。
* **チューニング**: Logistic Regression/Linear SVCの`C`、Random Forestの`max_depth`を変化させ、高次元疎ベクトルにおける過学習挙動を確認する。k-NNの距離指標（`euclidean`/`cosine`）は比較要因を増やさないよう事前に固定する。

---

## 4. 前処理・評価における共通ルール

### 4.1 データリーク防止
以下は学習データ全体に対して事前にfitせず、必ずPipeline / ColumnTransformer / imblearn Pipelineに含め、各交差検証Foldの学習データのみでfitする。

* `StandardScaler` / `SimpleImputer` / `OneHotEncoder`
* `TfidfVectorizer`（語彙・IDFの学習）
* 特徴量選択、サンプリング（SMOTE等）
* 閾値チューニング、`CalibratedClassifierCV`

一方、辞書固定の形態素解析や決定論的なクレンジング（NFKC正規化、neologdn等）はコーパス全体の統計を学習しないため、事前計算・キャッシュしてよい。

### 4.2 交差検証分割の共通化
外側CVの分割（`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`）は全モデル・全条件で同一のものを生成し、使い回す。条件ごとに異なる乱数分割を用いると、前処理差ではなくFoldの偶然性が結果に混入する。`shuffle=True`を明示し、データが目的変数・時系列・カテゴリ順に並んでいる場合のFold偏りを防ぐ。

### 4.3 ハイパーパラメータ選択の扱い（必須／発展の切り分け）
5-Fold CVでパラメータを比較し、その最高値をそのままモデル性能として報告すると、評価値が楽観的になる（同一データをチューニングと評価の両方に使用するため）。

* **必須**: デフォルトパラメータによるCV結果を主結果とする。Validation Curve（パラメータ変化に対するtrain/CV精度）は過学習/未学習の挙動観察として扱い、独立した一般化性能の推定値としては扱わない。
* **発展**: 代表モデルに限定してNested CV（外側5-Fold＝性能評価、内側3-Fold＝パラメータ選択）を実施する、または独立した最終テストセットを保持しCVはチューニング専用にする。

### 4.4 主指標の事前固定
モデル実行後に都合の良い指標を選ばないよう、実験ごとに主指標をあらかじめ固定する。

| 実験 | 主指標 | 副指標 |
| :--- | :--- | :--- |
| A | Accuracy | macro-F1、Balanced Accuracy |
| B | Accuracy（外側testは欠損なしで統一） | 学習データ保持率、macro-F1 |
| C | Average Precision（PR-AUC） | Recall、Precision、F1、Balanced Accuracy、MCC、ROC-AUC |
| D | macro-F1 | weighted-F1、Accuracy、クラス別Recall |

多クラス分類（実験D）ではmacro-F1を主指標とする。少数クラスの性能低下を隠しにくいため。

### 4.5 説明性評価の注意点
* 線形モデルの係数（`coef_`）は正・負・絶対値の上位を分けて提示し、特徴量スケールや相関構造の影響を受けることを明記する。
* Random Forestの`feature_importances_`（不純度ベース）は高カーディナリティ・連続値特徴量に偏りやすく、過学習モデルでは未知データ上の重要性を反映しない場合がある。発展項目として、検証Fold上のPermutation Importance（全モデル共通で適用可能）を併記する。相関の強い特徴量では両方の重要度が低く見える場合があることに留意する。
* Fold間で平均する場合は、単純な数値平均だけでなく標準偏差・上位選出回数も報告する。

### 4.6 処理コスト計測の定義
二重計上を避けるため、計測対象を以下のように分離して固定する。

1. データ読み込み時間
2. 決定論的前処理時間（NFKC・neologdn・形態素解析）
3. 学習型前処理のfit時間（Imputer・Scaler・One-Hot・TF-IDF）
4. 学習型前処理のtransform時間
5. モデルのfit時間
6. 純粋なpredict（またはdecision_function）時間
7. End-to-Endのfit時間
8. End-to-Endの推論時間

計測条件: `time.perf_counter()`を使用し、初回実行はウォームアップとして分離、3〜5回計測し平均・中央値を記録する。バッチ件数を固定し、k-NNは1件当たり推論時間とバッチ推論時間を分ける。計測は`n_jobs=1`を基本とし、CPU・RAM・OS・Pythonバージョンを記録する。

---

## 5. 前処理Before/After × 評価指標 比較設計

各実験で以下の指標をBefore/After・4モデルの組み合わせごとに記録する（主指標は4.4参照）。

| 指標カテゴリ | 具体的指標 | 記録方法 |
| :--- | :--- | :--- |
| **予測性能** | 4.4の主指標・副指標 | CV平均値 ± 標準偏差、単一split値と併記 |
| **信頼性（CV）** | 5-Fold各指標の平均・標準偏差、Fold単位のBefore→After差分（ペア差） | Foldごとの値を表出力し、改善Fold数・悪化Fold数を明記 |
| **説明性** | `coef_`（正/負/絶対値上位10）、`feature_importances_`上位10、（発展）Permutation Importance | 棒グラフ（Before/After並列表示） |
| **処理コスト** | 4.6の8区分の時間計測 | 秒単位で計測し表にまとめる |
| **ハイパーパラメータ挙動** | 対象パラメータ変化に対するtrain/CV精度カーブ（4.3で「挙動観察」と明記） | 折れ線グラフ（過学習/未学習領域を明示） |

Fold単位のペア差（`after_fold - before_fold`）は、平均差・差の標準偏差・改善/悪化Fold数として報告する。正式な有意差検定は行わず、事実としての記述にとどめる（CV Foldは独立標本ではないため、通常の検定は不適切）。

---

## 6. 実験成果物（Outputs）の出力・保存規約

実験結果（画像・数値データ）の散逸を防ぎ、レポート執筆時の参照および再実行時の再現性を確保するため、以下のディレクトリ構造および命名規則に従って自動保存する。

### 6.1 ディレクトリ構成

```
task9/
├── README.md
├── configs/                 # 実験条件（データセット・前処理・パラメータ候補）の設定ファイル
├── docs/execution_plan.md       # 本計画書
├── assets/styles/           # PDFレポート用CSS
├── scripts/
│   ├── core/                # 基本実験CLI
│   ├── extra/               # 発展実験・発展集計CLI
│   └── report/              # レポート構築・検査CLI
├── src/
│   ├── experiments/         # 実験・評価・前処理の再利用モジュール
│   └── reporting/           # PDF構築・レンダリング・検査モジュール
├── tests/                   # 前処理の単体テスト（エッジケース確認）
└── outputs/                 # 出力成果物ルート
    ├── exp_a/                # 実験A成果物
    ├── exp_b/                # 実験B成果物
    ├── exp_c/                # 実験C成果物
    └── exp_d/                # 実験D成果物
```

上記は2026-07-30のリファクタリング後の構成と一致している。

### 6.2 ファイル命名規則
各実験フォルダ内には、以下の命名規則に従って画像および数値化データを出力する。

| 対象 | ファイル命名規則の例 | 形式 | 役割 |
| :--- | :--- | :--- | :--- |
| **精度・CV比較表** | `exp{X}_metrics_summary.csv` | CSV / JSON | 各モデル・条件のCV平均、標準偏差、Fold単位ペア差、処理時間を記録 |
| **評価指標グラフ** | `exp{X}_cv_score_bar.png` | PNG（300dpi） | CV平均精度・標準偏差（エラーバー付）の比較棒グラフ |
| **混同行列** | `exp{X}_cm_{model_name}.png` | PNG（300dpi） | 混同行列のヒートマップ（不均衡データやテキスト分類用） |
| **特徴量重要度** | `exp{X}_feature_importance.png` | PNG（300dpi） | `coef_` / `feature_importances_` の上位特徴量比較グラフ |
| **パラメータ挙動** | `exp{X}_param_tuning.png` | PNG（300dpi） | ハイパーパラメータ変化に伴うTrain/CV精度カーブ |

※ `{X}` には A, B, C, D の実験識別子が入る。

### 6.3 運用ルール
* 出力先ディレクトリはスクリプト実行時に`os.makedirs("outputs/exp_a", exist_ok=True)`等で自動生成し、手動作成には依存しない。
* 数値結果は画像化前に必ずCSV/JSON（Long形式：experiment, condition, model, fold, metric, valueの列構成を推奨）として保存し、グラフ再描画が学習の再実行なしに行えるようにする。
* 同一ファイル名での再実行時は上書きを許容し、実行日時等によるバージョン管理は行わない（単一実施回のため）。

---

## 7. 再現性・環境固定

以下を実行環境情報として記録し、成果物（`outputs/`配下または別ファイル）に含める。

* Python、OS、CPU、RAMのバージョン・仕様
* scikit-learn、pandas、numpy、imbalanced-learn、neologdn、SudachiPy、Sudachi辞書、MeCabラッパー、MeCab辞書の各バージョン（`requirements.txt`等で固定）
* 使用データセット名・取得元・バージョン
* 辞書名・辞書バージョン・辞書パス
* 全乱数シード、実行日時
* 前処理バージョンタグ（例：`PREPROCESSOR_VERSION = "task9-ja-v1"`）

実験Dのテキストデータは、`raw_text`（原文）を上書きせず、`normalized_text`（クレンジング後）、`tokenized_text`（形態素解析後）として別列で保持する。前処理変更後も同じ原文から再実行できるようにするため。

---

## 8. 実施チェックリスト

作業を進める都度、該当項目のチェックボックスを `[ ]` から `[x]` に更新し、本ファイルを上書きすること。進捗はこのチェックリストの状態をもって管理する。**必須**を先に完了させ、**発展**は時間が余った場合のみ着手する。

### 共通準備（必須）
- [x] `outputs/exp_a〜exp_d`のディレクトリ構成・命名規則（第6章）に沿った出力用ヘルパー関数を実装する（`src/utils.py`）
- [x] 全モデル・全条件で共通利用する外側CV分割（`StratifiedKFold(shuffle=True, random_state=42)`）を1回生成し再利用する仕組みを実装する（`src/utils.py: get_outer_splits`）
- [x] Before/After・5-Fold CV・コスト計測（fit/predict秒）・特徴量重要度抽出を行う共通評価関数を実装する（`src/evaluation.py`, `src/explainability.py`）
- [x] 学習型前処理をPipeline / ColumnTransformer / imblearn Pipelineに統一し、Fold内fitを徹底する（`evaluate_pipeline_cv`はpipeline_factoryをFold毎にfit）

### 実験A（Breast Cancer・必須）
- [x] Before/After（標準化なし/あり）でのCV平均・標準偏差を算出する（`scripts/core/run_exp_a.py` → `outputs/exp_a/expA_metrics_summary_agg.csv`）
- [x] `coef_`（正/負/絶対値上位10）と`feature_importances_`を可視化する（`outputs/exp_a/expA_feature_importance.png`）
- [x] 前処理・学習・推論時間を計測する（`outputs/exp_a/expA_metrics_summary.csv`のfit_seconds/predict_seconds列。8区分の厳密分離は未実施）
- [x] `n_neighbors` / `C` を変化させ過学習/未学習の挙動を確認する（`outputs/exp_a/expA_param_tuning.png`、挙動観察として明記）

### 実験A（発展）
- [x] Permutation Importanceを算出しBefore/After・モデル間で比較する（`scripts/extra/run_exp_a_permutation.py`。検証Fold上・30反復で算出し、既存本文から分離した`outputs/exp_a_extra/`にCSV・比較図・Appendix原稿を保存）

### 実験B（欠損値データ・必須）
- [x] 完全データへの人工欠損注入（外側train限定・Before/Afterで同一マスク）を実装する（`scripts/core/run_exp_b.py: inject_missingness`）
- [x] Before/After（削除/補完、One-Hotは共通）でのCV平均・標準偏差を算出する（`outputs/exp_b/expB_metrics_summary_agg.csv`。After優位、全モデルで一貫して改善）
- [x] Foldごとの学習サンプル数を記録し分散への影響を確認する（`outputs/exp_b/expB_fold_sample_sizes.csv`。Beforeは640件→115〜137件に減少）
- [x] エッジケース（全欠損列、低頻度カテゴリ、目的変数欠損等）を確認・記録する（`outputs/exp_b/expB_edge_cases.csv`。今回の設定では該当なし）
- [x] `coef_` / `feature_importances_` を可視化する（`outputs/exp_b/expB_feature_importance.png`）
- [x] 欠損処理・学習・推論時間を計測する（`outputs/exp_b/expB_metrics_summary.csv`）
- [x] `max_depth` を変化させ過学習挙動を確認する（`outputs/exp_b/expB_param_tuning.png`）

### 実験B（発展）
- [x] 推論時欠損を含む実データ（Titanic等）で予測可能率（Coverage）を副実験として評価する（`scripts/extra/run_exp_b_coverage.py`。OpenML Titanic v1の自然欠損を使用し、`outputs/exp_b_extra/`にFold別・集約CSV、欠損プロファイル、比較図、Appendix原稿を保存）

### 実験C（不均衡データ・必須）
- [x] C0（補正なし）/C1（サンプリング調整、Fold内限定）でのCV平均・標準偏差（PR-AUC中心）を算出する（`outputs/exp_c/expC_metrics_summary_agg.csv`）
- [x] 混同行列を作成する（`outputs/exp_c/expC_cm_logistic_regression.png`、C0/C1のOOF混同行列）
- [x] `coef_` / `feature_importances_` を可視化する（`outputs/exp_c/expC_coefficients.csv`, `expC_feature_importance.csv`）
- [x] サンプリング処理時間・学習・推論時間を計測する（`outputs/exp_c/expC_metrics_summary.csv`のfit_seconds列。SMOTE込みのEnd-to-End時間として計測、サンプリング単独の分離計測は未実施）
- [x] `C` を変化させC0/C1間でRecall/Precisionのトレードオフを確認する（`outputs/exp_c/expC_precision_recall_tradeoff.png`（デフォルトC=1.0比較）、`outputs/exp_c/expC_param_tuning.png`（Cグリッドでの挙動観察））

### 実験C（発展）
- [x] C2（`class_weight='balanced'`、3モデル限定）を追加実験として評価する（`outputs/exp_c/expC_metrics_summary_agg.csv`にcondition=c2として含む）
- [x] 内側CVのみで閾値を選択し外側Foldで評価する（`scripts/extra/run_exp_c_threshold.py`。C0条件のLogistic Regressionと`CalibratedClassifierCV`使用Linear SVCをNested CVで評価し、`outputs/exp_c_extra/`にInner探索・Outer評価CSV、比較図、Appendix原稿を保存）

### 実験D（日本語テキスト分類・必須）
- [x] 使用データセットを確定する（livedoor News Corpus、9クラス、7,367記事。`data_cache/text/`にダウンロード・展開済み。ライセンス: CC BY-ND、社内分析用途）
- [x] メタデータリーク（カテゴリ名・URL・媒体名等）がないことを確認する（`outputs/exp_d/expD_metadata_leak_check.csv`。**実際に検出**: smaxカテゴリの記事861/870件に自社媒体名「S-MAX/smax」を含む本文末尾の関連リンク一覧が含まれていたため、`src/experiments/preprocessing.py`の`_strip_footer`で「■関連リンク」「■関連記事」以降を除去。修正後は861→1件に低減）
- [x] 重複・類似記事のハッシュベース除去を行い削除件数を記録する（7,367件中6件を完全一致で除去、`load_and_prepare`内で実施）
- [x] Before/After（IPA辞書/NEologd・Sudachi＋neologdn、パイプライン全体比較）でのCV平均・標準偏差を算出する（`outputs/exp_d/expD_metrics_summary_agg.csv`。macro-F1でBeforeがAfterよりわずかに上回るか同等、明確な改善は見られなかった）
- [x] TF-IDFをPipeline化しリークを防止する（`TfidfVectorizer`をPipeline内に配置しFold毎にfit）
- [x] 空文書・低トークン文書を確認し件数を記録する（`outputs/exp_d/expD_token_stats.csv`。空文書は0件、最小トークン数11）
- [x] `coef_`上位語彙（正/負）をBefore/Afterで比較する（`outputs/exp_d/expD_coefficients.csv`。sports-watch/movie-enter/it-life-hackの3クラスを代表として抽出）
- [x] 形態素解析時間・ベクトル化時間・学習・推論時間を計測する（`outputs/exp_d/expD_preprocessing_time.csv`：IPA約6秒 vs neologdn+Sudachi約19秒（全7,361件）。学習/推論時間は`expD_metrics_summary.csv`）
- [x] `C` / `max_depth` を変化させ過学習挙動を確認する（`outputs/exp_d/expD_param_tuning.png`）

### 実験D（発展）
- [x] D0〜D3の4条件アブレーションでクレンジング/解析器の効果を分離する（`scripts/extra/run_exp_d_ablation.py`。同一5-Fold・4モデルで2×2要因を評価し、`outputs/exp_d_extra/`にFold指標・要因差・語彙数・処理時間CSV、比較図、Appendix原稿を保存）
- [x] 記事の時系列・グループ構造を確認し、必要であれば時系列分割/Group分割で追加評価する（`scripts/extra/audit_exp_d_structure.py`、`outputs/exp_d_extra/expD_temporal_group_audit.csv`。URL・filename重複および明示的group IDがないためGroup分割は不要と判断。固定コーパス内の要因比較では共通StratifiedKFoldを維持し、将来時点への一般化は別途時系列評価が必要と記録）

### 集約・執筆
- [x] 全実験の指標（CV平均/標準偏差、Fold単位ペア差、特徴量重要度、コスト、パラメータ挙動）を表・グラフに集約する（必須課題は`outputs/SUMMARY_REPORT.md`第3〜6章、発展課題は`outputs/SUMMARY_REPORT_extra.md`付録C〜Fへ統合）
- [x] 比較考察を執筆する（`outputs/discussion_draft/exp_a〜exp_d.md`および`exp_a_extra〜exp_d_extra.md`を作成し、発展実験の実務的示唆を各Appendixへ反映）
- [x] 事実ベース・客観的表記で推敲する（`SUMMARY_REPORT.md`および`SUMMARY_REPORT_extra.md`で実測値を直接引用）
- [x] 実行環境・バージョン情報（第7章）をレポートに記載する（必須実験は`outputs/exp_a〜d/exp_*_environment.json`、発展実験は各Appendixの再現性節に記録）
- [x] レポートをPDF出力する（`outputs/SUMMARY_REPORT.pdf` 18ページ、`outputs/SUMMARY_REPORT_extra.pdf` 34ページ。Chrome headless経由で生成し、両方とも自動検査違反0件）
- [x] 引き継ぎ整理・日報作成を行う（現行保存先: `docs/reports/daily_report_*.md`）

---

## 9. レポート執筆における基本方針
* **客観的事実の徹底**: 誇張表現（「高精度を達成」「完璧な」等）を排除し、数値・グラフに基づいた事実のみを記述する。
* **トイデータと実データのギャップ記述**: ベンチマークデータでの高精度に頼らず、前処理の有無がもたらすトレードオフ（計算コスト、過学習リスク、データ保持率等）を論理的に考察する。
* **単一指標への依存回避**: Accuracyのみで優劣を判断せず、CVの標準偏差、主指標（4.4）、処理コスト、特徴量重要度の変化を併記した上で総合的に記述する。
* **再現性の担保**: 乱数シード（`random_state`）、CVの分割方法、前処理パイプラインの構成を明記し、他者が同一条件で再実行できる記述にする。

### レポート冒頭に明記する注意事項（そのまま使用可）

> 本実験では、データリークを防止するため、欠損補完、標準化、カテゴリ変数エンコーディング、TF-IDF語彙およびIDFの学習、サンプリング処理をPipeline内に実装し、各交差検証Foldの学習データのみでfitする。
>
> Before/After比較では、原則として評価対象データおよび交差検証分割を共通化し、変更対象以外の前処理条件を固定する。複数の処理を同時に変更する場合は、個別手法の因果的効果ではなく、前処理パイプライン全体の比較として解釈する。
>
> ハイパーパラメータ調整後の性能を最終評価値として報告する場合は、外側CVを性能評価、内側CVをパラメータ選択に用いるNested Cross-Validationを適用する。計算時間の制約により通常のValidation Curveのみを用いる場合、その結果は過学習・未学習の傾向観察として扱い、独立した一般化性能推定とは区別する。
>
> 不均衡データについてはAccuracyを主指標とせず、Average Precision、Recall、Precision、F1、Balanced Accuracyおよび混同行列を併記する。サンプリング処理は学習Fold内のみで実施し、評価データのクラス分布は変更しない。
>
> モデル説明では、線形モデルの係数が特徴量スケールや相関構造の影響を受けること、Random Forestの不純度ベース重要度が高カーディナリティ特徴量へ偏る可能性があることを明記する。必要に応じて検証Fold上のPermutation Importanceを併用する。
>
> 再現性確保のため、データセットの取得元・バージョン、Pythonおよび各ライブラリのバージョン、形態素解析器・辞書の種類とバージョン、全乱数シード、実行環境を記録する。原文データは保持し、前処理済みデータは再生成可能な派生データとして管理する。
