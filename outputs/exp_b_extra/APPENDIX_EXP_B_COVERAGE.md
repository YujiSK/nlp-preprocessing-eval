# APPENDIX_EXP_B_COVERAGE — 実験B（発展）：推論時欠損と予測可能率

## 目的とデータ

自然欠損を含むOpenML Titanic v1（data_id=40945、1,309件）を用い、推論要求に欠損が含まれる場合の行削除と補完を比較した。使用特徴量は `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked` である。結果発生後に判明する `boat`・`body`、識別性の高い `name`・`ticket`、欠損率の高い `cabin`・`home.dest` は除外した。

## 評価方法

- 本文と同じ5分割StratifiedKFold（shuffle=True, random_state=42）を全条件・全モデルで共用した。
- Before（行削除）: 学習時は欠損行を除外し、推論時は7特徴量がすべて揃う行だけを予測した。欠損行は予測不能（abstain）として数えた。
- After（補完）: 数値を学習Foldの中央値、カテゴリを学習Foldの最頻値で補完し、全推論行を予測した。補完器はPipeline内でFoldごとにfitした。
- `accuracy_predicted`は実際に予測できた行だけの性能、`correct_fraction_all`は正解予測数を全推論要求数で割った運用指標であり、予測不能を不正解相当として扱う。

## 結果

Beforeの平均Coverageは0.797、予測不可率は0.203だった。Afterは全Fold・全モデルでCoverage 1.000となった。

| モデル | Before: 予測行Accuracy | Before: 全要求中の正解割合 | After: 全行Accuracy | After: 欠損行Accuracy | 差（After−Before全要求） |
|:--|--:|--:|--:|--:|--:|
| logistic_regression | 0.785 | 0.626 | 0.788 | 0.793 | +0.162 |
| linear_svc | 0.780 | 0.621 | 0.787 | 0.799 | +0.166 |
| random_forest | 0.788 | 0.628 | 0.798 | 0.799 | +0.170 |
| knn | 0.789 | 0.629 | 0.807 | 0.825 | +0.179 |

*表 C.2: 実験Bにおける行削除（Before）と補完（After）のCoverage・Accuracy比較*

<div class="figure-grid-2col" markdown="1">

![Logistic RegressionのCoverage評価](expB_coverage_lr.png)
*図 B.1: Logistic Regressionの推論時欠損とCoverage*

![Linear SVCのCoverage評価](expB_coverage_svc.png)
*図 B.2: Linear SVCの推論時欠損とCoverage*

</div>

<div class="figure-grid-2col" markdown="1">

![Random ForestのCoverage評価](expB_coverage_rf.png)
*図 B.3: Random Forestの推論時欠損とCoverage*

![k-NNのCoverage評価](expB_coverage_knn.png)
*図 B.4: k-NNの推論時欠損とCoverage*

</div>

## 解釈

Beforeの予測行Accuracyは完全行に条件付けられた値であり、Afterの全行Accuracyと評価対象が異なるため、単独で優劣を判断できない。運用上はCoverageと全要求中の正解割合を併記する必要がある。AfterのCoverage 100%は「必ず出力する」ことを意味し、欠損行上の予測が同じ信頼性を持つことを保証しない。詳細CSVには完全行・欠損行別のAfter性能も保存した。

欠損は自然発生しておりMCARとは限らない。またTitanicは歴史的な小規模データであるため、結果はCoverage指標の挙動確認であり、現行業務への性能一般化を目的としない。

## 実務的示唆

- AccuracyとともにCoverage、予測不可率、全要求中の正解割合をサービス指標として管理する。
- 欠損行を一律削除せず、予測不能が特定属性や入力チャネルへ集中していないか監査する。
- Coverage 100%を品質保証とみなさず、欠損パターン別性能や信頼度に応じた保留・人手確認も設計する。

## 再現性

- Source: https://www.openml.org/d/40945
- 実行日時: 2026-07-30T10:27:52+07:00
- Python: 3.14.4
- scikit-learn: 1.9.0
- OS: Linux-7.0.0-28-generic-x86_64-with-glibc2.43
- seed: 42
