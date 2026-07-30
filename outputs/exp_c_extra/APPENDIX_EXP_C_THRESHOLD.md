# APPENDIX_EXP_C_THRESHOLD — 実験C（発展）：内側CVによる閾値最適化

## 目的と設計

不均衡二値分類において、判定閾値を外側testで調整する評価リークを避けながら、F1最大化閾値の効果を評価した。既存実験Cと同じ2,000件・正例約7%の合成データ、補正なしC0条件、外側5-Foldを使用した。閾値効果をSMOTEやクラス重みの効果と混在させないため、対象はLogistic RegressionとLinear SVCに限定した。

各外側train内で3-Fold OOF確率を生成し、0.05〜0.95（0.01刻み）からF1が最大となる閾値を選択した。同値の場合は0.5に近い値を採用した。選択後、外側train全体でモデルを再fitし、未使用の外側testで一度だけ評価した。Linear SVCは`CalibratedClassifierCV(method='sigmoid', cv=3)`を各学習範囲内で使用した。

## 結果

| モデル | 選択閾値 mean ± std | F1: 0.5 | F1: tuned | ΔF1 | Precision: 0.5→tuned | Recall: 0.5→tuned |
|:--|--:|--:|--:|--:|--:|--:|
| logistic_regression | 0.316 ± 0.038 | 0.524 | 0.575 | +0.050 | 0.819→0.642 | 0.392→0.528 |
| linear_svc | 0.298 ± 0.029 | 0.527 | 0.584 | +0.057 | 0.860→0.638 | 0.385→0.541 |

Fold単位のF1差では、logistic_regressionは改善5 Fold・悪化0 Fold、linear_svcは改善4 Fold・悪化1 Foldだった。

![Logistic RegressionのNested閾値最適化](expC_threshold_lr.png)
*図 C.1: Logistic Regressionの内側CV閾値選択と外側Fold性能*

![Linear SVCのNested閾値最適化](expC_threshold_svc.png)
*図 C.2: 校正済みLinear SVCの内側CV閾値選択と外側Fold性能*

## 解釈上の注意

最適化対象をF1に事前固定したため、閾値低下によってRecallが上がる一方、Precisionが下がる可能性がある。Average PrecisionとROC-AUCは確率ランキングに依存する閾値非依存指標であり、同一外側Fold・同一モデルではdefault/tuned間で変化しない。

選択閾値と外側性能にはFold間変動がある。ここで得た閾値を普遍的な固定値とは解釈せず、運用時のクラス比率・誤検知/見逃しコスト・確率校正の変化に応じて学習データ内だけで再選択する必要がある。外側Foldは閾値選択にも校正にも使用していない。

## 実務的示唆

- 閾値最適化の目的関数を事前固定し、外側testや本番結果で後から選び直さない。
- Linear SVCのスコアを確率として使う場合、学習範囲内だけで校正する。
- Average Precisionなどのランキング品質と、Recall・Precision・F1など運用閾値依存指標を分けて監視する。

## 再現性

- 実行日時: 2026-07-30T11:00:08+07:00
- Python: 3.14.4
- scikit-learn: 1.9.0
- OS: Linux-7.0.0-28-generic-x86_64-with-glibc2.43
- outer folds: 5 / inner folds: 3 / seed: 42
- condition: C0（補正なし） / threshold objective: F1
