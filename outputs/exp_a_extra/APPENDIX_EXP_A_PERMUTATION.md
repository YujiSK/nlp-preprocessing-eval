# Appendix A — 実験A（発展）：Permutation Importance

## 方法

- Breast Cancerデータ（569件、30特徴量）を使用した。
- 本文と同じ `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` の分割を、全条件・全モデルで共用した。
- 各モデルを学習Foldのみでfitし、未使用の検証Fold上でAccuracyを評価した。
- 各特徴量を検証Fold内で30回並べ替え、Accuracy低下量をPermutation Importanceとした。
- StandardScalerはAfter条件のみPipeline内でfitした。並べ替えはPipelineへの入力列に対して行った。

## 結果

| モデル | Before上位特徴量（平均PI） | After上位特徴量（平均PI） | 最大のBefore→After差 |
|:--|:--|:--|:--|
| logistic_regression | worst area (0.2776) | worst texture (0.0172) | worst area (-0.2701) |
| linear_svc | worst area (0.3447) | worst texture (0.0379) | worst area (-0.3079) |
| random_forest | worst area (0.0116) | worst area (0.0116) | mean concave points (+0.0006) |
| knn | worst area (0.3814) | worst texture (0.0107) | worst area (-0.3769) |

![Logistic RegressionのPermutation Importance](expA_permutation_lr.png)
*図 A.1: Logistic Regressionの検証Fold上Permutation Importance*

![Linear SVCのPermutation Importance](expA_permutation_svc.png)
*図 A.2: Linear SVCの検証Fold上Permutation Importance*

![Random ForestのPermutation Importance](expA_permutation_rf.png)
*図 A.3: Random Forestの検証Fold上Permutation Importance*

![k-NNのPermutation Importance](expA_permutation_knn.png)
*図 A.4: k-NNの検証Fold上Permutation Importance*

## 解釈上の注意

Permutation Importanceは未使用の検証Fold上で算出したため、係数の絶対値やRandom Forestの不純度ベース重要度よりモデル間比較に適する。一方、本データには相関の強い特徴量が複数あり、代替可能な特徴量を一つだけ並べ替えても予測性能が大きく低下しない場合がある。このため、値を各特徴量の独立した因果的寄与とは解釈しない。小さな負値は、有限標本と並べ替えによる変動の範囲で生じ得る。

未標準化Logistic Regressionは5 Foldすべてで `max_iter=2000` に達する収束警告が発生した。その条件の重要度は未収束モデルに基づく記述値であり、Afterとの差に標準化だけでなく最適化の収束状態も反映され得る。既存の実験条件を維持するため、発展実験のみで反復回数やソルバーは変更していない。

## 実務的示唆

- スケール依存モデルでは標準化と収束状況を説明性評価より先に品質ゲートとして確認する。
- 異なるモデルの判断感度を比較する場合、生の係数絶対値より検証データ上のPermutation Importanceを優先する。
- 相関特徴量の重要度を独立した因果寄与や削除根拠として扱わず、特徴量群としての冗長性を確認する。

## 再現性

- 実行日時: 2026-07-30T10:30:03+07:00
- Python: 3.14.4
- scikit-learn: 1.9.0
- OS: Linux-7.0.0-28-generic-x86_64-with-glibc2.43
- scoring: `accuracy` / repeats: 30 / seed: 42
