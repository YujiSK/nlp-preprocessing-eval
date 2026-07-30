"""線形モデルの係数・木モデルの特徴量重要度を抽出・整形する。

docs/execution_plan.md 第4.5章（説明性評価の注意点）に対応する。
- 係数は正/負/絶対値の上位を分けて提示する（絶対値だけでは正負の方向性が失われるため）。
- Random Forestの feature_importances_ は不純度ベースであり、高カーディナリティ・
  連続値特徴量に偏りやすい点に留意する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _final_estimator(pipeline):
    return pipeline[-1] if hasattr(pipeline, "__getitem__") else pipeline


def extract_linear_coefficients(
    pipeline, feature_names: list[str], top_n: int = 10, class_index: int = 0
) -> pd.DataFrame:
    """正の係数上位N・負の係数上位N・絶対値上位Nを分けて返す（二値分類はclass_index=0のみ使用）。"""
    model = _final_estimator(pipeline)
    coef = np.asarray(model.coef_)
    coef_row = coef[class_index] if coef.ndim == 2 else coef

    df = pd.DataFrame({"feature": feature_names, "coefficient": coef_row})
    df["abs_coefficient"] = df["coefficient"].abs()

    positive = df.sort_values("coefficient", ascending=False).head(top_n).assign(rank_type="positive")
    negative = df.sort_values("coefficient", ascending=True).head(top_n).assign(rank_type="negative")
    absolute = df.sort_values("abs_coefficient", ascending=False).head(top_n).assign(rank_type="absolute")
    return pd.concat([positive, negative, absolute], ignore_index=True)


def extract_tree_importances(pipeline, feature_names: list[str], top_n: int = 10) -> pd.DataFrame:
    """Random Forestの feature_importances_ 上位Nを返す。"""
    model = _final_estimator(pipeline)
    df = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
    return df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
