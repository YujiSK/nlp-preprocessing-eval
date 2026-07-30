"""Before/After × 5-Fold CVでの評価・処理時間計測を行う共通関数。

docs/execution_plan.md 第3章（各実験の検証内容）、第4.6章（処理コスト計測の定義）、
第5章（Fold単位のペア差）に対応する。
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone

from ..utils import timer


def _select(data, idx):
    if hasattr(data, "iloc"):
        return data.iloc[idx]
    return data[idx]


def _decision_scores(pipeline, X_test):
    """PR-AUC/ROC-AUC等、スコアを要する指標のためにdecision_function/predict_probaを取得する。"""
    if hasattr(pipeline, "decision_function"):
        return pipeline.decision_function(X_test)
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X_test)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba
    return None


def evaluate_pipeline_cv(
    experiment: str,
    condition: str,
    model_name: str,
    pipeline_factory: Callable[[], object],
    X,
    y,
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
    metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]],
    score_metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]] | None = None,
    seed: int = 42,
) -> list[dict]:
    """1つの(experiment, condition, model)についてFoldごとに学習・評価し、Long形式レコードを返す。

    - pipeline_factory: 呼び出すたびに未学習のPipelineを返す関数（学習型前処理はPipeline内でFold毎にfitされる）
    - metrics: `predict()` の出力(y_pred)を使う指標（accuracy, f1等）
    - score_metrics: `decision_function`/`predict_proba` の出力を使う指標（PR-AUC, ROC-AUC等）
    - outer_splits: `utils.get_outer_splits()` で生成した、全モデル・全条件で共通の分割
    """
    records: list[dict] = []
    y_array = np.asarray(y)

    for fold_id, (train_idx, test_idx) in enumerate(outer_splits):
        X_train, X_test = _select(X, train_idx), _select(X, test_idx)
        y_train, y_test = y_array[train_idx], y_array[test_idx]

        pipeline = clone(pipeline_factory())

        with timer() as fit_t:
            pipeline.fit(X_train, y_train)
        with timer() as predict_t:
            y_pred = pipeline.predict(X_test)

        base = dict(
            experiment=experiment,
            condition=condition,
            model=model_name,
            fold=fold_id,
            seed=seed,
            n_train=len(train_idx),
            n_test=len(test_idx),
            fit_seconds=fit_t.seconds,
            predict_seconds=predict_t.seconds,
        )

        for metric_name, fn in metrics.items():
            records.append({**base, "metric": metric_name, "value": fn(y_test, y_pred)})

        if score_metrics:
            y_score = _decision_scores(pipeline, X_test)
            if y_score is not None:
                for metric_name, fn in score_metrics.items():
                    records.append({**base, "metric": metric_name, "value": fn(y_test, y_score)})

    return records


def summarize_cv(records: list[dict]) -> pd.DataFrame:
    """experiment, condition, model, metric ごとにCV平均・標準偏差を集計する（第5章）。"""
    df = pd.DataFrame(records)
    return (
        df.groupby(["experiment", "condition", "model", "metric"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "cv_mean", "std": "cv_std", "count": "n_folds"})
    )


def paired_fold_diff(records: list[dict], before_label: str, after_label: str) -> pd.DataFrame:
    """Fold単位のBefore→After差分（平均差・差の標準偏差・改善/悪化Fold数、第5章）を算出する。"""
    df = pd.DataFrame(records)
    before = df[df["condition"] == before_label].set_index(["model", "metric", "fold"])["value"]
    after = df[df["condition"] == after_label].set_index(["model", "metric", "fold"])["value"]
    diff = (after - before).rename("diff").reset_index()
    summary = (
        diff.groupby(["model", "metric"])["diff"]
        .agg(
            mean_diff="mean",
            std_diff="std",
            n_improved=lambda s: int((s > 0).sum()),
            n_worsened=lambda s: int((s < 0).sum()),
        )
        .reset_index()
    )
    return summary
