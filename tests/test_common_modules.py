"""共通モジュール(src/utils.py, models.py, evaluation.py, explainability.py)のスモークテスト。

Breast Cancerデータ(実験Aと同一データセット)を用いて、
Before(未標準化)/After(StandardScaler)のCVパイプラインが一連の流れで動作することを確認する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments.evaluation import evaluate_pipeline_cv, paired_fold_diff, summarize_cv
from src.experiments.explainability import extract_linear_coefficients, extract_tree_importances
from src.experiments.models import build_model
from src.utils import append_long_records, ensure_output_dir, get_outer_splits


def _load_data():
    data = load_breast_cancer()
    return data.data, data.target, list(data.feature_names)


def _pipeline_factory(model_name: str, standardize: bool):
    def factory():
        steps = []
        if standardize:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", build_model(model_name)))
        return Pipeline(steps)

    return factory


def test_outer_splits_are_reusable_and_stratified():
    X, y, _ = _load_data()
    splits = get_outer_splits(X, y)
    assert len(splits) == 5
    for train_idx, test_idx in splits:
        assert len(set(train_idx) & set(test_idx)) == 0
        assert len(train_idx) + len(test_idx) == len(y)


def test_evaluate_pipeline_cv_before_after_all_models():
    X, y, feature_names = _load_data()
    outer_splits = get_outer_splits(X, y)
    metrics = {
        "accuracy": accuracy_score,
        "f1_macro": lambda yt, yp: f1_score(yt, yp, average="macro"),
    }

    all_records: list[dict] = []
    fitted_pipelines = {}
    for condition, standardize in [("before", False), ("after", True)]:
        for model_name in ["logistic_regression", "linear_svc", "random_forest", "knn"]:
            factory = _pipeline_factory(model_name, standardize)
            records = evaluate_pipeline_cv(
                experiment="A",
                condition=condition,
                model_name=model_name,
                pipeline_factory=factory,
                X=X,
                y=y,
                outer_splits=outer_splits,
                metrics=metrics,
            )
            assert len(records) == len(outer_splits) * len(metrics)
            all_records.extend(records)

            fitted = factory()
            fitted.fit(X, y)
            fitted_pipelines[(condition, model_name)] = fitted

    summary = summarize_cv(all_records)
    assert {"experiment", "condition", "model", "metric", "cv_mean", "cv_std", "n_folds"} <= set(
        summary.columns
    )
    assert (summary["n_folds"] == 5).all()

    diff = paired_fold_diff(all_records, before_label="before", after_label="after")
    assert {"model", "metric", "mean_diff", "std_diff", "n_improved", "n_worsened"} <= set(diff.columns)
    knn_acc_diff = diff[(diff["model"] == "knn") & (diff["metric"] == "accuracy")]
    assert len(knn_acc_diff) == 1

    coef_df = extract_linear_coefficients(
        fitted_pipelines[("after", "logistic_regression")], feature_names, top_n=10
    )
    assert set(coef_df["rank_type"]) == {"positive", "negative", "absolute"}
    assert len(coef_df[coef_df["rank_type"] == "positive"]) == 10

    importance_df = extract_tree_importances(
        fitted_pipelines[("after", "random_forest")], feature_names, top_n=10
    )
    assert len(importance_df) == 10
    assert (importance_df["importance"].diff().dropna() <= 1e-12).all()


def test_output_helpers(tmp_path, monkeypatch):
    import src.utils as utils_module

    monkeypatch.setattr(utils_module, "OUTPUTS_ROOT", tmp_path / "outputs")
    exp_dir = ensure_output_dir("smoke")
    assert exp_dir.exists()

    records = [{"experiment": "smoke", "condition": "before", "model": "knn", "fold": 0, "metric": "accuracy", "value": 0.9}]
    csv_path = exp_dir / "exp_smoke_metrics_summary.csv"
    df = append_long_records(records, csv_path)
    assert csv_path.exists()
    assert len(df) == 1

    df2 = append_long_records(records, csv_path)
    assert len(df2) == 2
