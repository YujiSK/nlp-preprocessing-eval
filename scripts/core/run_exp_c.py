"""実験C: 不均衡データにおけるC0(補正なし)/C1(サンプリング調整)比較（docs/execution_plan.md 3章「実験C」）。

C0/C1は4モデル共通条件として評価する（主指標: Average Precision / PR-AUC）。
C2（class_weight='balanced'、3モデル限定）は発展項目として追加実験する。
サンプリング（SMOTE）は imblearn.pipeline.Pipeline 内に置き、学習Foldのみに適用する。
"""

from __future__ import annotations

import sys
from pathlib import Path

TASK9_ROOT = Path(__file__).resolve().parents[2]
if str(TASK9_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK9_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import validation_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.experiments.evaluation import evaluate_pipeline_cv, paired_fold_diff, summarize_cv
from src.experiments.explainability import extract_linear_coefficients, extract_tree_importances
from src.experiments.models import MODEL_ORDER, SUPPORTS_CLASS_WEIGHT, build_model
from src.experiments.preprocessing import build_imbalanced_dataset
from src.utils import RANDOM_STATE, ensure_output_dir, get_outer_cv, get_outer_splits, save_environment_info

EXPERIMENT = "C"

METRICS = {
    "accuracy": lambda yt, yp: (yt == yp).mean(),
    "recall": recall_score,
    "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
    "f1": f1_score,
    "balanced_accuracy": balanced_accuracy_score,
    "mcc": matthews_corrcoef,
}
SCORE_METRICS = {
    "average_precision": average_precision_score,
    "roc_auc": roc_auc_score,
}


def build_dataset():
    return build_imbalanced_dataset()


def pipeline_factory(model_name: str, condition: str):
    def factory():
        if condition == "c0":
            return Pipeline([("scaler", StandardScaler()), ("model", build_model(model_name))])
        if condition == "c1":
            return ImbPipeline(
                [
                    ("scaler", StandardScaler()),
                    ("sampler", SMOTE(random_state=RANDOM_STATE)),
                    ("model", build_model(model_name)),
                ]
            )
        if condition == "c2":
            return Pipeline(
                [("scaler", StandardScaler()), ("model", build_model(model_name, class_weight="balanced"))]
            )
        raise ValueError(condition)

    return factory


def run_cv(X, y, outer_splits):
    records: list[dict] = []
    for condition in ["c0", "c1"]:
        for model_name in MODEL_ORDER:
            records.extend(
                evaluate_pipeline_cv(
                    experiment=EXPERIMENT,
                    condition=condition,
                    model_name=model_name,
                    pipeline_factory=pipeline_factory(model_name, condition),
                    X=X,
                    y=y,
                    outer_splits=outer_splits,
                    metrics=METRICS,
                    score_metrics=SCORE_METRICS,
                )
            )
    # C2（発展）: class_weight='balanced' はk-NNに適用不可のため3モデル限定
    for model_name in MODEL_ORDER:
        if model_name not in SUPPORTS_CLASS_WEIGHT:
            continue
        records.extend(
            evaluate_pipeline_cv(
                experiment=EXPERIMENT,
                condition="c2",
                model_name=model_name,
                pipeline_factory=pipeline_factory(model_name, "c2"),
                X=X,
                y=y,
                outer_splits=outer_splits,
                metrics=METRICS,
                score_metrics=SCORE_METRICS,
            )
        )
    return records


def compute_fold_class_counts(y, outer_splits) -> pd.DataFrame:
    rows = []
    for fold_id, (train_idx, test_idx) in enumerate(outer_splits):
        rows.append(
            dict(
                fold=fold_id,
                n_train=len(train_idx),
                n_pos_train=int(y[train_idx].sum()),
                n_test=len(test_idx),
                n_pos_test=int(y[test_idx].sum()),
            )
        )
    return pd.DataFrame(rows)


def compute_oof_confusion(model_name, condition, X, y, outer_splits) -> np.ndarray:
    cm_total = np.zeros((2, 2), dtype=int)
    factory = pipeline_factory(model_name, condition)
    for train_idx, test_idx in outer_splits:
        pipeline = clone(factory())
        pipeline.fit(X[train_idx], y[train_idx])
        y_pred = pipeline.predict(X[test_idx])
        cm_total += confusion_matrix(y[test_idx], y_pred, labels=[0, 1])
    return cm_total


def plot_confusion_matrices(X, y, outer_splits, out_path, model_name="logistic_regression"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, condition in zip(axes, ["c0", "c1"]):
        cm = compute_oof_confusion(model_name, condition, X, y, outer_splits)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(f"{model_name} ({condition}) - OOF confusion matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_cv_score_bar(summary: pd.DataFrame, out_path):
    ap = summary[(summary["metric"] == "average_precision") & (summary["condition"].isin(["c0", "c1"]))].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    models = MODEL_ORDER
    width = 0.35
    x = range(len(models))
    for offset, condition in zip([-width / 2, width / 2], ["c0", "c1"]):
        rows = ap[ap["condition"] == condition].set_index("model").reindex(models)
        ax.bar([i + offset for i in x], rows["cv_mean"], width=width, yerr=rows["cv_std"], capsize=4, label=condition)
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=20)
    ax.set_ylabel("Average Precision (5-Fold CV mean +/- std)")
    ax.set_title("Experiment C: No correction (C0) vs Sampling (C1) - PR-AUC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_tradeoff(summary: pd.DataFrame, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for condition, marker in [("c0", "o"), ("c1", "s")]:
        for model_name in MODEL_ORDER:
            recall = summary[
                (summary["condition"] == condition) & (summary["model"] == model_name) & (summary["metric"] == "recall")
            ]["cv_mean"]
            precision = summary[
                (summary["condition"] == condition)
                & (summary["model"] == model_name)
                & (summary["metric"] == "precision")
            ]["cv_mean"]
            if len(recall) and len(precision):
                ax.scatter(recall.iloc[0], precision.iloc[0], marker=marker, s=80, label=f"{model_name}-{condition}")
    ax.set_xlabel("Recall (CV mean)")
    ax.set_ylabel("Precision (CV mean)")
    ax.set_title("Experiment C: Recall/Precision trade-off (C0 vs C1)")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_param_tuning(X, y, outer_cv, out_path):
    """LogisticRegressionのCを変化させ、C0/C1でRecall/Precisionのトレードオフがどう変化するかを確認する。"""
    c_range = [0.01, 0.1, 1, 10, 100]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, condition in zip(axes, ["c0", "c1"]):
        pipeline = clone(pipeline_factory("logistic_regression", condition)())
        train_scores, test_scores = validation_curve(
            pipeline,
            X,
            y,
            param_name="model__C",
            param_range=c_range,
            cv=outer_cv,
            scoring="average_precision",
        )
        ax.plot(c_range, train_scores.mean(axis=1), marker="o", label="train")
        ax.plot(c_range, test_scores.mean(axis=1), marker="o", label="CV")
        ax.set_xscale("log")
        ax.set_xlabel("C")
        ax.set_ylabel("Average Precision")
        ax.set_title(f"Logistic Regression ({condition}): C")
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    X, y = build_dataset()
    outer_splits = get_outer_splits(X, y)
    outer_cv = get_outer_cv()

    records = run_cv(X, y, outer_splits)
    summary = summarize_cv(records)

    exp_dir = ensure_output_dir(EXPERIMENT)
    pd.DataFrame(records).to_csv(exp_dir / "expC_metrics_summary.csv", index=False)
    summary.to_csv(exp_dir / "expC_metrics_summary_agg.csv", index=False)

    fold_counts = compute_fold_class_counts(y, outer_splits)
    fold_counts.to_csv(exp_dir / "expC_fold_class_counts.csv", index=False)

    # C0(補正なし)→C1(サンプリング)のFold単位ペア差（knnはc2が存在しないためc0/c1のみで比較）
    diff = paired_fold_diff(records, before_label="c0", after_label="c1")
    diff.to_csv(exp_dir / "expC_paired_fold_diff.csv", index=False)

    plot_cv_score_bar(summary, exp_dir / "expC_cv_score_bar.png")
    plot_confusion_matrices(X, y, outer_splits, exp_dir / "expC_cm_logistic_regression.png")
    plot_tradeoff(summary, exp_dir / "expC_precision_recall_tradeoff.png")
    plot_param_tuning(X, y, outer_cv, exp_dir / "expC_param_tuning.png")

    # 説明性（記述的分析）: C0/C1/C2それぞれで全データ再学習し、係数・重要度を条件間で比較できるようにする
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    coef_rows = []
    importance_rows = []
    for condition in ["c0", "c1", "c2"]:
        logreg = clone(pipeline_factory("logistic_regression", condition)())
        logreg.fit(X, y)
        coef_df = extract_linear_coefficients(logreg, feature_names, top_n=10)
        coef_df["condition"] = condition
        coef_rows.append(coef_df)

        rf = clone(pipeline_factory("random_forest", condition)())
        rf.fit(X, y)
        importance_df = extract_tree_importances(rf, feature_names, top_n=10)
        importance_df["condition"] = condition
        importance_rows.append(importance_df)

    pd.concat(coef_rows, ignore_index=True).to_csv(exp_dir / "expC_coefficients.csv", index=False)
    pd.concat(importance_rows, ignore_index=True).to_csv(exp_dir / "expC_feature_importance.csv", index=False)

    save_environment_info(EXPERIMENT)

    print("=== Experiment C: fold class counts ===")
    print(fold_counts.to_string(index=False))
    print("\n=== Experiment C: CV summary (average_precision) ===")
    print(
        summary[summary["metric"] == "average_precision"][["condition", "model", "cv_mean", "cv_std"]].to_string(
            index=False
        )
    )
    print("\n=== Experiment C: CV summary (recall/precision, C0 vs C1) ===")
    print(
        summary[summary["metric"].isin(["recall", "precision"])][
            ["condition", "model", "metric", "cv_mean", "cv_std"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
