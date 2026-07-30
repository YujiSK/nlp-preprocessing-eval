"""実験A: Breast Cancerデータにおける標準化Before/Afterの比較（docs/execution_plan.md 3章「実験A」）。

必須項目のみを対象とする:
- Before(未標準化)/After(StandardScaler)でのCV平均・標準偏差
- coef_(正/負/絶対値上位10)とfeature_importances_の可視化
- 前処理・学習・推論時間の計測
- n_neighbors/Cのチューニングによる過学習/未学習の挙動観察(Validation Curve)
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
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import validation_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.experiments.evaluation import evaluate_pipeline_cv, paired_fold_diff, summarize_cv
from src.experiments.explainability import extract_linear_coefficients, extract_tree_importances
from src.experiments.models import MODEL_ORDER, build_model
from src.utils import ensure_output_dir, get_outer_cv, get_outer_splits, save_environment_info

EXPERIMENT = "A"
METRICS = {
    "accuracy": accuracy_score,
    "f1_macro": lambda yt, yp: f1_score(yt, yp, average="macro"),
}


def build_pipeline(model_name: str, standardize: bool):
    def factory():
        steps = []
        if standardize:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", build_model(model_name)))
        return Pipeline(steps)

    return factory


def run_cv(X, y, outer_splits) -> list[dict]:
    records: list[dict] = []
    for condition, standardize in [("before", False), ("after", True)]:
        for model_name in MODEL_ORDER:
            factory = build_pipeline(model_name, standardize)
            records.extend(
                evaluate_pipeline_cv(
                    experiment=EXPERIMENT,
                    condition=condition,
                    model_name=model_name,
                    pipeline_factory=factory,
                    X=X,
                    y=y,
                    outer_splits=outer_splits,
                    metrics=METRICS,
                )
            )
    return records


def plot_cv_score_bar(summary: pd.DataFrame, out_path):
    acc = summary[summary["metric"] == "accuracy"].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    models = MODEL_ORDER
    width = 0.35
    x = range(len(models))
    for offset, condition in zip([-width / 2, width / 2], ["before", "after"]):
        rows = acc[acc["condition"] == condition].set_index("model").reindex(models)
        ax.bar(
            [i + offset for i in x],
            rows["cv_mean"],
            width=width,
            yerr=rows["cv_std"],
            capsize=4,
            label=condition,
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=20)
    ax.set_ylabel("Accuracy (5-Fold CV mean +/- std)")
    ax.set_title("Experiment A: Before/After Standardization - CV Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_feature_importance(coef_df: pd.DataFrame, importance_df: pd.DataFrame, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    abs_coef = coef_df[coef_df["rank_type"] == "absolute"].sort_values("abs_coefficient")
    axes[0].barh(abs_coef["feature"], abs_coef["coefficient"])
    axes[0].set_title("Logistic Regression: |coef_| top10 (After)")
    axes[0].axvline(0, color="black", linewidth=0.8)

    imp = importance_df.sort_values("importance")
    axes[1].barh(imp["feature"], imp["importance"])
    axes[1].set_title("Random Forest: feature_importances_ top10")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_validation_curves(X, y, outer_cv, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    n_neighbors_range = [1, 3, 5, 10, 20]
    train_scores, test_scores = validation_curve(
        Pipeline([("scaler", StandardScaler()), ("model", build_model("knn"))]),
        X,
        y,
        param_name="model__n_neighbors",
        param_range=n_neighbors_range,
        cv=outer_cv,
        scoring="accuracy",
    )
    axes[0].plot(n_neighbors_range, train_scores.mean(axis=1), marker="o", label="train")
    axes[0].plot(n_neighbors_range, test_scores.mean(axis=1), marker="o", label="CV")
    axes[0].set_xlabel("n_neighbors")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("k-NN: n_neighbors")
    axes[0].legend()

    c_range = [0.01, 0.1, 1, 10, 100]
    train_scores, test_scores = validation_curve(
        Pipeline([("scaler", StandardScaler()), ("model", build_model("logistic_regression"))]),
        X,
        y,
        param_name="model__C",
        param_range=c_range,
        cv=outer_cv,
        scoring="accuracy",
    )
    axes[1].plot(c_range, train_scores.mean(axis=1), marker="o", label="train")
    axes[1].plot(c_range, test_scores.mean(axis=1), marker="o", label="CV")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("C")
    axes[1].set_title("Logistic Regression: C")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    data = load_breast_cancer()
    X, y, feature_names = data.data, data.target, list(data.feature_names)

    outer_splits = get_outer_splits(X, y)
    outer_cv = get_outer_cv()

    records = run_cv(X, y, outer_splits)
    summary = summarize_cv(records)
    diff = paired_fold_diff(records, before_label="before", after_label="after")

    exp_dir = ensure_output_dir(EXPERIMENT)
    pd.DataFrame(records).to_csv(exp_dir / "expA_metrics_summary.csv", index=False)
    summary.to_csv(exp_dir / "expA_metrics_summary_agg.csv", index=False)
    diff.to_csv(exp_dir / "expA_paired_fold_diff.csv", index=False)

    plot_cv_score_bar(summary, exp_dir / "expA_cv_score_bar.png")

    after_logreg = Pipeline([("scaler", StandardScaler()), ("model", build_model("logistic_regression"))])
    after_logreg.fit(X, y)
    coef_df = extract_linear_coefficients(after_logreg, feature_names, top_n=10)
    coef_df.to_csv(exp_dir / "expA_coefficients.csv", index=False)

    rf = Pipeline([("model", build_model("random_forest"))])
    rf.fit(X, y)
    importance_df = extract_tree_importances(rf, feature_names, top_n=10)
    importance_df.to_csv(exp_dir / "expA_feature_importance.csv", index=False)

    plot_feature_importance(coef_df, importance_df, exp_dir / "expA_feature_importance.png")
    plot_validation_curves(X, y, outer_cv, exp_dir / "expA_param_tuning.png")

    save_environment_info(EXPERIMENT)

    print("=== Experiment A: CV accuracy summary ===")
    print(
        summary[summary["metric"] == "accuracy"][["condition", "model", "cv_mean", "cv_std"]].to_string(
            index=False
        )
    )
    print("\n=== Experiment A: paired fold diff (accuracy) ===")
    print(diff[diff["metric"] == "accuracy"].to_string(index=False))


if __name__ == "__main__":
    main()
