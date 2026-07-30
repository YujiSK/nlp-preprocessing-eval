"""実験B: 欠損値処理Before/Afterの比較（docs/execution_plan.md 3章「実験B」）。

完全データ（合成・欠損なし）を用意し、外側trainのみに人工欠損を注入する。
Before/Afterで同一の欠損マスクを共有し、外側test（欠損なし）は共通のまま評価する。
One-Hot Encodingは両条件で共通化し、欠損処理（削除 vs 補完）のみを比較対象とする。
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
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import validation_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.experiments.evaluation import paired_fold_diff, summarize_cv
from src.experiments.explainability import extract_linear_coefficients, extract_tree_importances
from src.experiments.models import MODEL_ORDER, build_model
from src.utils import RANDOM_STATE, ensure_output_dir, get_outer_cv, get_outer_splits, save_environment_info, timer

EXPERIMENT = "B"
MISSING_RATE = 0.15
METRICS = {
    "accuracy": accuracy_score,
    "f1_macro": lambda yt, yp: f1_score(yt, yp, average="macro"),
}


def build_dataset():
    """数値＋カテゴリ変数混在の完全データ（欠損なし）を合成する。"""
    X_num, y = make_classification(
        n_samples=800,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        random_state=RANDOM_STATE,
    )
    numeric_cols = [f"num_{i}" for i in range(X_num.shape[1])]
    df = pd.DataFrame(X_num, columns=numeric_cols)

    rng = np.random.default_rng(RANDOM_STATE)
    df["cat_region"] = pd.qcut(df["num_0"], q=4, labels=["north", "south", "east", "west"]).astype(str)
    df["cat_channel"] = rng.choice(["online", "store", "phone"], size=len(df), p=[0.5, 0.35, 0.15])

    categorical_cols = ["cat_region", "cat_channel"]
    return df, y, numeric_cols, categorical_cols


def inject_missingness(df: pd.DataFrame, rng: np.random.Generator, missing_rate: float = MISSING_RATE) -> pd.DataFrame:
    """全列に対して独立にmissing_rateの確率で欠損させる（MCAR：完全ランダム欠損）。"""
    df = df.copy()
    for col in df.columns:
        mask = rng.random(len(df)) < missing_rate
        df.loc[mask, col] = np.nan
    return df


def build_column_transformer(numeric_cols, categorical_cols, impute: bool) -> ColumnTransformer:
    if impute:
        numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
            ]
        )
    else:
        numeric_pipeline = Pipeline([("scaler", StandardScaler())])
        categorical_pipeline = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2))])

    return ColumnTransformer(
        [("num", numeric_pipeline, numeric_cols), ("cat", categorical_pipeline, categorical_cols)]
    )


def run_cv(df, y, numeric_cols, categorical_cols, outer_splits):
    records: list[dict] = []
    edge_case_rows: list[dict] = []

    for fold_id, (train_idx, test_idx) in enumerate(outer_splits):
        X_train_full = df.iloc[train_idx].reset_index(drop=True)
        y_train_full = y[train_idx]
        X_test = df.iloc[test_idx]
        y_test = y[test_idx]

        fold_rng = np.random.default_rng(RANDOM_STATE + fold_id)
        X_train_missing = inject_missingness(X_train_full, fold_rng)

        # エッジケース確認（必須）: 学習Fold内での全欠損列・低頻度カテゴリを記録する
        fully_missing_cols = X_train_missing.columns[X_train_missing.isna().all()].tolist()
        edge_case_row = {"fold": fold_id, "fully_missing_columns": ";".join(fully_missing_cols)}
        for col in categorical_cols:
            counts = X_train_missing[col].value_counts(dropna=True)
            edge_case_row[f"min_category_count_{col}"] = int(counts.min()) if len(counts) else 0
        edge_case_rows.append(edge_case_row)

        complete_mask = (~X_train_missing.isna().any(axis=1)).to_numpy()
        X_train_before = X_train_missing[complete_mask]
        y_train_before = y_train_full[complete_mask]

        conditions = [
            ("before", X_train_before, y_train_before, False),
            ("after", X_train_missing, y_train_full, True),
        ]

        for condition, X_train, y_train, impute in conditions:
            for model_name in MODEL_ORDER:
                pipeline = Pipeline(
                    [
                        ("preprocessor", build_column_transformer(numeric_cols, categorical_cols, impute=impute)),
                        ("model", build_model(model_name)),
                    ]
                )
                with timer() as fit_t:
                    pipeline.fit(X_train, y_train)
                with timer() as predict_t:
                    y_pred = pipeline.predict(X_test)

                base = dict(
                    experiment=EXPERIMENT,
                    condition=condition,
                    model=model_name,
                    fold=fold_id,
                    seed=RANDOM_STATE,
                    n_train=len(X_train),
                    n_test=len(X_test),
                    fit_seconds=fit_t.seconds,
                    predict_seconds=predict_t.seconds,
                )
                for metric_name, fn in METRICS.items():
                    records.append({**base, "metric": metric_name, "value": fn(y_test, y_pred)})

    return records, edge_case_rows


def plot_cv_score_bar(summary: pd.DataFrame, out_path):
    acc = summary[summary["metric"] == "accuracy"].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    models = MODEL_ORDER
    width = 0.35
    x = range(len(models))
    for offset, condition in zip([-width / 2, width / 2], ["before", "after"]):
        rows = acc[acc["condition"] == condition].set_index("model").reindex(models)
        ax.bar([i + offset for i in x], rows["cv_mean"], width=width, yerr=rows["cv_std"], capsize=4, label=condition)
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=20)
    ax.set_ylabel("Accuracy (5-Fold CV mean +/- std)")
    ax.set_title("Experiment B: Row-deletion vs Imputation - CV Accuracy")
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
    axes[1].set_title("Random Forest: feature_importances_ top10 (After)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_param_tuning(df, y, numeric_cols, categorical_cols, outer_cv, out_path):
    fold_rng = np.random.default_rng(RANDOM_STATE + 1000)
    X_missing = inject_missingness(df, fold_rng)

    depth_range = [3, 5, 10, 20]
    pipeline = Pipeline(
        [
            ("preprocessor", build_column_transformer(numeric_cols, categorical_cols, impute=True)),
            ("model", build_model("random_forest")),
        ]
    )
    train_scores, test_scores = validation_curve(
        pipeline,
        X_missing,
        y,
        param_name="model__max_depth",
        param_range=depth_range,
        cv=outer_cv,
        scoring="accuracy",
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(len(depth_range))
    ax.plot(x, train_scores.mean(axis=1), marker="o", label="train")
    ax.plot(x, test_scores.mean(axis=1), marker="o", label="CV")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(d) for d in depth_range])
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("Experiment B: Random Forest max_depth (After condition)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    df, y, numeric_cols, categorical_cols = build_dataset()
    outer_splits = get_outer_splits(df, y)
    outer_cv = get_outer_cv()

    records, edge_case_rows = run_cv(df, y, numeric_cols, categorical_cols, outer_splits)
    summary = summarize_cv(records)
    diff = paired_fold_diff(records, before_label="before", after_label="after")

    exp_dir = ensure_output_dir(EXPERIMENT)
    pd.DataFrame(records).to_csv(exp_dir / "expB_metrics_summary.csv", index=False)
    summary.to_csv(exp_dir / "expB_metrics_summary_agg.csv", index=False)
    diff.to_csv(exp_dir / "expB_paired_fold_diff.csv", index=False)
    pd.DataFrame(edge_case_rows).to_csv(exp_dir / "expB_edge_cases.csv", index=False)

    n_train_by_condition = (
        pd.DataFrame(records)[["condition", "fold", "n_train"]].drop_duplicates().sort_values(["condition", "fold"])
    )
    n_train_by_condition.to_csv(exp_dir / "expB_fold_sample_sizes.csv", index=False)

    plot_cv_score_bar(summary, exp_dir / "expB_cv_score_bar.png")

    fold_rng = np.random.default_rng(RANDOM_STATE + 1000)
    X_missing_full = inject_missingness(df, fold_rng)

    after_logreg = Pipeline(
        [
            ("preprocessor", build_column_transformer(numeric_cols, categorical_cols, impute=True)),
            ("model", build_model("logistic_regression")),
        ]
    )
    after_logreg.fit(X_missing_full, y)
    feature_names = list(after_logreg.named_steps["preprocessor"].get_feature_names_out())
    coef_df = extract_linear_coefficients(after_logreg, feature_names, top_n=10)
    coef_df.to_csv(exp_dir / "expB_coefficients.csv", index=False)

    after_rf = Pipeline(
        [
            ("preprocessor", build_column_transformer(numeric_cols, categorical_cols, impute=True)),
            ("model", build_model("random_forest")),
        ]
    )
    after_rf.fit(X_missing_full, y)
    importance_df = extract_tree_importances(after_rf, feature_names, top_n=10)
    importance_df.to_csv(exp_dir / "expB_feature_importance.csv", index=False)

    plot_feature_importance(coef_df, importance_df, exp_dir / "expB_feature_importance.png")
    plot_param_tuning(df, y, numeric_cols, categorical_cols, outer_cv, exp_dir / "expB_param_tuning.png")

    save_environment_info(EXPERIMENT)

    print("=== Experiment B: CV accuracy summary ===")
    print(summary[summary["metric"] == "accuracy"][["condition", "model", "cv_mean", "cv_std"]].to_string(index=False))
    print("\n=== Experiment B: Fold sample sizes (train) ===")
    print(n_train_by_condition.to_string(index=False))
    print("\n=== Experiment B: paired fold diff (accuracy) ===")
    print(diff[diff["metric"] == "accuracy"].to_string(index=False))
    print("\n=== Experiment B: edge cases ===")
    print(pd.DataFrame(edge_case_rows).to_string(index=False))


if __name__ == "__main__":
    main()
