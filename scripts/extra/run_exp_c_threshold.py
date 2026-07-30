"""実験C（発展）: Inner CVで閾値を選択し、Outer Foldだけで性能評価する。

既存の実験C成果物・最終レポートには変更を加えず、Appendix用成果物だけを
``outputs/exp_c_extra/`` に保存する。
"""

from __future__ import annotations

import platform
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TASK9_ROOT = Path(__file__).resolve().parents[2]
if str(TASK9_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK9_ROOT))

from src.experiments.preprocessing import build_imbalanced_dataset as build_dataset
from src.experiments.models import build_model
from src.utils import RANDOM_STATE, get_outer_splits

OUTPUT_DIR = TASK9_ROOT / "outputs" / "exp_c_extra"
MODELS = ["logistic_regression", "linear_svc"]
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)
INNER_SPLITS = 3


def build_probability_estimator(model_name: str):
    """確率出力可能な推定器を返す。校正も与えられた学習範囲内だけで行う。"""
    base = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", build_model(model_name)),
        ]
    )
    if model_name == "logistic_regression":
        return base
    if model_name == "linear_svc":
        return CalibratedClassifierCV(
            estimator=base,
            method="sigmoid",
            cv=INNER_SPLITS,
            n_jobs=1,
        )
    raise ValueError(model_name)


def select_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    model_name: str,
    outer_fold: int,
) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in THRESHOLD_GRID:
        y_pred = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "model": model_name,
                "outer_fold": outer_fold,
                "threshold": threshold,
                "inner_f1": f1_score(y_true, y_pred, zero_division=0),
                "inner_precision": precision_score(y_true, y_pred, zero_division=0),
                "inner_recall": recall_score(y_true, y_pred, zero_division=0),
                "inner_balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            }
        )
    grid = pd.DataFrame(rows)
    # F1最大。同値なら既定値0.5に近い閾値、さらに同値なら低い閾値を選ぶ。
    ranked = grid.assign(distance_from_default=(grid["threshold"] - 0.5).abs()).sort_values(
        ["inner_f1", "distance_from_default", "threshold"],
        ascending=[False, True, True],
    )
    return float(ranked.iloc[0]["threshold"]), grid


def classification_metrics(y_true, y_pred, probabilities) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "average_precision": average_precision_score(y_true, probabilities),
        "roc_auc": roc_auc_score(y_true, probabilities),
    }


def run_nested_cv(X: np.ndarray, y: np.ndarray):
    metric_rows: list[dict] = []
    selection_rows: list[dict] = []
    grid_frames: list[pd.DataFrame] = []

    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(
        get_outer_splits(X, y)
    ):
        X_train, X_test = X[outer_train_idx], X[outer_test_idx]
        y_train, y_test = y[outer_train_idx], y[outer_test_idx]
        inner_cv = StratifiedKFold(
            n_splits=INNER_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE + outer_fold,
        )

        for model_name in MODELS:
            inner_estimator = build_probability_estimator(model_name)
            inner_oof_probability = cross_val_predict(
                inner_estimator,
                X_train,
                y_train,
                cv=inner_cv,
                method="predict_proba",
                n_jobs=1,
            )[:, 1]
            selected_threshold, grid = select_threshold(
                y_train,
                inner_oof_probability,
                model_name,
                outer_fold,
            )
            grid_frames.append(grid)

            final_estimator = build_probability_estimator(model_name)
            final_estimator.fit(X_train, y_train)
            outer_probability = final_estimator.predict_proba(X_test)[:, 1]

            selection_rows.append(
                {
                    "model": model_name,
                    "outer_fold": outer_fold,
                    "selected_threshold": selected_threshold,
                    "inner_best_f1": float(grid["inner_f1"].max()),
                    "n_outer_train": len(outer_train_idx),
                    "n_outer_test": len(outer_test_idx),
                    "n_positive_train": int(y_train.sum()),
                    "n_positive_test": int(y_test.sum()),
                    "inner_splits": INNER_SPLITS,
                    "seed": RANDOM_STATE,
                }
            )

            for threshold_mode, threshold in (
                ("default_0.5", 0.5),
                ("inner_cv_tuned", selected_threshold),
            ):
                outer_pred = (outer_probability >= threshold).astype(int)
                base = {
                    "experiment": "C_threshold",
                    "condition": "c0",
                    "model": model_name,
                    "outer_fold": outer_fold,
                    "threshold_mode": threshold_mode,
                    "threshold": threshold,
                    "seed": RANDOM_STATE,
                }
                for metric, value in classification_metrics(
                    y_test, outer_pred, outer_probability
                ).items():
                    metric_rows.append({**base, "metric": metric, "value": value})

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(selection_rows),
        pd.concat(grid_frames, ignore_index=True),
    )


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["condition", "model", "threshold_mode", "metric"])["value"]
        .agg(cv_mean="mean", cv_std="std", n_outer_folds="count")
        .reset_index()
    )


def paired_differences(metrics: pd.DataFrame) -> pd.DataFrame:
    pivot = metrics.pivot(
        index=["model", "outer_fold", "metric"],
        columns="threshold_mode",
        values="value",
    ).reset_index()
    pivot["tuned_minus_default"] = pivot["inner_cv_tuned"] - pivot["default_0.5"]
    return pivot


def plot_results(
    summary: pd.DataFrame, selections: pd.DataFrame, out_path: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    metrics = ["f1", "precision", "recall"]
    colors = {"default_0.5": "#7f8c8d", "inner_cv_tuned": "#d95f02"}
    x = np.arange(len(MODELS))
    width = 0.36

    for ax, metric in zip(axes.flat[:3], metrics):
        rows = summary[summary["metric"] == metric]
        for offset, mode in ((-width / 2, "default_0.5"), (width / 2, "inner_cv_tuned")):
            values = rows[rows["threshold_mode"] == mode].set_index("model").reindex(
                MODELS
            )
            ax.bar(
                x + offset,
                values["cv_mean"],
                width,
                yerr=values["cv_std"],
                capsize=3,
                color=colors[mode],
                label=mode,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=15)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Outer-fold {metric}")
        ax.set_ylabel("5-Fold mean +/- std")
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend()

    threshold_ax = axes[1, 1]
    for i, model_name in enumerate(MODELS):
        values = selections[selections["model"] == model_name].sort_values(
            "outer_fold"
        )
        threshold_ax.plot(
            values["outer_fold"],
            values["selected_threshold"],
            marker="o",
            label=model_name,
        )
    threshold_ax.axhline(0.5, color="#7f8c8d", linestyle="--", label="default 0.5")
    threshold_ax.set_xticks(range(5))
    threshold_ax.set_xlabel("Outer fold")
    threshold_ax.set_ylabel("Inner-CV selected threshold")
    threshold_ax.set_title("Selected threshold by outer fold")
    threshold_ax.set_ylim(0, 1)
    threshold_ax.legend(fontsize=8)
    threshold_ax.grid(alpha=0.2)

    fig.suptitle("Experiment C Appendix: Nested Threshold Selection (C0)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def write_appendix(
    summary: pd.DataFrame,
    selections: pd.DataFrame,
    differences: pd.DataFrame,
    out_path: Path,
) -> None:
    lines = [
        "# APPENDIX_EXP_C_THRESHOLD — 実験C（発展）：内側CVによる閾値最適化",
        "",
        "## 目的と設計",
        "",
        "不均衡二値分類において、判定閾値を外側testで調整する評価リークを避けながら、F1最大化閾値の効果を評価した。既存実験Cと同じ2,000件・正例約7%の合成データ、補正なしC0条件、外側5-Foldを使用した。閾値効果をSMOTEやクラス重みの効果と混在させないため、対象はLogistic RegressionとLinear SVCに限定した。",
        "",
        "各外側train内で3-Fold OOF確率を生成し、0.05〜0.95（0.01刻み）からF1が最大となる閾値を選択した。同値の場合は0.5に近い値を採用した。選択後、外側train全体でモデルを再fitし、未使用の外側testで一度だけ評価した。Linear SVCは`CalibratedClassifierCV(method='sigmoid', cv=3)`を各学習範囲内で使用した。",
        "",
        "## 結果",
        "",
        "| モデル | 選択閾値 mean ± std | F1: 0.5 | F1: tuned | ΔF1 | Precision: 0.5→tuned | Recall: 0.5→tuned |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]

    def summary_value(model: str, mode: str, metric: str, column="cv_mean") -> float:
        return float(
            summary[
                (summary["model"] == model)
                & (summary["threshold_mode"] == mode)
                & (summary["metric"] == metric)
            ][column].iloc[0]
        )

    for model_name in MODELS:
        threshold_values = selections[selections["model"] == model_name][
            "selected_threshold"
        ]
        f1_default = summary_value(model_name, "default_0.5", "f1")
        f1_tuned = summary_value(model_name, "inner_cv_tuned", "f1")
        precision_default = summary_value(model_name, "default_0.5", "precision")
        precision_tuned = summary_value(model_name, "inner_cv_tuned", "precision")
        recall_default = summary_value(model_name, "default_0.5", "recall")
        recall_tuned = summary_value(model_name, "inner_cv_tuned", "recall")
        lines.append(
            f"| {model_name} | {threshold_values.mean():.3f} ± {threshold_values.std(ddof=1):.3f} "
            f"| {f1_default:.3f} | {f1_tuned:.3f} | {f1_tuned - f1_default:+.3f} "
            f"| {precision_default:.3f}→{precision_tuned:.3f} "
            f"| {recall_default:.3f}→{recall_tuned:.3f} |"
        )

    f1_differences = differences[differences["metric"] == "f1"]
    fold_statements = []
    for model_name in MODELS:
        values = f1_differences[f1_differences["model"] == model_name][
            "tuned_minus_default"
        ]
        fold_statements.append(
            f"{model_name}は改善{int((values > 0).sum())} Fold・"
            f"悪化{int((values < 0).sum())} Fold"
        )

    lines.extend(
        [
            "",
            "Fold単位のF1差では、" + "、".join(fold_statements) + "だった。",
            "",
            "![閾値最適化比較](expC_threshold_comparison.png)",
            "",
            "## 解釈上の注意",
            "",
            "最適化対象をF1に事前固定したため、閾値低下によってRecallが上がる一方、Precisionが下がる可能性がある。Average PrecisionとROC-AUCは確率ランキングに依存する閾値非依存指標であり、同一外側Fold・同一モデルではdefault/tuned間で変化しない。",
            "",
            "選択閾値と外側性能にはFold間変動がある。ここで得た閾値を普遍的な固定値とは解釈せず、運用時のクラス比率・誤検知/見逃しコスト・確率校正の変化に応じて学習データ内だけで再選択する必要がある。外側Foldは閾値選択にも校正にも使用していない。",
            "",
            "## 再現性",
            "",
            f"- 実行日時: {datetime.now().astimezone().isoformat(timespec='seconds')}",
            f"- Python: {sys.version.split()[0]}",
            f"- scikit-learn: {sklearn.__version__}",
            f"- OS: {platform.platform()}",
            f"- outer folds: 5 / inner folds: {INNER_SPLITS} / seed: {RANDOM_STATE}",
            "- condition: C0（補正なし） / threshold objective: F1",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y = build_dataset()
    metrics, selections, grid = run_nested_cv(X, y)
    summary = summarize(metrics)
    differences = paired_differences(metrics)

    metrics.to_csv(OUTPUT_DIR / "expC_threshold_outer_fold_metrics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "expC_threshold_summary.csv", index=False)
    selections.to_csv(OUTPUT_DIR / "expC_threshold_selections.csv", index=False)
    grid.to_csv(OUTPUT_DIR / "expC_threshold_inner_grid.csv", index=False)
    differences.to_csv(
        OUTPUT_DIR / "expC_threshold_paired_differences.csv", index=False
    )
    plot_results(summary, selections, OUTPUT_DIR / "expC_threshold_comparison.png")
    write_appendix(
        summary,
        selections,
        differences,
        OUTPUT_DIR / "APPENDIX_EXP_C_THRESHOLD.md",
    )

    print("=== Experiment C advanced: selected thresholds ===")
    print(selections.to_string(index=False))
    print("\n=== Outer-fold F1/Precision/Recall summary ===")
    print(
        summary[summary["metric"].isin(["f1", "precision", "recall"])].to_string(
            index=False
        )
    )
    print(f"\nSaved Appendix artifacts to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
