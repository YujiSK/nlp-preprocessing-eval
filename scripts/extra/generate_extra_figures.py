"""発展AppendixのPDF向け個別図を、保存済みCSVから再生成する。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TASK9_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = TASK9_ROOT / "outputs"
MODEL_ORDER = ["logistic_regression", "linear_svc", "random_forest", "knn"]
MODEL_SHORT = {
    "logistic_regression": "lr",
    "linear_svc": "svc",
    "random_forest": "rf",
    "knn": "knn",
}


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_a() -> list[Path]:
    out = OUTPUTS / "exp_a_extra"
    summary = pd.read_csv(out / "expA_permutation_importance_summary.csv")
    paths = []
    for model in MODEL_ORDER:
        rows = summary[summary["model"] == model]
        selected = rows.groupby("feature")["pi_mean"].max().nlargest(10).index
        mean = (
            rows[rows["feature"].isin(selected)]
            .pivot(index="feature", columns="condition", values="pi_mean")
            .reindex(selected)
            .sort_values("after")
        )
        std = (
            rows[rows["feature"].isin(mean.index)]
            .pivot(index="feature", columns="condition", values="pi_std")
            .reindex(index=mean.index, columns=mean.columns)
        )
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        mean.plot.barh(
            ax=ax,
            xerr=std,
            capsize=2,
            color={"before": "#7f8c8d", "after": "#2878b5"},
        )
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_title(f"Permutation Importance: {model}")
        ax.set_xlabel("Mean decrease in validation accuracy")
        ax.set_ylabel("")
        ax.legend(title="")
        path = out / f"expA_permutation_{MODEL_SHORT[model]}.png"
        _save(fig, path)
        paths.append(path)
    return paths


def generate_b() -> list[Path]:
    out = OUTPUTS / "exp_b_extra"
    summary = pd.read_csv(out / "expB_coverage_summary.csv")
    metrics = [
        ("coverage", "Coverage"),
        ("accuracy_predicted", "Accuracy on predicted rows"),
        ("correct_fraction_all", "Correct predictions / all requests"),
    ]
    paths = []
    for model in MODEL_ORDER:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        x = np.arange(len(metrics))
        width = 0.36
        for offset, condition, color in (
            (-width / 2, "before", "#7f8c8d"),
            (width / 2, "after", "#2878b5"),
        ):
            means, stds = [], []
            for metric, _ in metrics:
                row = summary[
                    (summary["model"] == model)
                    & (summary["condition"] == condition)
                    & (summary["metric"] == metric)
                ].iloc[0]
                means.append(row["cv_mean"])
                stds.append(row["cv_std"])
            ax.bar(
                x + offset,
                means,
                width,
                yerr=stds,
                capsize=3,
                label=condition,
                color=color,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in metrics], rotation=12, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("5-Fold mean +/- std")
        ax.set_title(f"Inference-time Missingness: {model}")
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
        path = out / f"expB_coverage_{MODEL_SHORT[model]}.png"
        _save(fig, path)
        paths.append(path)
    return paths


def generate_c() -> list[Path]:
    out = OUTPUTS / "exp_c_extra"
    summary = pd.read_csv(out / "expC_threshold_summary.csv")
    selections = pd.read_csv(out / "expC_threshold_selections.csv")
    paths = []
    for model in ["logistic_regression", "linear_svc"]:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
        metrics = ["f1", "precision", "recall"]
        x = np.arange(len(metrics))
        width = 0.36
        for offset, mode, color in (
            (-width / 2, "default_0.5", "#7f8c8d"),
            (width / 2, "inner_cv_tuned", "#d95f02"),
        ):
            rows = (
                summary[
                    (summary["model"] == model)
                    & (summary["threshold_mode"] == mode)
                ]
                .set_index("metric")
                .reindex(metrics)
            )
            axes[0].bar(
                x + offset,
                rows["cv_mean"],
                width,
                yerr=rows["cv_std"],
                capsize=3,
                label=mode,
                color=color,
            )
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(["F1", "Precision", "Recall"])
        axes[0].set_ylim(0, 1.05)
        axes[0].set_ylabel("Outer 5-Fold mean +/- std")
        axes[0].set_title("Outer-fold performance")
        axes[0].legend(fontsize=8)

        selected = selections[selections["model"] == model].sort_values("outer_fold")
        axes[1].plot(
            selected["outer_fold"],
            selected["selected_threshold"],
            marker="o",
            color="#d95f02",
            label="inner-CV tuned",
        )
        axes[1].axhline(0.5, color="#7f8c8d", linestyle="--", label="default 0.5")
        axes[1].set_xticks(range(5))
        axes[1].set_ylim(0, 0.65)
        axes[1].set_xlabel("Outer fold")
        axes[1].set_ylabel("Selected threshold")
        axes[1].set_title("Threshold selected without outer test")
        axes[1].legend(fontsize=8)
        fig.suptitle(f"Nested Threshold Selection: {model}")
        path = out / f"expC_threshold_{MODEL_SHORT[model]}.png"
        _save(fig, path)
        paths.append(path)
    return paths


def generate_d() -> list[Path]:
    out = OUTPUTS / "exp_d_extra"
    summary = pd.read_csv(out / "expD_ablation_summary.csv")
    effects = pd.read_csv(out / "expD_ablation_effect_summary.csv")
    vocab = pd.read_csv(out / "expD_ablation_vocabulary.csv")
    timing = pd.read_csv(out / "expD_ablation_preprocessing_time.csv")
    paths = []
    conditions = ["D0", "D1", "D2", "D3"]
    effect_order = [
        "cleaning_effect_simple_D1_minus_D0",
        "cleaning_effect_advanced_D3_minus_D2",
        "analyzer_effect_raw_D2_minus_D0",
        "analyzer_effect_clean_D3_minus_D1",
    ]
    effect_labels = ["clean/simple", "clean/advanced", "analyzer/raw", "analyzer/clean"]
    for model in MODEL_ORDER:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
        rows = (
            summary[
                (summary["model"] == model) & (summary["metric"] == "f1_macro")
            ]
            .set_index("condition")
            .reindex(conditions)
        )
        axes[0].bar(
            conditions,
            rows["cv_mean"],
            yerr=rows["cv_std"],
            capsize=3,
            color=["#4c78a8", "#72b7b2", "#f58518", "#e45756"],
        )
        axes[0].set_ylim(0.7, 1.0)
        axes[0].set_ylabel("macro-F1 (5-Fold mean +/- std)")
        axes[0].set_title("D0-D3 performance")

        effect_rows = effects[effects["model"] == model].set_index("effect").reindex(
            effect_order
        )
        axes[1].bar(
            effect_labels,
            effect_rows["mean_diff"],
            yerr=effect_rows["std_diff"],
            capsize=3,
            color=["#72b7b2", "#54a24b", "#f58518", "#e45756"],
        )
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_ylabel("Paired macro-F1 difference")
        axes[1].set_title("Factor effects")
        axes[1].tick_params(axis="x", rotation=20)
        fig.suptitle(f"Cleaning x Analyzer Ablation: {model}")
        path = out / f"expD_ablation_{MODEL_SHORT[model]}.png"
        _save(fig, path)
        paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    axes[0].bar(
        vocab["condition"],
        vocab["vocabulary_size"],
        color=["#4c78a8", "#72b7b2", "#f58518", "#e45756"],
    )
    axes[0].set_ylabel("Vocabulary size (min_df=2)")
    axes[0].set_title("Vocabulary size")
    axes[1].bar(
        timing["condition"],
        timing["tokenization_seconds"],
        label="tokenization",
        color="#4c78a8",
    )
    axes[1].bar(
        timing["condition"],
        timing["cleaning_seconds"],
        bottom=timing["tokenization_seconds"],
        label="cleaning",
        color="#f58518",
    )
    axes[1].set_ylabel("Seconds (single run)")
    axes[1].set_title("Deterministic preprocessing cost")
    axes[1].legend()
    path = out / "expD_ablation_resources.png"
    _save(fig, path)
    paths.append(path)
    return paths


def main() -> None:
    paths = generate_a() + generate_b() + generate_c() + generate_d()
    for path in paths:
        print(path.relative_to(TASK9_ROOT))
    print(f"generated {len(paths)} individual figures")


if __name__ == "__main__":
    main()
