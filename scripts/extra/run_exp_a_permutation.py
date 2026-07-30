"""実験A（発展）: 検証Fold上のPermutation Importance比較。

既存の実験A成果物・最終レポートには変更を加えず、Appendix用成果物だけを
``outputs/exp_a_extra/`` に保存する。
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
from sklearn.datasets import load_breast_cancer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TASK9_ROOT = Path(__file__).resolve().parents[2]
if str(TASK9_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK9_ROOT))

from src.experiments.models import MODEL_ORDER, build_model
from src.utils import RANDOM_STATE, get_outer_splits

OUTPUT_DIR = TASK9_ROOT / "outputs" / "exp_a_extra"
N_REPEATS = 30
SCORING = "accuracy"


def build_pipeline(model_name: str, standardize: bool) -> Pipeline:
    steps = []
    if standardize:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", build_model(model_name)))
    return Pipeline(steps)


def calculate_importances(X, y, feature_names: list[str]) -> pd.DataFrame:
    """共通5-Foldの各検証FoldでPIを算出し、repeat単位のLong形式で返す。"""
    records: list[dict] = []
    outer_splits = get_outer_splits(X, y)
    conditions = (("before", False), ("after", True))

    for condition, standardize in conditions:
        for model_name in MODEL_ORDER:
            for fold, (train_idx, test_idx) in enumerate(outer_splits):
                pipeline = build_pipeline(model_name, standardize)
                pipeline.fit(X[train_idx], y[train_idx])
                baseline_accuracy = pipeline.score(X[test_idx], y[test_idx])
                result = permutation_importance(
                    pipeline,
                    X[test_idx],
                    y[test_idx],
                    scoring=SCORING,
                    n_repeats=N_REPEATS,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                )
                for feature_idx, feature in enumerate(feature_names):
                    for repeat, importance in enumerate(result.importances[feature_idx]):
                        records.append(
                            {
                                "experiment": "A",
                                "condition": condition,
                                "model": model_name,
                                "fold": fold,
                                "repeat": repeat,
                                "feature": feature,
                                "importance": importance,
                                "baseline_accuracy": baseline_accuracy,
                                "scoring": SCORING,
                                "seed": RANDOM_STATE,
                            }
                        )
    return pd.DataFrame(records)


def summarize(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = (
        records.groupby(["condition", "model", "feature"])["importance"]
        .agg(pi_mean="mean", pi_std="std", n_observations="count")
        .reset_index()
    )

    fold_means = (
        records.groupby(["condition", "model", "fold", "feature"], as_index=False)["importance"]
        .mean()
    )
    fold_means["rank"] = fold_means.groupby(["condition", "model", "fold"])[
        "importance"
    ].rank(method="min", ascending=False)
    top_counts = (
        fold_means.assign(is_top10=fold_means["rank"] <= 10)
        .groupby(["condition", "model", "feature"], as_index=False)["is_top10"]
        .sum()
        .rename(columns={"is_top10": "top10_fold_count"})
    )
    summary = summary.merge(top_counts, on=["condition", "model", "feature"], how="left")

    paired = records.pivot(
        index=["model", "fold", "repeat", "feature"],
        columns="condition",
        values="importance",
    ).reset_index()
    paired["after_minus_before"] = paired["after"] - paired["before"]
    diff = (
        paired.groupby(["model", "feature"])["after_minus_before"]
        .agg(mean_diff="mean", std_diff="std", n_pairs="count")
        .reset_index()
    )
    return summary, fold_means, diff


def plot_comparison(summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    for ax, model_name in zip(axes.flat, MODEL_ORDER):
        model_rows = summary[summary["model"] == model_name]
        selected = (
            model_rows.groupby("feature")["pi_mean"].max().nlargest(10).index.tolist()
        )
        pivot = (
            model_rows[model_rows["feature"].isin(selected)]
            .pivot(index="feature", columns="condition", values="pi_mean")
            .reindex(selected)
            .sort_values("after")
        )
        errors = (
            model_rows[model_rows["feature"].isin(pivot.index)]
            .pivot(index="feature", columns="condition", values="pi_std")
            .reindex(index=pivot.index, columns=pivot.columns)
        )
        pivot.plot.barh(
            ax=ax,
            xerr=errors,
            capsize=2,
            color={"before": "#7f8c8d", "after": "#2878b5"},
        )
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_title(model_name)
        ax.set_xlabel("Mean decrease in validation accuracy")
        ax.set_ylabel("")
        ax.legend(title="")
    fig.suptitle(
        "Experiment A (Appendix): Permutation Importance on Held-out CV Folds",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def write_appendix(summary: pd.DataFrame, diff: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# Appendix A — 実験A（発展）：Permutation Importance",
        "",
        "## 方法",
        "",
        "- Breast Cancerデータ（569件、30特徴量）を使用した。",
        "- 本文と同じ `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` の分割を、全条件・全モデルで共用した。",
        "- 各モデルを学習Foldのみでfitし、未使用の検証Fold上でAccuracyを評価した。",
        f"- 各特徴量を検証Fold内で{N_REPEATS}回並べ替え、Accuracy低下量をPermutation Importanceとした。",
        "- StandardScalerはAfter条件のみPipeline内でfitした。並べ替えはPipelineへの入力列に対して行った。",
        "",
        "## 結果",
        "",
        "| モデル | Before上位特徴量（平均PI） | After上位特徴量（平均PI） | 最大のBefore→After差 |",
        "|:--|:--|:--|:--|",
    ]
    for model_name in MODEL_ORDER:
        model_summary = summary[summary["model"] == model_name]
        before = model_summary[model_summary["condition"] == "before"].nlargest(1, "pi_mean").iloc[0]
        after = model_summary[model_summary["condition"] == "after"].nlargest(1, "pi_mean").iloc[0]
        largest_diff = diff[diff["model"] == model_name].iloc[
            diff[diff["model"] == model_name]["mean_diff"].abs().argmax()
        ]
        lines.append(
            f"| {model_name} | {before['feature']} ({before['pi_mean']:.4f}) "
            f"| {after['feature']} ({after['pi_mean']:.4f}) "
            f"| {largest_diff['feature']} ({largest_diff['mean_diff']:+.4f}) |"
        )
    lines.extend(
        [
            "",
            "![Permutation Importance Before/After比較](expA_permutation_importance_comparison.png)",
            "",
            "## 解釈上の注意",
            "",
            "Permutation Importanceは未使用の検証Fold上で算出したため、係数の絶対値やRandom Forestの不純度ベース重要度よりモデル間比較に適する。一方、本データには相関の強い特徴量が複数あり、代替可能な特徴量を一つだけ並べ替えても予測性能が大きく低下しない場合がある。このため、値を各特徴量の独立した因果的寄与とは解釈しない。小さな負値は、有限標本と並べ替えによる変動の範囲で生じ得る。",
            "",
            "未標準化Logistic Regressionは5 Foldすべてで `max_iter=2000` に達する収束警告が発生した。その条件の重要度は未収束モデルに基づく記述値であり、Afterとの差に標準化だけでなく最適化の収束状態も反映され得る。既存の実験条件を維持するため、発展実験のみで反復回数やソルバーは変更していない。",
            "",
            "## 再現性",
            "",
            f"- 実行日時: {datetime.now().astimezone().isoformat(timespec='seconds')}",
            f"- Python: {sys.version.split()[0]}",
            f"- scikit-learn: {sklearn.__version__}",
            f"- OS: {platform.platform()}",
            f"- scoring: `{SCORING}` / repeats: {N_REPEATS} / seed: {RANDOM_STATE}",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_breast_cancer()
    X, y = data.data, data.target
    records = calculate_importances(X, y, list(data.feature_names))
    summary, fold_means, diff = summarize(records)

    records.to_csv(OUTPUT_DIR / "expA_permutation_importance_long.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "expA_permutation_importance_summary.csv", index=False)
    fold_means.to_csv(OUTPUT_DIR / "expA_permutation_importance_fold_means.csv", index=False)
    diff.to_csv(OUTPUT_DIR / "expA_permutation_importance_before_after_diff.csv", index=False)
    plot_comparison(summary, OUTPUT_DIR / "expA_permutation_importance_comparison.png")
    write_appendix(summary, diff, OUTPUT_DIR / "APPENDIX_EXP_A_PERMUTATION.md")

    print("=== Experiment A advanced: top permutation importance ===")
    print(
        summary.sort_values("pi_mean", ascending=False)
        .groupby(["condition", "model"], sort=False)
        .head(3)
        .sort_values(["model", "condition", "pi_mean"], ascending=[True, True, False])
        .to_string(index=False)
    )
    print(f"\nSaved Appendix artifacts to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
