"""実験B（発展）: 自然欠損を含むTitanicデータで推論Coverageを評価する。

既存レポート・既存実験B成果物には触れず、Appendix用成果物のみを
``outputs/exp_b_extra/`` に保存する。
"""

from __future__ import annotations

import json
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
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TASK9_ROOT = Path(__file__).resolve().parents[2]
if str(TASK9_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK9_ROOT))

from src.experiments.models import MODEL_ORDER, build_model
from src.utils import RANDOM_STATE, get_outer_splits

OUTPUT_DIR = TASK9_ROOT / "outputs" / "exp_b_extra"
DATA_HOME = TASK9_ROOT / "data_cache" / "openml"

NUMERIC_COLS = ["age", "sibsp", "parch", "fare"]
CATEGORICAL_COLS = ["pclass", "sex", "embarked"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


def load_dataset() -> tuple[pd.DataFrame, np.ndarray, dict]:
    """OpenML Titanic v1（data_id=40945）を読み込み、予測時点で利用可能な列だけを返す。"""
    dataset = fetch_openml(
        "titanic",
        version=1,
        as_frame=True,
        data_home=str(DATA_HOME),
    )
    frame = dataset.frame.copy()
    X = frame[FEATURE_COLS].copy()
    y = pd.to_numeric(frame["survived"], errors="raise").astype(int).to_numpy()
    metadata = {
        "dataset": "Titanic",
        "source": "OpenML",
        "openml_data_id": int(dataset.details["id"]),
        "openml_version": int(dataset.details["version"]),
        "source_url": f"https://www.openml.org/d/{dataset.details['id']}",
        "n_samples": len(X),
        "target": "survived",
        "features_used": FEATURE_COLS,
        "excluded_columns": [
            "name",
            "ticket",
            "cabin",
            "boat",
            "body",
            "home.dest",
        ],
        "exclusion_reason": (
            "識別性の高い列、欠損率が極端に高い列、および結果発生後の情報を除外"
        ),
        "missing_by_feature": {col: int(X[col].isna().sum()) for col in FEATURE_COLS},
    }
    return X, y, metadata


def build_preprocessor(impute: bool) -> ColumnTransformer:
    if impute:
        numeric_steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
        categorical_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ]
    else:
        numeric_steps = [("scaler", StandardScaler())]
        categorical_steps = [
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2))
        ]
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), NUMERIC_COLS),
            ("cat", Pipeline(categorical_steps), CATEGORICAL_COLS),
        ]
    )


def evaluate(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    records: list[dict] = []
    for fold, (train_idx, test_idx) in enumerate(get_outer_splits(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_complete = ~X_train.isna().any(axis=1)
        test_complete = (~X_test.isna().any(axis=1)).to_numpy()

        for condition, impute in (("before", False), ("after", True)):
            if condition == "before":
                fit_X = X_train.loc[train_complete]
                fit_y = y_train[train_complete.to_numpy()]
                predictable = test_complete
            else:
                fit_X = X_train
                fit_y = y_train
                predictable = np.ones(len(X_test), dtype=bool)

            for model_name in MODEL_ORDER:
                pipeline = Pipeline(
                    [
                        ("preprocessor", build_preprocessor(impute=impute)),
                        ("model", build_model(model_name)),
                    ]
                )
                pipeline.fit(fit_X, fit_y)
                y_pred = pipeline.predict(X_test.iloc[predictable])

                coverage = float(predictable.mean())
                correct = int((y_pred == y_test[predictable]).sum())
                base = {
                    "experiment": "B_coverage",
                    "condition": condition,
                    "model": model_name,
                    "fold": fold,
                    "seed": RANDOM_STATE,
                    "n_train_total": len(X_train),
                    "n_train_used": len(fit_X),
                    "n_test_total": len(X_test),
                    "n_predicted": int(predictable.sum()),
                    "n_abstained": int((~predictable).sum()),
                }
                metrics = {
                    "coverage": coverage,
                    "unavailable_rate": 1.0 - coverage,
                    "accuracy_predicted": accuracy_score(y_test[predictable], y_pred),
                    "f1_macro_predicted": f1_score(
                        y_test[predictable], y_pred, average="macro"
                    ),
                    # 予測不能を不正解相当として全リクエスト数で割る運用指標。
                    "correct_fraction_all": correct / len(X_test),
                }

                complete_pred = pipeline.predict(X_test.iloc[test_complete])
                metrics["accuracy_complete_rows"] = accuracy_score(
                    y_test[test_complete], complete_pred
                )
                metrics["f1_macro_complete_rows"] = f1_score(
                    y_test[test_complete], complete_pred, average="macro"
                )

                if condition == "after" and (~test_complete).any():
                    incomplete_pred = pipeline.predict(X_test.iloc[~test_complete])
                    metrics["accuracy_incomplete_rows"] = accuracy_score(
                        y_test[~test_complete], incomplete_pred
                    )
                    metrics["f1_macro_incomplete_rows"] = f1_score(
                        y_test[~test_complete], incomplete_pred, average="macro"
                    )

                for metric, value in metrics.items():
                    records.append({**base, "metric": metric, "value": value})
    return pd.DataFrame(records)


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    return (
        records.groupby(["condition", "model", "metric"])["value"]
        .agg(cv_mean="mean", cv_std="std", n_folds="count")
        .reset_index()
    )


def plot_results(summary: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("coverage", "Coverage"),
        ("accuracy_predicted", "Accuracy on predicted rows"),
        ("correct_fraction_all", "Correct predictions / all requests"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(MODEL_ORDER))
    width = 0.36
    colors = {"before": "#7f8c8d", "after": "#2878b5"}

    for ax, (metric, title) in zip(axes, metrics):
        rows = summary[summary["metric"] == metric]
        for offset, condition in ((-width / 2, "before"), (width / 2, "after")):
            values = rows[rows["condition"] == condition].set_index("model").reindex(
                MODEL_ORDER
            )
            ax.bar(
                x + offset,
                values["cv_mean"],
                width,
                yerr=values["cv_std"],
                capsize=3,
                color=colors[condition],
                label=condition,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER, rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.set_ylabel("5-Fold mean +/- std")
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend()
    fig.suptitle("Experiment B Appendix: Inference-time Missingness and Coverage")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def write_appendix(summary: pd.DataFrame, metadata: dict, out_path: Path) -> None:
    coverage_before = summary[
        (summary["condition"] == "before") & (summary["metric"] == "coverage")
    ]["cv_mean"].iloc[0]
    unavailable = 1.0 - coverage_before
    lines = [
        "# APPENDIX_EXP_B_COVERAGE — 実験B（発展）：推論時欠損と予測可能率",
        "",
        "## 目的とデータ",
        "",
        f"自然欠損を含むOpenML Titanic v1（data_id={metadata['openml_data_id']}、{metadata['n_samples']:,}件）を用い、推論要求に欠損が含まれる場合の行削除と補完を比較した。使用特徴量は `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked` である。結果発生後に判明する `boat`・`body`、識別性の高い `name`・`ticket`、欠損率の高い `cabin`・`home.dest` は除外した。",
        "",
        "## 評価方法",
        "",
        "- 本文と同じ5分割StratifiedKFold（shuffle=True, random_state=42）を全条件・全モデルで共用した。",
        "- Before（行削除）: 学習時は欠損行を除外し、推論時は7特徴量がすべて揃う行だけを予測した。欠損行は予測不能（abstain）として数えた。",
        "- After（補完）: 数値を学習Foldの中央値、カテゴリを学習Foldの最頻値で補完し、全推論行を予測した。補完器はPipeline内でFoldごとにfitした。",
        "- `accuracy_predicted`は実際に予測できた行だけの性能、`correct_fraction_all`は正解予測数を全推論要求数で割った運用指標であり、予測不能を不正解相当として扱う。",
        "",
        "## 結果",
        "",
        f"Beforeの平均Coverageは{coverage_before:.3f}、予測不可率は{unavailable:.3f}だった。Afterは全Fold・全モデルでCoverage 1.000となった。",
        "",
        "| モデル | Before: 予測行Accuracy | Before: 全要求中の正解割合 | After: 全行Accuracy | After: 欠損行Accuracy | 差（After−Before全要求） |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for model_name in MODEL_ORDER:
        def value(condition: str, metric: str) -> float:
            return float(
                summary[
                    (summary["condition"] == condition)
                    & (summary["model"] == model_name)
                    & (summary["metric"] == metric)
                ]["cv_mean"].iloc[0]
            )

        before_acc = value("before", "accuracy_predicted")
        before_all = value("before", "correct_fraction_all")
        after_acc = value("after", "accuracy_predicted")
        after_incomplete = value("after", "accuracy_incomplete_rows")
        lines.append(
            f"| {model_name} | {before_acc:.3f} | {before_all:.3f} "
            f"| {after_acc:.3f} | {after_incomplete:.3f} "
            f"| {after_acc - before_all:+.3f} |"
        )
    lines.extend(
        [
            "",
            "![Coverage比較](expB_coverage_comparison.png)",
            "",
            "## 解釈",
            "",
            "Beforeの予測行Accuracyは完全行に条件付けられた値であり、Afterの全行Accuracyと評価対象が異なるため、単独で優劣を判断できない。運用上はCoverageと全要求中の正解割合を併記する必要がある。AfterのCoverage 100%は「必ず出力する」ことを意味し、欠損行上の予測が同じ信頼性を持つことを保証しない。詳細CSVには完全行・欠損行別のAfter性能も保存した。",
            "",
            "欠損は自然発生しておりMCARとは限らない。またTitanicは歴史的な小規模データであるため、結果はCoverage指標の挙動確認であり、現行業務への性能一般化を目的としない。",
            "",
            "## 再現性",
            "",
            f"- Source: {metadata['source_url']}",
            f"- 実行日時: {datetime.now().astimezone().isoformat(timespec='seconds')}",
            f"- Python: {sys.version.split()[0]}",
            f"- scikit-learn: {sklearn.__version__}",
            f"- OS: {platform.platform()}",
            f"- seed: {RANDOM_STATE}",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y, metadata = load_dataset()
    records = evaluate(X, y)
    summary = summarize(records)
    missing_profile = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "missing_count": [int(X[c].isna().sum()) for c in FEATURE_COLS],
            "missing_rate": [float(X[c].isna().mean()) for c in FEATURE_COLS],
        }
    )

    records.to_csv(OUTPUT_DIR / "expB_coverage_fold_metrics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "expB_coverage_summary.csv", index=False)
    missing_profile.to_csv(OUTPUT_DIR / "expB_missingness_profile.csv", index=False)
    (OUTPUT_DIR / "expB_dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_results(summary, OUTPUT_DIR / "expB_coverage_comparison.png")
    write_appendix(summary, metadata, OUTPUT_DIR / "APPENDIX_EXP_B_COVERAGE.md")

    print("=== Experiment B advanced: Coverage summary ===")
    selected = summary[
        summary["metric"].isin(
            ["coverage", "accuracy_predicted", "correct_fraction_all"]
        )
    ]
    print(selected.to_string(index=False))
    print(f"\nSaved Appendix artifacts to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
