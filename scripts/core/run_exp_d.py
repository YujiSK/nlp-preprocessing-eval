"""実験D: 日本語テキスト分類におけるBefore/After前処理パイプラインの比較（docs/execution_plan.md 3章「実験D」）。

データセット: livedoor News Corpus（9クラス、`data_cache/text/`に展開済み）。
Before: クレンジングなし＋IPA辞書（MeCab）＋TF-IDF
After : neologdnクレンジング＋Sudachi(Mode C, core辞書)＋TF-IDF
クレンジング有無・解析器・辞書が同時に変わるため、個別手法の効果ではなく
前処理パイプライン全体（Before/After）の比較として解釈する（計画書3章の注記）。
"""

from __future__ import annotations

import time
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import validation_curve
from sklearn.pipeline import Pipeline

from src.experiments.evaluation import evaluate_pipeline_cv, paired_fold_diff, summarize_cv
from src.experiments.explainability import extract_linear_coefficients, extract_tree_importances
from src.experiments.models import MODEL_ORDER, build_model
from src.experiments.preprocessing import (
    IpadicTokenizer,
    SudachiTokenizer,
    deduplicate_by_raw_text,
    load_livedoor_corpus,
)
from src.utils import RANDOM_STATE, ensure_output_dir, get_outer_cv, get_outer_splits, save_environment_info

EXPERIMENT = "D"
CORPUS_ROOT = TASK9_ROOT / "data_cache" / "text"
TFIDF_KWARGS = dict(tokenizer=str.split, token_pattern=None, lowercase=False, min_df=2, max_df=0.95, max_features=30000)

METRICS = {
    "accuracy": accuracy_score,
    "f1_macro": lambda yt, yp: f1_score(yt, yp, average="macro"),
    "f1_weighted": lambda yt, yp: f1_score(yt, yp, average="weighted"),
}

REPRESENTATIVE_CLASSES = ["sports-watch", "movie-enter", "it-life-hack"]


def load_and_prepare():
    df = load_livedoor_corpus(CORPUS_ROOT)
    df, n_duplicates_removed = deduplicate_by_raw_text(df)

    # メタデータリーク確認（必須）: カテゴリ名が本文中に literal に含まれていないか
    leak_counts = {
        category: int(df.loc[df["category"] == category, "raw_text"].str.contains(category, regex=False).sum())
        for category in df["category"].unique()
    }

    ipadic_tok = IpadicTokenizer()
    t0 = time.perf_counter()
    df["ipadic_tokens"] = df["raw_text"].apply(ipadic_tok.tokenize)
    ipadic_seconds = time.perf_counter() - t0

    sudachi_tok = SudachiTokenizer()
    t0 = time.perf_counter()
    df["cleaned_text"] = df["raw_text"].apply(sudachi_tok.clean)
    cleaning_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    df["sudachi_tokens"] = df["cleaned_text"].apply(sudachi_tok.tokenize)
    sudachi_seconds = time.perf_counter() - t0

    timing_info = dict(
        n_documents=len(df),
        n_duplicates_removed=n_duplicates_removed,
        ipadic_tokenize_seconds=ipadic_seconds,
        neologdn_clean_seconds=cleaning_seconds,
        sudachi_tokenize_seconds=sudachi_seconds,
    )
    return df, leak_counts, timing_info


def token_length_stats(series: pd.Series) -> dict:
    lengths = series.str.split().apply(len)
    return dict(
        n_empty=int((lengths == 0).sum()),
        min_tokens=int(lengths.min()),
        median_tokens=float(lengths.median()),
        max_tokens=int(lengths.max()),
    )


def pipeline_factory(model_name: str, condition: str):
    def factory():
        return Pipeline(
            [("tfidf", TfidfVectorizer(**TFIDF_KWARGS)), ("model", build_model(model_name))]
        )

    return factory


def run_cv(X_before, X_after, y, outer_splits):
    records: list[dict] = []
    for condition, X in [("before", X_before), ("after", X_after)]:
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
                )
            )
    return records


def plot_cv_score_bar(summary: pd.DataFrame, out_path):
    f1m = summary[summary["metric"] == "f1_macro"].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    models = MODEL_ORDER
    width = 0.35
    x = range(len(models))
    for offset, condition in zip([-width / 2, width / 2], ["before", "after"]):
        rows = f1m[f1m["condition"] == condition].set_index("model").reindex(models)
        ax.bar([i + offset for i in x], rows["cv_mean"], width=width, yerr=rows["cv_std"], capsize=4, label=condition)
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=20)
    ax.set_ylabel("macro-F1 (5-Fold CV mean +/- std)")
    ax.set_title("Experiment D: IPA+no cleaning (Before) vs NEologd/Sudachi+neologdn (After)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_param_tuning(X_after, y, outer_cv, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    c_range = [0.01, 0.1, 1, 10, 100]
    train_scores, test_scores = validation_curve(
        pipeline_factory("logistic_regression", "after")(),
        X_after,
        y,
        param_name="model__C",
        param_range=c_range,
        cv=outer_cv,
        scoring="f1_macro",
    )
    axes[0].plot(c_range, train_scores.mean(axis=1), marker="o", label="train")
    axes[0].plot(c_range, test_scores.mean(axis=1), marker="o", label="CV")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("C")
    axes[0].set_title("Logistic Regression: C (After)")
    axes[0].legend()

    depth_range = [3, 5, 10, 20]
    train_scores, test_scores = validation_curve(
        pipeline_factory("random_forest", "after")(),
        X_after,
        y,
        param_name="model__max_depth",
        param_range=depth_range,
        cv=outer_cv,
        scoring="f1_macro",
    )
    x = range(len(depth_range))
    axes[1].plot(x, train_scores.mean(axis=1), marker="o", label="train")
    axes[1].plot(x, test_scores.mean(axis=1), marker="o", label="CV")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels([str(d) for d in depth_range])
    axes[1].set_xlabel("max_depth")
    axes[1].set_title("Random Forest: max_depth (After)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    df, leak_counts, timing_info = load_and_prepare()

    codes, uniques = pd.factorize(df["category"], sort=True)
    y = codes
    class_names = list(uniques)

    X_before = df["ipadic_tokens"].to_numpy()
    X_after = df["sudachi_tokens"].to_numpy()

    outer_splits = get_outer_splits(np.arange(len(df)), y)
    outer_cv = get_outer_cv()

    exp_dir = ensure_output_dir(EXPERIMENT)

    # 空文書・トークン数統計（必須）
    token_stats = pd.DataFrame(
        [
            {"condition": "before", **token_length_stats(df["ipadic_tokens"])},
            {"condition": "after", **token_length_stats(df["sudachi_tokens"])},
        ]
    )
    token_stats.to_csv(exp_dir / "expD_token_stats.csv", index=False)

    # 語彙数の圧縮効果（記述統計。モデル評価には使わない。max_featuresで打ち切られると
    # 圧縮効果が見えなくなるため、ここではmax_featuresなしの生語彙数を数える）
    vocab_kwargs = {k: v for k, v in TFIDF_KWARGS.items() if k != "max_features"}
    vocab_sizes = {}
    for condition, X in [("before", X_before), ("after", X_after)]:
        vec = TfidfVectorizer(**vocab_kwargs)
        vec.fit(X)
        vocab_sizes[condition] = len(vec.vocabulary_)
    pd.DataFrame([vocab_sizes]).to_csv(exp_dir / "expD_vocab_size.csv", index=False)

    pd.DataFrame([timing_info]).to_csv(exp_dir / "expD_preprocessing_time.csv", index=False)
    pd.DataFrame([leak_counts]).T.reset_index().rename(
        columns={"index": "category", 0: "n_docs_containing_own_category_name"}
    ).to_csv(exp_dir / "expD_metadata_leak_check.csv", index=False)

    records = run_cv(X_before, X_after, y, outer_splits)
    summary = summarize_cv(records)
    diff = paired_fold_diff(records, before_label="before", after_label="after")

    pd.DataFrame(records).to_csv(exp_dir / "expD_metrics_summary.csv", index=False)
    summary.to_csv(exp_dir / "expD_metrics_summary_agg.csv", index=False)
    diff.to_csv(exp_dir / "expD_paired_fold_diff.csv", index=False)

    plot_cv_score_bar(summary, exp_dir / "expD_cv_score_bar.png")
    plot_param_tuning(X_after, y, outer_cv, exp_dir / "expD_param_tuning.png")

    # 説明性（発展寄り・記述的分析）: Afterパイプラインを全データで再学習し、代表クラスの上位語を抽出する
    after_logreg = pipeline_factory("logistic_regression", "after")()
    after_logreg.fit(X_after, y)
    feature_names = list(after_logreg.named_steps["tfidf"].get_feature_names_out())

    coef_rows = []
    for class_name in REPRESENTATIVE_CLASSES:
        class_index = class_names.index(class_name)
        coef_df = extract_linear_coefficients(after_logreg, feature_names, top_n=10, class_index=class_index)
        coef_df["class"] = class_name
        coef_rows.append(coef_df)
    pd.concat(coef_rows, ignore_index=True).to_csv(exp_dir / "expD_coefficients.csv", index=False)

    after_rf = pipeline_factory("random_forest", "after")()
    after_rf.fit(X_after, y)
    importance_df = extract_tree_importances(after_rf, feature_names, top_n=10)
    importance_df.to_csv(exp_dir / "expD_feature_importance.csv", index=False)

    save_environment_info(EXPERIMENT)

    print("=== Experiment D: preprocessing timing ===")
    print(timing_info)
    print("\n=== Experiment D: vocabulary size (before/after) ===")
    print(vocab_sizes)
    print("\n=== Experiment D: token length stats ===")
    print(token_stats.to_string(index=False))
    print("\n=== Experiment D: CV summary (f1_macro) ===")
    print(summary[summary["metric"] == "f1_macro"][["condition", "model", "cv_mean", "cv_std"]].to_string(index=False))
    print("\n=== Experiment D: paired fold diff (f1_macro) ===")
    print(diff[diff["metric"] == "f1_macro"].to_string(index=False))


if __name__ == "__main__":
    main()
