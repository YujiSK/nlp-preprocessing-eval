"""実験D（発展）: クレンジング有無×形態素解析器の2×2アブレーション。

D0: raw       × MeCab/IPAdic
D1: neologdn  × MeCab/IPAdic
D2: raw       × Sudachi core/Mode C
D3: neologdn  × Sudachi core/Mode C

既存の実験D成果物・最終レポートには変更を加えず、Appendix用成果物だけを
``outputs/exp_d_extra/`` に保存する。
"""

from __future__ import annotations

import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neologdn
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sudachipy import dictionary as sudachi_dictionary
from sudachipy import tokenizer as sudachi_tokenizer

TASK9_ROOT = Path(__file__).resolve().parents[2]
if str(TASK9_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK9_ROOT))

from src.experiments.evaluation import evaluate_pipeline_cv, summarize_cv
from src.experiments.models import MODEL_ORDER, build_model
from src.experiments.preprocessing import (
    IpadicTokenizer,
    deduplicate_by_raw_text,
    load_livedoor_corpus,
)
from src.utils import RANDOM_STATE, get_outer_splits

OUTPUT_DIR = TASK9_ROOT / "outputs" / "exp_d_extra"
CORPUS_ROOT = TASK9_ROOT / "data_cache" / "text"
TFIDF_KWARGS = dict(
    tokenizer=str.split,
    token_pattern=None,
    lowercase=False,
    min_df=2,
    max_df=0.95,
    max_features=30000,
)
CONDITION_ORDER = ["D0", "D1", "D2", "D3"]
CONDITION_META = {
    "D0": {"cleaning": "none", "analyzer": "mecab_ipadic"},
    "D1": {"cleaning": "neologdn", "analyzer": "mecab_ipadic"},
    "D2": {"cleaning": "none", "analyzer": "sudachi_core_mode_c"},
    "D3": {"cleaning": "neologdn", "analyzer": "sudachi_core_mode_c"},
}
METRICS = {
    "accuracy": accuracy_score,
    "f1_macro": lambda yt, yp: f1_score(yt, yp, average="macro"),
    "f1_weighted": lambda yt, yp: f1_score(yt, yp, average="weighted"),
}


class RawSudachiTokenizer:
    """入力を再クレンジングせず、Sudachi core/Mode Cだけを適用する。"""

    def __init__(self) -> None:
        self._tokenizer = sudachi_dictionary.Dictionary(dict="core").create()
        self._mode = sudachi_tokenizer.Tokenizer.SplitMode.C

    def tokenize(self, text: str) -> str:
        return " ".join(
            morpheme.surface()
            for morpheme in self._tokenizer.tokenize(text, self._mode)
        )


def timed_map(series: pd.Series, fn) -> tuple[pd.Series, float]:
    start = time.perf_counter()
    result = series.apply(fn)
    return result, time.perf_counter() - start


def prepare_conditions():
    df = load_livedoor_corpus(CORPUS_ROOT)
    df, n_duplicates_removed = deduplicate_by_raw_text(df)

    cleaned, cleaning_seconds = timed_map(df["raw_text"], neologdn.normalize)
    ipadic = IpadicTokenizer()
    sudachi = RawSudachiTokenizer()

    tokenized: dict[str, pd.Series] = {}
    tokenized["D0"], d0_tokenize = timed_map(df["raw_text"], ipadic.tokenize)
    tokenized["D1"], d1_tokenize = timed_map(cleaned, ipadic.tokenize)
    tokenized["D2"], d2_tokenize = timed_map(df["raw_text"], sudachi.tokenize)
    tokenized["D3"], d3_tokenize = timed_map(cleaned, sudachi.tokenize)

    tokenize_times = {
        "D0": d0_tokenize,
        "D1": d1_tokenize,
        "D2": d2_tokenize,
        "D3": d3_tokenize,
    }
    timing_rows = []
    for condition in CONDITION_ORDER:
        uses_cleaning = CONDITION_META[condition]["cleaning"] == "neologdn"
        cleaning_component = cleaning_seconds if uses_cleaning else 0.0
        timing_rows.append(
            {
                "condition": condition,
                **CONDITION_META[condition],
                "n_documents": len(df),
                "n_duplicates_removed": n_duplicates_removed,
                "cleaning_seconds": cleaning_component,
                "tokenization_seconds": tokenize_times[condition],
                "total_preprocessing_seconds": cleaning_component
                + tokenize_times[condition],
                "timing_repeats": 1,
            }
        )
    return df, tokenized, pd.DataFrame(timing_rows)


def pipeline_factory(model_name: str):
    def factory():
        return Pipeline(
            [
                ("tfidf", TfidfVectorizer(**TFIDF_KWARGS)),
                ("model", build_model(model_name)),
            ]
        )

    return factory


def evaluate_conditions(tokenized, y, outer_splits) -> pd.DataFrame:
    records: list[dict] = []
    for condition in CONDITION_ORDER:
        X = tokenized[condition].to_numpy()
        for model_name in MODEL_ORDER:
            condition_records = evaluate_pipeline_cv(
                experiment="D_ablation",
                condition=condition,
                model_name=model_name,
                pipeline_factory=pipeline_factory(model_name),
                X=X,
                y=y,
                outer_splits=outer_splits,
                metrics=METRICS,
            )
            for record in condition_records:
                record.update(CONDITION_META[condition])
            records.extend(condition_records)
    return pd.DataFrame(records)


def describe_texts(tokenized) -> tuple[pd.DataFrame, pd.DataFrame]:
    vocab_rows = []
    token_rows = []
    vocab_kwargs = {k: v for k, v in TFIDF_KWARGS.items() if k != "max_features"}
    for condition in CONDITION_ORDER:
        texts = tokenized[condition]
        vectorizer = TfidfVectorizer(**vocab_kwargs)
        vectorizer.fit(texts)
        vocab_rows.append(
            {
                "condition": condition,
                **CONDITION_META[condition],
                "vocabulary_size": len(vectorizer.vocabulary_),
            }
        )
        lengths = texts.str.split().apply(len)
        token_rows.append(
            {
                "condition": condition,
                **CONDITION_META[condition],
                "n_empty": int((lengths == 0).sum()),
                "min_tokens": int(lengths.min()),
                "median_tokens": float(lengths.median()),
                "max_tokens": int(lengths.max()),
            }
        )
    return pd.DataFrame(vocab_rows), pd.DataFrame(token_rows)


def compute_factor_effects(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    macro = metrics[metrics["metric"] == "f1_macro"].pivot(
        index=["model", "fold"], columns="condition", values="value"
    )
    contrasts = {
        "cleaning_effect_simple_D1_minus_D0": macro["D1"] - macro["D0"],
        "cleaning_effect_advanced_D3_minus_D2": macro["D3"] - macro["D2"],
        "analyzer_effect_raw_D2_minus_D0": macro["D2"] - macro["D0"],
        "analyzer_effect_clean_D3_minus_D1": macro["D3"] - macro["D1"],
    }
    rows = []
    for effect, values in contrasts.items():
        frame = values.rename("f1_macro_diff").reset_index()
        frame["effect"] = effect
        rows.append(frame)
    fold_effects = pd.concat(rows, ignore_index=True)
    summary = (
        fold_effects.groupby(["model", "effect"])["f1_macro_diff"]
        .agg(
            mean_diff="mean",
            std_diff="std",
            n_outer_folds="count",
            n_improved=lambda s: int((s > 0).sum()),
            n_worsened=lambda s: int((s < 0).sum()),
        )
        .reset_index()
    )
    return fold_effects, summary


def plot_results(
    summary: pd.DataFrame,
    vocab: pd.DataFrame,
    timing: pd.DataFrame,
    effects: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    colors = ["#4c78a8", "#72b7b2", "#f58518", "#e45756"]

    f1_rows = summary[summary["metric"] == "f1_macro"]
    x = np.arange(len(MODEL_ORDER))
    width = 0.2
    for index, condition in enumerate(CONDITION_ORDER):
        values = (
            f1_rows[f1_rows["condition"] == condition]
            .set_index("model")
            .reindex(MODEL_ORDER)
        )
        axes[0, 0].bar(
            x + (index - 1.5) * width,
            values["cv_mean"],
            width,
            yerr=values["cv_std"],
            capsize=2,
            color=colors[index],
            label=condition,
        )
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(MODEL_ORDER, rotation=20)
    axes[0, 0].set_ylim(0.7, 1.0)
    axes[0, 0].set_ylabel("macro-F1 (5-Fold mean +/- std)")
    axes[0, 0].set_title("Predictive performance")
    axes[0, 0].legend()

    axes[0, 1].bar(vocab["condition"], vocab["vocabulary_size"], color=colors)
    axes[0, 1].set_ylabel("Vocabulary size (min_df=2)")
    axes[0, 1].set_title("Vocabulary size")
    axes[0, 1].grid(axis="y", alpha=0.2)

    axes[1, 0].bar(
        timing["condition"],
        timing["tokenization_seconds"],
        label="tokenization",
        color="#4c78a8",
    )
    axes[1, 0].bar(
        timing["condition"],
        timing["cleaning_seconds"],
        bottom=timing["tokenization_seconds"],
        label="cleaning",
        color="#f58518",
    )
    axes[1, 0].set_ylabel("Seconds (single run, 7,361 docs)")
    axes[1, 0].set_title("Deterministic preprocessing cost")
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.2)

    effect_order = [
        "cleaning_effect_simple_D1_minus_D0",
        "cleaning_effect_advanced_D3_minus_D2",
        "analyzer_effect_raw_D2_minus_D0",
        "analyzer_effect_clean_D3_minus_D1",
    ]
    effect_labels = ["clean/simple", "clean/advanced", "analyzer/raw", "analyzer/clean"]
    effect_x = np.arange(len(effect_order))
    model_width = 0.18
    for index, model_name in enumerate(MODEL_ORDER):
        values = (
            effects[effects["model"] == model_name]
            .set_index("effect")
            .reindex(effect_order)
        )
        axes[1, 1].bar(
            effect_x + (index - 1.5) * model_width,
            values["mean_diff"],
            model_width,
            yerr=values["std_diff"],
            capsize=2,
            label=model_name,
        )
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_xticks(effect_x)
    axes[1, 1].set_xticklabels(effect_labels, rotation=15)
    axes[1, 1].set_ylabel("macro-F1 paired difference")
    axes[1, 1].set_title("Factor effects by model")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Experiment D Appendix: 2x2 Cleaning x Analyzer Ablation")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def write_appendix(
    summary: pd.DataFrame,
    vocab: pd.DataFrame,
    timing: pd.DataFrame,
    effects: pd.DataFrame,
    out_path: Path,
) -> None:
    lines = [
        "# APPENDIX_EXP_D_ABLATION — 実験D（発展）：D0〜D3アブレーション",
        "",
        "## 設計",
        "",
        "| 条件 | クレンジング | 形態素解析 |",
        "|:--|:--|:--|",
        "| D0 | なし | MeCab＋IPAdic |",
        "| D1 | neologdn | MeCab＋IPAdic |",
        "| D2 | なし | Sudachi core＋Mode C |",
        "| D3 | neologdn | Sudachi core＋Mode C |",
        "",
        "livedoor News Corpusの重複除去後7,361記事を使用した。既存実験Dと同じ外側5-Fold、TF-IDF設定、4モデル、乱数seedを全条件で共用し、TF-IDFは各学習Fold内だけでfitした。クレンジングと解析器以外の条件を固定し、同一Foldのペア差として要因効果を算出した。",
        "",
        "## macro-F1",
        "",
        "| モデル | D0 | D1 | D2 | D3 |",
        "|:--|--:|--:|--:|--:|",
    ]

    for model_name in MODEL_ORDER:
        values = []
        for condition in CONDITION_ORDER:
            row = summary[
                (summary["model"] == model_name)
                & (summary["condition"] == condition)
                & (summary["metric"] == "f1_macro")
            ].iloc[0]
            values.append(f"{row['cv_mean']:.4f} ± {row['cv_std']:.4f}")
        lines.append(f"| {model_name} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## 語彙数・決定論的前処理コスト",
            "",
            "| 条件 | 語彙数 | クレンジング秒 | 解析秒 | 合計秒 |",
            "|:--|--:|--:|--:|--:|",
        ]
    )
    for condition in CONDITION_ORDER:
        vocab_row = vocab[vocab["condition"] == condition].iloc[0]
        time_row = timing[timing["condition"] == condition].iloc[0]
        lines.append(
            f"| {condition} | {int(vocab_row['vocabulary_size']):,} "
            f"| {time_row['cleaning_seconds']:.3f} "
            f"| {time_row['tokenization_seconds']:.3f} "
            f"| {time_row['total_preprocessing_seconds']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 要因別macro-F1差",
            "",
            "| モデル | Cleaning: simple D1−D0 | Cleaning: advanced D3−D2 | Analyzer: raw D2−D0 | Analyzer: cleaned D3−D1 |",
            "|:--|--:|--:|--:|--:|",
        ]
    )
    effect_order = [
        "cleaning_effect_simple_D1_minus_D0",
        "cleaning_effect_advanced_D3_minus_D2",
        "analyzer_effect_raw_D2_minus_D0",
        "analyzer_effect_clean_D3_minus_D1",
    ]
    for model_name in MODEL_ORDER:
        model_effects = effects[effects["model"] == model_name].set_index("effect")
        values = [
            f"{model_effects.loc[effect, 'mean_diff']:+.4f}"
            for effect in effect_order
        ]
        lines.append(f"| {model_name} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "![D0〜D3比較](expD_ablation_comparison.png)",
            "",
            "## 結果の要約",
            "",
            "クレンジング効果は、MeCab/IPAdic条件ではLinear SVC（+0.0001）を除く3モデルでmacro-F1が低下し、Sudachi条件では4モデルすべてで低下した。最大低下はMeCab/IPAdic＋k-NNの−0.0078だった。解析器効果は−0.0022〜+0.0052の範囲でモデル・クレンジング条件により方向が異なり、一貫した改善は観測されなかった。",
            "",
            "MeCab/IPAdicではneologdn適用前後の語彙数がともに42,123だった。Sudachiでは47,222から46,936へ286語（0.61%）減少したが、MeCab/IPAdicより約4,800語多かった。単一実行の合計前処理時間はD0 5.94秒、D1 10.23秒、D2 12.79秒、D3 16.44秒であり、本データ・実装では高度解析とクレンジングはいずれも処理コストを増加させた。",
            "",
            "## 解釈上の注意",
            "",
            "2×2比較により、クレンジング効果は同じ解析器内（D1−D0、D3−D2）、解析器効果は同じクレンジング条件内（D2−D0、D3−D1）で評価した。差は同一外側Foldの記述的ペア差であり、CV Foldを独立標本とみなす有意差検定は行っていない。",
            "",
            "語彙数は全データに記述統計としてTF-IDF（min_df=2、max_features制限なし）をfitした値で、性能評価には使用していない。前処理時間は単一実行の参考値であり、クレンジング済みテキストをD1/D3で共用して計算した実測時間を、それぞれのEnd-to-End想定コストへ加算した。CPU負荷等による変動を含む。",
            "",
            "## 再現性",
            "",
            f"- 実行日時: {datetime.now().astimezone().isoformat(timespec='seconds')}",
            f"- Python: {sys.version.split()[0]}",
            f"- scikit-learn: {sklearn.__version__}",
            f"- OS: {platform.platform()}",
            f"- outer folds: 5 / seed: {RANDOM_STATE}",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, tokenized, timing = prepare_conditions()
    y, _ = pd.factorize(df["category"], sort=True)
    outer_splits = get_outer_splits(np.arange(len(df)), y)

    vocab, token_stats = describe_texts(tokenized)
    metrics = evaluate_conditions(tokenized, y, outer_splits)
    summary = summarize_cv(metrics.to_dict("records"))
    fold_effects, effect_summary = compute_factor_effects(metrics)

    metrics.to_csv(OUTPUT_DIR / "expD_ablation_fold_metrics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "expD_ablation_summary.csv", index=False)
    vocab.to_csv(OUTPUT_DIR / "expD_ablation_vocabulary.csv", index=False)
    timing.to_csv(OUTPUT_DIR / "expD_ablation_preprocessing_time.csv", index=False)
    token_stats.to_csv(OUTPUT_DIR / "expD_ablation_token_stats.csv", index=False)
    fold_effects.to_csv(OUTPUT_DIR / "expD_ablation_fold_effects.csv", index=False)
    effect_summary.to_csv(OUTPUT_DIR / "expD_ablation_effect_summary.csv", index=False)
    plot_results(
        summary,
        vocab,
        timing,
        effect_summary,
        OUTPUT_DIR / "expD_ablation_comparison.png",
    )
    write_appendix(
        summary,
        vocab,
        timing,
        effect_summary,
        OUTPUT_DIR / "APPENDIX_EXP_D_ABLATION.md",
    )

    print("=== Experiment D advanced: preprocessing ===")
    print(timing.to_string(index=False))
    print("\n=== Vocabulary ===")
    print(vocab.to_string(index=False))
    print("\n=== macro-F1 ===")
    print(
        summary[summary["metric"] == "f1_macro"][
            ["condition", "model", "cv_mean", "cv_std"]
        ].to_string(index=False)
    )
    print("\n=== Factor effects ===")
    print(effect_summary.to_string(index=False))
    print(f"\nSaved Appendix artifacts to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
