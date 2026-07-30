"""実験D（日本語テキスト分類）向けのコーパス読込・形態素解析ユーティリティ。

docs/execution_plan.md 3章「実験D」・4.1章（辞書固定の形態素解析は事前計算・キャッシュしてよい）に対応する。

- Before: クレンジングなし ＋ IPA辞書（MeCab）での形態素解析
- After : neologdnクレンジング ＋ Sudachi（Mode C, core辞書）での形態素解析

形態素解析器は本モジュール内で1回だけ生成し、再利用する（コスト計測4.6章の注意点に対応）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import ipadic
import MeCab
import neologdn
import pandas as pd
from sklearn.datasets import make_classification
from sudachipy import dictionary as sudachi_dictionary
from sudachipy import tokenizer as sudachi_tokenizer

from ..utils import RANDOM_STATE

_IGNORED_FILES = {"CHANGES.txt", "README.txt", "LICENSE.txt"}

# 記事末尾の関連リンク・関連記事一覧は、媒体名（例: smaxカテゴリの「エスマックス（S-MAX）」）を
# 含みメタデータリークになるため、本文から除去する（計画書3章「実験D」リーク確認）。
_FOOTER_MARKERS = ["■関連リンク", "■関連記事"]


def build_imbalanced_dataset():
    """実験Cと発展閾値実験で共有する再現可能な不均衡データを返す。"""
    return make_classification(
        n_samples=2000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        weights=[0.93, 0.07],
        flip_y=0.01,
        random_state=RANDOM_STATE,
    )


def _strip_footer(body: str) -> str:
    cut_positions = [body.find(marker) for marker in _FOOTER_MARKERS if marker in body]
    if not cut_positions:
        return body
    return body[: min(cut_positions)].rstrip()


def load_livedoor_corpus(root: Path) -> pd.DataFrame:
    """livedoor News Corpus（`text/`ディレクトリ）を読み込み、DataFrameとして返す。

    列: category, filename, url, date, title, body, raw_text
    raw_text はタイトル＋本文（原文）で、前処理変更後も再実行できるよう上書きしない（第7章）。
    """
    rows = []
    for category_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        category = category_dir.name
        for file_path in sorted(category_dir.glob("*.txt")):
            if file_path.name in _IGNORED_FILES:
                continue
            lines = file_path.read_text(encoding="utf-8").splitlines()
            if len(lines) < 4:
                continue
            url, date, title = lines[0], lines[1], lines[2]
            body = "\n".join(lines[3:]).strip()
            body = _strip_footer(body)
            raw_text = f"{title}\n{body}"
            rows.append(
                dict(
                    category=category,
                    filename=file_path.name,
                    url=url,
                    date=date,
                    title=title,
                    body=body,
                    raw_text=raw_text,
                )
            )
    return pd.DataFrame(rows)


def deduplicate_by_raw_text(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """raw_textの完全一致による重複記事を除去する（第3章「実験D」リーク確認）。"""
    hashes = df["raw_text"].apply(lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest())
    is_duplicate = hashes.duplicated(keep="first")
    removed = int(is_duplicate.sum())
    return df.loc[~is_duplicate].reset_index(drop=True), removed


class IpadicTokenizer:
    """クレンジングなし・IPA辞書によるBefore条件の形態素解析器。"""

    def __init__(self) -> None:
        self._tagger = MeCab.Tagger(ipadic.MECAB_ARGS)

    def tokenize(self, text: str) -> str:
        node = self._tagger.parseToNode(text)
        surfaces = []
        while node:
            if node.surface:
                surfaces.append(node.surface)
            node = node.next
        return " ".join(surfaces)


class SudachiTokenizer:
    """neologdnクレンジング＋Sudachi(Mode C, core辞書)によるAfter条件の形態素解析器。"""

    def __init__(self) -> None:
        self._tokenizer_obj = sudachi_dictionary.Dictionary(dict="core").create()
        self._mode = sudachi_tokenizer.Tokenizer.SplitMode.C

    def clean(self, text: str) -> str:
        return neologdn.normalize(text)

    def tokenize(self, text: str) -> str:
        cleaned = self.clean(text)
        return " ".join(m.surface() for m in self._tokenizer_obj.tokenize(cleaned, self._mode))
