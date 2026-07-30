"""生成済みPDFを`pdfplumber`で解析し、`source_registry.json`と突き合わせてレイアウト不整合を検出する。

前提として「PDF解析のみで完璧な自動修復ができる」わけではない。本モジュールはテキスト内容の
ページ間一致検索（および画像・文字座標の補助情報）に基づくヒューリスティックであり、
定義した検査規則の範囲でのみ違反を検出する（layout_report.md の留意事項を参照）。

検出する違反種別:
  - orphan_heading            : 見出しと直後の本文ブロックが異なるページに分断されている
  - figure_caption_image_split: 図のキャプションが載っているページに画像が存在しない
  - short_block_split         : 短い（分割対象外の）コード／表が不自然にページ分断されている
  - overflow                  : 文字がページの印刷可能領域（マージン）を越えている
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

# Chromium(Skia)がPDF化する際に確認された、字形が近い別コードポイントへの置換の例。
# NFKC正規化で解決できないもの（Unicode上は非互換な文字同士）のみ手動で補う。
# 例: "・"(U+30FB 片仮名中点) が "‧"(U+2027 ハイフネーションポイント) として埋め込まれる。
_KNOWN_GLYPH_SUBSTITUTIONS = {
    "‧": "・",  # U+2027 HYPHENATION POINT -> U+30FB KATAKANA MIDDLE DOT
    "⻑": "長",  # U+2ED1 CJK RADICAL LONG ONE -> U+9577 (NFKCでは解決されない)
}

MM_TO_PT = 72 / 25.4
PAGE_MARGIN_TOP_BOTTOM_PT = 18 * MM_TO_PT
PAGE_MARGIN_LEFT_RIGHT_PT = 16 * MM_TO_PT
OVERFLOW_TOLERANCE_PT = 3.0

VIOLATION_WEIGHTS = {
    "orphan_heading": 2,
    "figure_caption_image_split": 3,
    "short_block_split": 2,
    "overflow": 1,
}


def _normalize(text: str) -> str:
    text = text or ""
    # NFKC正規化: 康熙部首ブロック等への字形置換（例: "⽅"U+2F45 -> "方"U+65B9）の大半を解消する
    text = unicodedata.normalize("NFKC", text)
    for wrong, right in _KNOWN_GLYPH_SUBSTITUTIONS.items():
        text = text.replace(wrong, right)
    return re.sub(r"\s+", "", text)


def _trigrams(text: str) -> set[str]:
    if len(text) < 3:
        return {text} if text else set()
    return {text[i : i + 3] for i in range(len(text) - 2)}


@dataclass
class PageData:
    number: int  # 1-indexed
    width: float
    height: float
    compact_text: str
    n_images: int
    words: list[dict] = field(default_factory=list)
    trigrams: frozenset[str] = field(default_factory=frozenset)


def extract_pages(pdf_path: Path) -> list[PageData]:
    pages: list[PageData] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            words = page.extract_words() or []
            compact = _normalize(text)
            pages.append(
                PageData(
                    number=i,
                    width=float(page.width),
                    height=float(page.height),
                    compact_text=compact,
                    n_images=len(page.images),
                    words=words,
                    trigrams=frozenset(_trigrams(compact)),
                )
            )
    return pages


def load_registry(registry_path: Path) -> list[dict]:
    return json.loads(Path(registry_path).read_text(encoding="utf-8"))


TEXT_MATCH_THRESHOLD = 0.6  # needleのトライグラムのうち、このFraction以上が一致すれば「そのページに存在する」とみなす
UNRESOLVED_WEIGHT = 2


def _word_locations(page: PageData, needle: str) -> list[float]:
    """短い見出しを含むPDF単語の上端座標を、ページ内の出現順で返す。"""
    needle_n = _normalize(needle)
    exact: list[float] = []
    partial: list[float] = []
    for word in page.words:
        word_n = _normalize(word.get("text", ""))
        if word_n == needle_n:
            exact.append(float(word["top"]))
        elif needle_n and needle_n in word_n:
            partial.append(float(word["top"]))
    return exact or partial


def _first_token_locations(page: PageData, needle: str) -> list[float]:
    """長文プローブの先頭文字列に対応する単語位置を返す。"""
    token = _normalize(needle)[:6]
    if not token:
        return []
    locations = []
    for word in page.words:
        word_n = _normalize(word.get("text", ""))
        if word_n.startswith(token[:3]) or token[:3] in word_n:
            locations.append(float(word["top"]))
    return locations


def _find_text_location(
    pages: list[PageData],
    needle: str,
    min_len: int = 4,
    min_page: int = 1,
    min_top: float = 0.0,
) -> tuple[int | None, bool, float | None, float]:
    """文書位置カーソルを考慮してテキストのページ・Y座標・一致度を返す。"""
    needle_n = _normalize(needle)
    if not needle_n:
        return None, False, None, 0.0

    candidates = [p for p in pages if p.number >= min_page]
    if not candidates:
        return None, False, None, 0.0

    # 2〜3文字の日本語見出しはトライグラムを作れないため、PDF単語との完全一致を優先する。
    if len(needle_n) < min_len:
        for page in candidates:
            floor = min_top if page.number == min_page else 0.0
            locations = [top for top in _word_locations(page, needle_n) if top >= floor]
            if locations:
                return page.number, False, min(locations), 1.0
        return None, False, None, 0.0

    needle_trigrams = _trigrams(needle_n)
    if not needle_trigrams:
        return None, False, None, 0.0

    scored: list[tuple[float, int, float | None]] = []
    for page in candidates:
        locations = _first_token_locations(page, needle_n)
        floor = min_top if page.number == min_page else 0.0
        valid_locations = [top for top in locations if top >= floor]
        if page.number == min_page and locations and not valid_locations:
            continue
        score = len(needle_trigrams & page.trigrams) / len(needle_trigrams)
        top = min(valid_locations) if valid_locations else None
        scored.append((score, page.number, top))

    best_score, best_page, best_top = max(scored, default=(0.0, None, None), key=lambda item: item[0])
    if best_page is not None and best_score >= TEXT_MATCH_THRESHOLD:
        return best_page, False, best_top, best_score

    # 隣接ページ結合はページ境界の分断検出専用。開始ページのYカーソルも尊重する。
    best_pair_page, best_pair_score = None, 0.0
    for i in range(len(candidates) - 1):
        first = candidates[i]
        if first.number == min_page:
            locations = _first_token_locations(first, needle_n)
            if locations and not any(top >= min_top for top in locations):
                continue
        combined = first.trigrams | candidates[i + 1].trigrams
        score = len(needle_trigrams & combined) / len(needle_trigrams)
        if score > best_pair_score:
            best_pair_page, best_pair_score = first.number, score

    if best_pair_page is not None and best_pair_score >= TEXT_MATCH_THRESHOLD:
        return best_pair_page, True, None, best_pair_score

    return None, False, None, max(best_score, best_pair_score)


def find_page_for_text(
    pages: list[PageData],
    needle: str,
    min_len: int = 4,
    min_page: int = 1,
    min_top: float = 0.0,
) -> tuple[int | None, bool]:
    """`needle`が最も高い割合で出現するページ番号 (1-indexed, `min_page`以降) を返す。閾値未満なら None。

    完全一致ではなくトライグラム（3文字連続部分列）の含有率によるファジーマッチングを用いる。
    これは、Chromium(Skia)がPDF化する際、CJK文字の一部を字形の近い別のUnicodeコードポイント
    （例: "方"(U+65B9) を康熙部首の"⽅"(U+2F45) に）へ置き換えて埋め込むケースが確認されており、
    厳密な文字列一致では大多数のテキストが「不一致」と誤判定されるため。

    `min_page` は、文書順に処理していく呼び出し側（`resolve_registry_pages`）が「これより前のページは
    探索しない」という下限を渡すためのもの。【事実】【メカニズム】等、章をまたいで同一文言が繰り返される
    見出しをトライグラムだけで一意に識別できないための対策。

    戻り値の2要素目は「単独ページでは閾値に届かず、隣接ページ結合でのみ閾値を満たした
    （ページ境界を跨いで分断されている可能性がある）」ことを示すフラグ。
    """
    page, split, _, _ = _find_text_location(
        pages,
        needle,
        min_len=min_len,
        min_page=min_page,
        min_top=min_top,
    )
    return page, split


def resolve_registry_pages(pages: list[PageData], registry: list[dict]) -> dict[str, dict]:
    """`registry`を文書出現順に走査し、各要素（および比較用の付随テキスト）のページ番号を解決する。

    見出しラベルの反復（【事実】等）や表・コードの識別に、単純な全文検索では位置を一意に定められない
    ため、「1つ前に解決できた要素のページ以降のみを探索する」というカーソルを進めながら解決する。
    registry は `report_build.py` で文書順に構築されているため、この前提が成り立つ。
    """
    resolved: dict[str, dict] = {}
    cursor_page = 1
    cursor_top = 0.0

    for entry in registry:
        probes: list[tuple[str, str]] = []
        if entry["type"] == "heading":
            probes.append(("self", entry["text"]))
            if entry.get("next_block_text"):
                probes.append(("next", entry["next_block_text"]))
        elif entry["type"] == "figure":
            probes.append(("self", entry["caption"]))
        elif entry["type"] == "table":
            probes.append(("first", entry["first_row_text"]))
            if entry["last_row_text"] and entry["last_row_text"] != entry["first_row_text"]:
                probes.append(("last", entry["last_row_text"]))
        elif entry["type"] == "code":
            probes.append(("first", entry["first_line"]))
            if entry["last_line"] and entry["last_line"] != entry["first_line"]:
                probes.append(("last", entry["last_line"]))

        entry_result: dict[str, dict] = {}
        local_page = cursor_page
        local_top = cursor_top
        for probe_name, text in probes:
            page, split, top, score = _find_text_location(
                pages,
                text,
                min_page=local_page,
                min_top=local_top,
            )
            entry_result[probe_name] = {
                "page": page,
                "top": top,
                "match_score": score,
                "split_across_pages": split,
            }
            if page is not None:
                if page > local_page:
                    local_top = 0.0
                local_page = page
                if top is not None:
                    local_top = top + 0.1

        resolved[entry["id"]] = entry_result

        anchor = entry_result.get("self") or entry_result.get("first")
        if anchor and anchor["page"] is not None:
            if anchor["page"] > cursor_page:
                cursor_top = 0.0
            cursor_page = anchor["page"]
            if anchor.get("top") is not None:
                cursor_top = anchor["top"] + 0.1

    return resolved


def _word_page_position_ratio(pages: list[PageData], page_number: int, needle: str) -> float | None:
    """`needle`の先頭語がページ内のどの高さ（0=上端,1=下端）にあるかを概算する。"""
    page = next((p for p in pages if p.number == page_number), None)
    if page is None or not page.words:
        return None
    first_token = _normalize(needle)[:6]
    for w in page.words:
        if first_token and _normalize(w["text"]).startswith(first_token[:3]):
            return w["top"] / page.height if page.height else None
    return None


def detect_orphan_headings(
    pages: list[PageData], registry: list[dict], resolved: dict[str, dict]
) -> tuple[list[dict], list[dict]]:
    violations: list[dict] = []
    unresolved: list[dict] = []

    for entry in registry:
        if entry["type"] != "heading":
            continue
        if not entry.get("next_block_text"):
            continue  # 文書末尾の見出し等、比較対象が存在しない

        probes = resolved.get(entry["id"], {})
        heading_page = probes.get("self", {}).get("page")
        next_page = probes.get("next", {}).get("page")

        if heading_page is None or next_page is None:
            failed = [
                name
                for name, page in (("self", heading_page), ("next", next_page))
                if page is None
            ]
            unresolved.append(
                {
                    "id": entry["id"],
                    "check": "orphan_heading",
                    "reason": "text_not_matched",
                    "failed_probes": failed,
                    "heading_text": entry["text"],
                    "next_block_text": entry.get("next_block_text", ""),
                }
            )
            continue

        if heading_page != next_page:
            violations.append(
                {
                    "type": "orphan_heading",
                    "id": entry["id"],
                    "detail": f"見出し「{entry['text'][:30]}」がp.{heading_page}、直後の本文がp.{next_page}に分断されている",
                    "pages": [heading_page, next_page],
                }
            )
        else:
            top = probes.get("self", {}).get("top")
            page = next((p for p in pages if p.number == heading_page), None)
            ratio = (top / page.height) if top is not None and page and page.height else None
            if ratio is None:
                ratio = _word_page_position_ratio(pages, heading_page, entry["text"])
            if ratio is not None and ratio > 0.90:
                violations.append(
                    {
                        "type": "orphan_heading",
                        "id": entry["id"],
                        "detail": f"見出し「{entry['text'][:30]}」がp.{heading_page}の下端付近（高さ比{ratio:.2f}）に位置している",
                        "pages": [heading_page],
                    }
                )

    return violations, unresolved


def detect_figure_caption_split(
    pages: list[PageData], registry: list[dict], resolved: dict[str, dict]
) -> tuple[list[dict], list[dict]]:
    violations: list[dict] = []
    unresolved: list[dict] = []

    for entry in registry:
        if entry["type"] != "figure":
            continue
        caption_page = resolved.get(entry["id"], {}).get("self", {}).get("page")
        if caption_page is None:
            unresolved.append({"id": entry["id"], "check": "figure_caption_image_split", "reason": "text_not_matched"})
            continue

        page = next(p for p in pages if p.number == caption_page)
        if page.n_images == 0:
            violations.append(
                {
                    "type": "figure_caption_image_split",
                    "id": entry["id"],
                    "detail": f"キャプション「{entry['caption'][:30]}」はp.{caption_page}にあるが、同ページに画像が存在しない",
                    "pages": [caption_page],
                }
            )

    return violations, unresolved


def detect_short_block_split(
    pages: list[PageData], registry: list[dict], resolved: dict[str, dict]
) -> tuple[list[dict], list[dict]]:
    violations: list[dict] = []
    unresolved: list[dict] = []

    for entry in registry:
        if entry["type"] not in ("table", "code"):
            continue
        if entry["size_class"] not in ("small-table", "short-code"):
            continue  # 長大な表・コードは分割許可のため対象外

        if entry["type"] == "table":
            first_key, last_key = entry["first_row_text"], entry["last_row_text"]
        else:
            first_key, last_key = entry["first_line"], entry["last_line"]

        if not first_key or not last_key or first_key == last_key:
            continue  # 1行のみ等、比較不能

        probes = resolved.get(entry["id"], {})
        first_page = probes.get("first", {}).get("page")
        last_page = probes.get("last", probes.get("first", {})).get("page")

        if first_page is None or last_page is None:
            unresolved.append({"id": entry["id"], "check": "short_block_split", "reason": "text_not_matched"})
            continue

        if first_page != last_page:
            violations.append(
                {
                    "type": "short_block_split",
                    "id": entry["id"],
                    "detail": f"{entry['type']}（{entry['size_class']}）がp.{first_page}〜p.{last_page}に分断されている",
                    "pages": [first_page, last_page],
                }
            )

    return violations, unresolved


def detect_overflow(pages: list[PageData]) -> list[dict]:
    violations: list[dict] = []
    for page in pages:
        bottom_limit = page.height - PAGE_MARGIN_TOP_BOTTOM_PT + OVERFLOW_TOLERANCE_PT
        right_limit = page.width - PAGE_MARGIN_LEFT_RIGHT_PT + OVERFLOW_TOLERANCE_PT
        left_limit = PAGE_MARGIN_LEFT_RIGHT_PT - OVERFLOW_TOLERANCE_PT

        offenders = [
            w
            for w in page.words
            if w["bottom"] > bottom_limit or w["x1"] > right_limit or w["x0"] < left_limit
        ]
        if offenders:
            sample = offenders[0]
            violations.append(
                {
                    "type": "overflow",
                    "id": f"page-{page.number}",
                    "detail": (
                        f"p.{page.number}で印刷可能領域を超える文字を検出（例: '{sample['text'][:20]}'、"
                        f"x0={sample['x0']:.1f}, x1={sample['x1']:.1f}, bottom={sample['bottom']:.1f} / "
                        f"page {page.width:.1f}x{page.height:.1f}pt）"
                    ),
                    "pages": [page.number],
                }
            )
    return violations


def compute_score(violations: list[dict], unresolved: list[dict] | None = None) -> int:
    unresolved = unresolved or []
    return -(
        sum(VIOLATION_WEIGHTS.get(v["type"], 1) for v in violations)
        + UNRESOLVED_WEIGHT * len(unresolved)
    )


def run_checks(pdf_path: Path, registry_path: Path) -> dict:
    pages = extract_pages(pdf_path)
    registry = load_registry(registry_path)
    resolved = resolve_registry_pages(pages, registry)

    violations: list[dict] = []
    unresolved: list[dict] = []

    for detector in (detect_orphan_headings, detect_figure_caption_split, detect_short_block_split):
        v, u = detector(pages, registry, resolved)
        violations.extend(v)
        unresolved.extend(u)

    violations.extend(detect_overflow(pages))

    status = "FAIL" if violations else ("INDETERMINATE" if unresolved else "PASS")
    return {
        "pdf": str(pdf_path),
        "page_count": len(pages),
        "n_registry_entries": len(registry),
        "violations": violations,
        "unresolved": unresolved,
        "status": status,
        "score": compute_score(violations, unresolved),
    }


def main(argv: list[str] | None = None) -> int:
    """単一PDFを検査するCLI入口。"""
    import sys

    args = sys.argv[1:] if argv is None else argv
    root = Path(__file__).resolve().parents[2]
    pdf = Path(args[0]) if args else root / "outputs" / "SUMMARY_REPORT.pdf"
    registry = (
        Path(args[1])
        if len(args) > 1
        else root / "outputs" / "renders" / "_build" / "summary_report" / "SUMMARY_REPORT.source_registry.json"
    )
    result = run_checks(pdf, registry)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["status"] != "PASS" else 0


def main_all_reports() -> int:
    """本編・発展版の生成済みPDFをまとめて検査する。"""
    root = Path(__file__).resolve().parents[2]
    reports = {}
    for stem in ("SUMMARY_REPORT", "SUMMARY_REPORT_extra"):
        reports[stem] = run_checks(
            root / "outputs" / f"{stem}.pdf",
            root / "outputs" / "renders" / "_build" / stem.lower() / f"{stem}.source_registry.json",
        )
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 1 if any(report["status"] != "PASS" for report in reports.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
