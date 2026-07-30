"""SUMMARY_REPORT*.md -> 構造化中間HTML のビルドスクリプト。

pdf_pipeline_audit.md で指摘した再現性の欠如（使い捨てHTML・ID未付与）を解消するため、
- 中間HTMLをレポート名ごとに`outputs/renders/_build/{stem}.render.html`として保存する
- 見出し／図／表／コードブロックに安定した `data-source-id` を付与する
- 呼び出し元から直接受け取ったレイアウト指示を反映する
- `outputs/renders/_build/source_registry.json` に ID→要素情報のレジストリを出力する
  （`src/reporting/layout_checker.py` がこのレジストリとPDFを突き合わせて検査する）

入力Markdownの本文そのものは書き換えない。
"""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path

import markdown
from bs4 import BeautifulSoup, NavigableString, Tag

TASK9_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MD_PATH = TASK9_ROOT / "outputs" / "SUMMARY_REPORT.md"
DEFAULT_CSS_PATH = TASK9_ROOT / "assets" / "styles" / "report.css"
DEFAULT_BUILD_DIR = TASK9_ROOT / "outputs" / "renders" / "_build"

LONG_TABLE_ROW_THRESHOLD = 15  # これを超える行数の表は "long-table"（分割許可）とする
LONG_CODE_LINE_THRESHOLD = 30  # これを超える行数のコードは "long-code"（分割許可）とする

_SECTION_LABEL_SLUGS = {
    "事実": "fact",
    "メカニズム": "mechanism",
    "改善策・提言": "recommendation",
    "方法": "method",
    "結果": "results",
    "解釈": "interpretation",
    "解釈上の注意": "interpretation-notes",
    "実務的示唆": "practical-implications",
    "再現性": "reproducibility",
    "設計": "design",
    "目的と設計": "purpose-design",
    "目的とデータ": "purpose-data",
    "評価方法": "evaluation-method",
    "結果の要約": "results-summary",
    "語彙数・決定論的前処理コスト": "vocabulary-processing-cost",
    "要因別macro-F1差": "factor-macro-f1-difference",
    "macro-F1": "macro-f1",
}


def markdown_to_html(md_text: str) -> str:
    return markdown.markdown(
        md_text,
        extensions=["extra", "tables", "fenced_code", "sane_lists"],
    )


class _IdAllocator:
    """見出し階層と本文から、挿入順に依存しない安定IDを割り当てる。"""

    def __init__(self) -> None:
        self.chapter = "root"
        self._heading_stack: dict[int, str] = {}
        self._used_heading_ids: set[str] = set()
        self._content_counters: dict[str, int] = {}

    @staticmethod
    def _semantic_slug(text: str) -> str:
        inner = re.sub(r"[【】]", "", text).strip()
        if inner in _SECTION_LABEL_SLUGS:
            return _SECTION_LABEL_SLUGS[inner]
        experiment = re.match(r"実験([A-D])", inner)
        if experiment:
            return f"experiment-{experiment.group(1).lower()}"
        ascii_tokens = re.findall(r"[A-Za-z0-9]+", inner.lower())
        return "-".join(ascii_tokens[:6]) or "section"

    def _fallback_heading_id(self, level: int, text: str) -> str:
        parent = self._heading_stack.get(level - 1, f"heading-{self.chapter}")
        slug = self._semantic_slug(text)
        normalized = " ".join(text.split())
        digest = hashlib.sha1(f"{parent}|h{level}|{normalized}".encode("utf-8")).hexdigest()[:10]
        return f"{parent}-{slug}-{digest}"

    def _register_heading(self, level: int, heading_id: str) -> str:
        if heading_id in self._used_heading_ids:
            raise ValueError(
                f"Duplicate semantic heading id {heading_id!r}; "
                "add an explicit numbered heading or make sibling heading text unique."
            )
        self._used_heading_ids.add(heading_id)
        self._heading_stack[level] = heading_id
        for child_level in tuple(self._heading_stack):
            if child_level > level:
                del self._heading_stack[child_level]
        return heading_id

    def heading_id(self, level: str, text: str) -> str:
        text = text.strip()
        level_number = int(level[1:])

        if level == "h1":
            return self._register_heading(level_number, "heading-title")

        if level == "h2":
            m = re.match(r"第(\d+)章", text)
            if m:
                self.chapter = m.group(1)
                return self._register_heading(level_number, f"heading-{self.chapter}")
            m = re.match(r"付録([A-Za-z])", text)
            if m:
                self.chapter = f"appendix-{m.group(1).lower()}"
                return self._register_heading(level_number, f"heading-{self.chapter}")
            return self._register_heading(level_number, self._fallback_heading_id(level_number, text))

        # h3 / h4
        m = re.match(r"^(\d+)\.(\d+)", text)
        if m:
            return self._register_heading(level_number, f"heading-{m.group(1)}-{m.group(2)}")
        m = re.match(r"^([A-Za-z])\.(\d+)", text)
        if m:
            return self._register_heading(
                level_number, f"heading-appendix-{m.group(1).lower()}-{m.group(2)}"
            )
        m = re.match(r"原則(\d+)", text)
        if m:
            return self._register_heading(
                level_number, f"heading-{self.chapter}-principle-{m.group(1)}"
            )

        return self._register_heading(level_number, self._fallback_heading_id(level_number, text))

    def next_index(self, kind: str) -> int:
        key = f"{self.chapter}-{kind}"
        self._content_counters[key] = self._content_counters.get(key, 0) + 1
        return self._content_counters[key]


def _visible_text(tag) -> str:
    return " ".join(tag.get_text(" ", strip=True).split())


_FIGURE_CAPTION_RE = re.compile(r"図\s*([0-9]+\.[0-9]+)")


def _try_wrap_figure(soup: BeautifulSoup, p: Tag, allocator: _IdAllocator) -> Tag | None:
    """`<p><img/><em>図 X.Y: ...</em></p>` を `<figure><img/><figcaption>...</figcaption></figure>` へ変換する。

    対象パターンでなければ None を返し、呼び出し側は p をそのまま扱う。
    """
    children = [c for c in p.children if not (isinstance(c, NavigableString) and not c.strip())]
    if len(children) != 2:
        return None
    img, em = children
    if not (isinstance(img, Tag) and img.name == "img"):
        return None
    if not (isinstance(em, Tag) and em.name == "em"):
        return None

    caption_text = _visible_text(em)
    m = _FIGURE_CAPTION_RE.search(caption_text)
    fig_id = f"figure-{m.group(1).replace('.', '-')}" if m else f"figure-{allocator.chapter}-{allocator.next_index('figure')}"

    figure = soup.new_tag("figure")
    figure["id"] = fig_id
    figure["data-source-id"] = fig_id
    img.extract()
    figure.append(img)
    figcaption = soup.new_tag("figcaption")
    figcaption.string = caption_text
    figure.append(figcaption)

    p.replace_with(figure)

    figure._registry_entry = {  # type: ignore[attr-defined]
        "id": fig_id,
        "type": "figure",
        "caption": caption_text,
        "img_src": img.get("src", ""),
    }
    return figure


def _process_document_order(soup: BeautifulSoup, allocator: _IdAllocator, registry: list[dict]) -> None:
    """見出し・図・表・コードブロックを文書出現順に1回だけ走査し、章コンテキストを保ったままIDを付与する。"""
    elements = soup.find_all(["h1", "h2", "h3", "h4", "p", "table", "pre"])
    first_h2_seen = False

    for el in elements:
        if el.name in ("h1", "h2", "h3", "h4"):
            text = _visible_text(el)
            hid = allocator.heading_id(el.name, text)
            el["id"] = hid
            el["data-source-id"] = hid

            if el.name == "h2":
                if not first_h2_seen:
                    classes = el.get("class", [])
                    classes.append("no-page-break-before")
                    el["class"] = classes
                    first_h2_seen = True

            next_el = el.find_next_sibling()
            next_text = _visible_text(next_el)[:60] if next_el else ""

            registry.append(
                {
                    "id": hid,
                    "type": "heading",
                    "level": el.name,
                    "text": text,
                    "next_block_text": next_text,
                }
            )

        elif el.name == "p":
            figure = _try_wrap_figure(soup, el, allocator)
            if figure is not None:
                registry.append(figure._registry_entry)  # type: ignore[attr-defined]
                del figure._registry_entry  # type: ignore[attr-defined]

        elif el.name == "table":
            rows = el.find_all("tr")
            n_rows = len(rows)
            tid = f"table-{allocator.chapter}-{allocator.next_index('table')}"
            el["id"] = tid
            el["data-source-id"] = tid

            size_class = "long-table" if n_rows > LONG_TABLE_ROW_THRESHOLD else "small-table"
            classes = el.get("class", [])
            classes.append(size_class)
            el["class"] = classes

            first_row = _visible_text(rows[0]) if rows else ""
            last_row = _visible_text(rows[-1]) if rows else ""

            registry.append(
                {
                    "id": tid,
                    "type": "table",
                    "n_rows": n_rows,
                    "size_class": size_class,
                    "first_row_text": first_row[:60],
                    "last_row_text": last_row[:60],
                }
            )

        elif el.name == "pre":
            code = el.find("code")
            text = code.get_text() if code else el.get_text()
            lines = text.rstrip("\n").split("\n")
            n_lines = len(lines)
            cid = f"code-{allocator.chapter}-{allocator.next_index('code')}"
            el["id"] = cid
            el["data-source-id"] = cid

            size_class = "long-code" if n_lines > LONG_CODE_LINE_THRESHOLD else "short-code"
            classes = el.get("class", [])
            classes.append(size_class)
            el["class"] = classes

            registry.append(
                {
                    "id": cid,
                    "type": "code",
                    "n_lines": n_lines,
                    "size_class": size_class,
                    "first_line": lines[0][:60] if lines else "",
                    "last_line": lines[-1][:60] if lines else "",
                }
            )


def _apply_overrides(soup: BeautifulSoup, overrides: dict) -> None:
    missing_ids: list[str] = []
    for oid in overrides.get("page_break_before", []):
        el = soup.find(attrs={"data-source-id": oid})
        if el is None:
            missing_ids.append(oid)
            continue
        classes = el.get("class", [])
        classes.append("force-page-break")
        el["class"] = classes

    for oid in overrides.get("keep_together", []):
        el = soup.find(attrs={"data-source-id": oid})
        if el is None:
            missing_ids.append(oid)
            continue
        classes = el.get("class", [])
        classes.append("keep-together")
        el["class"] = classes

    if missing_ids:
        raise ValueError(
            "Unknown layout override data-source-id(s): "
            + ", ".join(sorted(set(missing_ids)))
        )


def build(
    md_path: Path = DEFAULT_MD_PATH,
    overrides: dict | None = None,
    css_path: Path = DEFAULT_CSS_PATH,
    build_dir: Path = DEFAULT_BUILD_DIR,
) -> tuple[Path, Path]:
    """Markdownをビルドし、(html_path, registry_path)を返す。"""
    build_dir.mkdir(parents=True, exist_ok=True)

    md_text = md_path.read_text(encoding="utf-8")
    body_html = markdown_to_html(md_text)

    soup = BeautifulSoup(f"<html><body>{body_html}</body></html>", "html.parser")

    registry: list[dict] = []
    allocator = _IdAllocator()

    _process_document_order(soup, allocator, registry)

    _apply_overrides(
        soup,
        overrides or {"page_break_before": [], "keep_together": []},
    )

    css_text = css_path.read_text(encoding="utf-8")
    full_html = (
        "<!doctype html>\n"
        '<html lang="ja">\n<head>\n<meta charset="utf-8">\n<base href="/">\n'
        "<title>機械学習前処理パイプラインの定量的評価と実務ガイドライン</title>\n"
        f"<style>\n{css_text}\n</style>\n</head>\n<body>\n"
        f"{soup.body.decode_contents()}\n</body>\n</html>\n"
    )

    registry_out = build_dir / f"{md_path.stem}.source_registry.json"
    registry_out.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")

    html_out = build_dir / f"{md_path.stem}.render.html"
    html_out.write_text(full_html, encoding="utf-8")

    return html_out, registry_out


if __name__ == "__main__":
    html_path, registry_path = build()
    print("wrote", html_path)
    print("wrote", registry_path)
