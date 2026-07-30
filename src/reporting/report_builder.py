"""SUMMARY_REPORT*.md -> 構造化中間HTML のビルドスクリプト。

pdf_pipeline_audit.md で指摘した再現性の欠如（使い捨てHTML・ID未付与）を解消するため、
- 中間HTMLをレポート名ごとに`outputs/renders/_build/{stem}.render.html`として保存する
- 見出し／図／表／コードブロックに安定した `data-source-id` を付与する
- `configs/layout_overrides.yml` の指示（page_break_before / keep_together）を反映する
- `outputs/renders/_build/source_registry.json` に ID→要素情報のレジストリを出力する
  （`src/reporting/layout_checker.py` がこのレジストリとPDFを突き合わせて検査する）

入力Markdownの本文そのものは書き換えない。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import markdown
import yaml
from bs4 import BeautifulSoup, NavigableString, Tag

TASK9_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MD_PATH = TASK9_ROOT / "outputs" / "SUMMARY_REPORT.md"
DEFAULT_OVERRIDES_PATH = TASK9_ROOT / "configs" / "layout_overrides.yml"
DEFAULT_CSS_PATH = TASK9_ROOT / "assets" / "styles" / "report.css"
DEFAULT_RENDERS_DIR = TASK9_ROOT / "outputs" / "renders"
DEFAULT_BUILD_DIR = DEFAULT_RENDERS_DIR / "_build"

LONG_TABLE_ROW_THRESHOLD = 15  # これを超える行数の表は "long-table"（分割許可）とする
LONG_CODE_LINE_THRESHOLD = 30  # これを超える行数のコードは "long-code"（分割許可）とする

_SECTION_LABEL_SLUGS = {
    "事実": "fact",
    "メカニズム": "mechanism",
    "改善策・提言": "recommendation",
}


def load_overrides(path: Path = DEFAULT_OVERRIDES_PATH) -> dict:
    if not path.exists():
        return {"page_break_before": [], "keep_together": []}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {
        "page_break_before": list(data.get("page_break_before") or []),
        "keep_together": list(data.get("keep_together") or []),
    }


def markdown_to_html(md_text: str) -> str:
    return markdown.markdown(
        md_text,
        extensions=["extra", "tables", "fenced_code", "sane_lists"],
    )


class _IdAllocator:
    """文書順に見出しを走査し、章コンテキストに基づき安定したIDを割り当てる。"""

    def __init__(self) -> None:
        self.chapter = "root"
        self._fallback_counters: dict[str, int] = {}

    def heading_id(self, level: str, text: str) -> str:
        text = text.strip()

        if level == "h1":
            return "heading-title"

        if level == "h2":
            m = re.match(r"第(\d+)章", text)
            if m:
                self.chapter = m.group(1)
                return f"heading-{self.chapter}"
            m = re.match(r"付録([A-Za-z])", text)
            if m:
                self.chapter = f"appendix-{m.group(1).lower()}"
                return f"heading-{self.chapter}"
            return "heading-subtitle"

        # h3 / h4
        m = re.match(r"^(\d+)\.(\d+)", text)
        if m:
            return f"heading-{m.group(1)}-{m.group(2)}"
        m = re.match(r"^([A-Za-z])\.(\d+)", text)
        if m:
            return f"heading-appendix-{m.group(1).lower()}-{m.group(2)}"
        m = re.match(r"原則(\d+)", text)
        if m:
            return f"heading-{self.chapter}-principle-{m.group(1)}"

        inner = re.sub(r"[【】]", "", text)
        slug = _SECTION_LABEL_SLUGS.get(inner)
        if slug is None:
            key = f"{self.chapter}-fallback"
            self._fallback_counters[key] = self._fallback_counters.get(key, 0) + 1
            slug = f"section-{self._fallback_counters[key]}"
        return f"heading-{self.chapter}-{slug}"

    def next_index(self, kind: str) -> int:
        key = f"{self.chapter}-{kind}"
        self._fallback_counters[key] = self._fallback_counters.get(key, 0) + 1
        return self._fallback_counters[key]


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
    for oid in overrides.get("page_break_before", []):
        el = soup.find(attrs={"data-source-id": oid})
        if el is None:
            continue
        classes = el.get("class", [])
        classes.append("force-page-break")
        el["class"] = classes

    for oid in overrides.get("keep_together", []):
        el = soup.find(attrs={"data-source-id": oid})
        if el is None:
            continue
        classes = el.get("class", [])
        classes.append("keep-together")
        el["class"] = classes


def build(
    md_path: Path = DEFAULT_MD_PATH,
    overrides_path: Path = DEFAULT_OVERRIDES_PATH,
    css_path: Path = DEFAULT_CSS_PATH,
    build_dir: Path = DEFAULT_BUILD_DIR,
    render_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Markdownをビルドし、(html_path, registry_path)を返す。"""
    build_dir.mkdir(parents=True, exist_ok=True)

    md_text = md_path.read_text(encoding="utf-8")
    body_html = markdown_to_html(md_text)

    soup = BeautifulSoup(f"<html><body>{body_html}</body></html>", "html.parser")

    registry: list[dict] = []
    allocator = _IdAllocator()

    _process_document_order(soup, allocator, registry)

    overrides = load_overrides(overrides_path)
    _apply_overrides(soup, overrides)

    css_text = css_path.read_text(encoding="utf-8")
    full_html = (
        "<!doctype html>\n"
        '<html lang="ja">\n<head>\n<meta charset="utf-8">\n<base href="/">\n'
        "<title>機械学習前処理パイプラインの定量的評価と実務ガイドライン</title>\n"
        f"<style>\n{css_text}\n</style>\n</head>\n<body>\n"
        f"{soup.body.decode_contents()}\n</body>\n</html>\n"
    )

    # 最終レポートはrenders/へ集約し、任意のテスト文書は入力元の近傍へ隔離する。
    if render_dir is None:
        render_dir = DEFAULT_RENDERS_DIR if md_path.parent == DEFAULT_MD_PATH.parent else md_path.parent
    html_out = Path(render_dir) / f"_{md_path.stem.lower()}_render.html"
    html_out.parent.mkdir(parents=True, exist_ok=True)
    html_out.write_text(full_html, encoding="utf-8")

    registry_out = build_dir / f"{md_path.stem}.source_registry.json"
    registry_out.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")

    # 監査・再現性のため、生成HTMLの控えを outputs/renders/_build/ にも保存する
    (build_dir / f"{md_path.stem}.render.html").write_text(full_html, encoding="utf-8")

    return html_out, registry_out


if __name__ == "__main__":
    html_path, registry_path = build()
    print("wrote", html_path)
    print("wrote", registry_path)
