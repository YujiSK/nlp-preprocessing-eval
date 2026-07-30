"""SUMMARY_REPORT / SUMMARY_REPORT_extraをbuild -> render -> checkするオーケストレータ。

停止条件:
  - 検出違反件数が0件になった場合
  - スコア（違反の重み付き件数の符号反転）が前回より改善しなかった場合は、
    直前の最良状態（configs/layout_overrides.yml と成果物）へロールバックして停止する

自動修復は `configs/layout_overrides.yml` の `page_break_before` / `keep_together` を
    違反内容から機械的に追記する形で行う（入力Markdown本文は変更しない）。
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from . import layout_checker as checker
from . import pdf_renderer as pdf_render
from . import report_builder as report_build

import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("pdfminer").setLevel(logging.ERROR)

TASK9_ROOT = Path(__file__).resolve().parents[2]
OVERRIDES_PATH = TASK9_ROOT / "configs" / "layout_overrides.yml"
OUTPUTS_DIR = TASK9_ROOT / "outputs"
RENDERS_ROOT = OUTPUTS_DIR / "renders"
MD_PATH = OUTPUTS_DIR / "SUMMARY_REPORT.md"
PDF_PATH = OUTPUTS_DIR / "SUMMARY_REPORT.pdf"
BUILD_DIR = RENDERS_ROOT / "_build" / "summary"
RENDERS_DIR = RENDERS_ROOT / "summary"

# テストが`_build_render_check()`を引数なしでモックできる後方互換性を維持しつつ、
# run_pipelineごとに対象レポートを切り替える。
_ACTIVE_MD_PATH = MD_PATH
_ACTIVE_PDF_PATH = PDF_PATH
_ACTIVE_BUILD_DIR = BUILD_DIR
_ACTIVE_RENDERS_DIR = RENDERS_DIR

MAX_ITERATIONS = 3
REPORT_STEMS = {
    "main": ("SUMMARY_REPORT",),
    "extra": ("SUMMARY_REPORT_extra",),
    "both": ("SUMMARY_REPORT", "SUMMARY_REPORT_extra"),
}


def _load_overrides() -> dict:
    if not OVERRIDES_PATH.exists():
        return {"page_break_before": [], "keep_together": []}
    data = yaml.safe_load(OVERRIDES_PATH.read_text(encoding="utf-8")) or {}
    return {
        "page_break_before": list(data.get("page_break_before") or []),
        "keep_together": list(data.get("keep_together") or []),
    }


def _save_overrides(overrides: dict) -> None:
    header = (
        "# SUMMARY_REPORT*.md本文を書き換えずにPDFレイアウトを調整するためのオーバーライド定義。\n"
        "# src/reporting/report_builder.py がビルド時に読み込み、対応する data-source-id 要素へ\n"
        "# CSSクラス（force-page-break / keep-together）を付与する。\n"
        "#\n"
        "# id は outputs/renders/_build/source_registry.json に列挙されているものを指定する。\n"
        "# 本ファイルは src/reporting/layout_pipeline.py の自動修復ループによっても更新される\n"
        "# （手動編集後に自動修復を実行すると上書きされる場合がある点に注意）。\n\n"
    )
    body = yaml.safe_dump(
        {
            "page_break_before": sorted(set(overrides.get("page_break_before", []))),
            "keep_together": sorted(set(overrides.get("keep_together", []))),
        },
        allow_unicode=True,
        sort_keys=False,
    )
    OVERRIDES_PATH.write_text(header + body, encoding="utf-8")


def _suggest_fixes(violations: list[dict]) -> dict:
    """検出違反から、次イテレーションで追加すべきオーバーライドを機械的に提案する。"""
    page_break_before: set[str] = set()
    keep_together: set[str] = set()

    for v in violations:
        if v["type"] == "orphan_heading":
            # 見出しを新規ページの先頭へ強制移動し、直後の本文との分断を防ぐ
            page_break_before.add(v["id"])
        elif v["type"] == "figure_caption_image_split":
            keep_together.add(v["id"])
        elif v["type"] == "short_block_split":
            keep_together.add(v["id"])
        # "overflow" はID単位のオーバーライドでは直接解消できないため、自動修復の対象外とし
        # layout_report.md 側で手動対応が必要な項目として報告する。

    return {"page_break_before": sorted(page_break_before), "keep_together": sorted(keep_together)}


def _merge_overrides(base: dict, additions: dict) -> dict:
    return {
        "page_break_before": sorted(set(base.get("page_break_before", [])) | set(additions.get("page_break_before", []))),
        "keep_together": sorted(set(base.get("keep_together", [])) | set(additions.get("keep_together", []))),
    }


def _build_render_check() -> dict:
    html_path, registry_path = report_build.build(
        md_path=_ACTIVE_MD_PATH,
        build_dir=_ACTIVE_BUILD_DIR,
    )
    pdf_render.render_html_to_pdf(html_path, _ACTIVE_PDF_PATH)
    return checker.run_checks(_ACTIVE_PDF_PATH, registry_path)


def run_pipeline(
    max_iterations: int = MAX_ITERATIONS,
    md_path: Path = MD_PATH,
    pdf_path: Path = PDF_PATH,
    build_dir: Path = BUILD_DIR,
    renders_dir: Path = RENDERS_DIR,
) -> dict:
    global _ACTIVE_MD_PATH, _ACTIVE_PDF_PATH, _ACTIVE_BUILD_DIR, _ACTIVE_RENDERS_DIR
    _ACTIVE_MD_PATH = Path(md_path)
    _ACTIVE_PDF_PATH = Path(pdf_path)
    _ACTIVE_BUILD_DIR = Path(build_dir)
    _ACTIVE_RENDERS_DIR = Path(renders_dir)

    starting_overrides = _load_overrides()
    current_overrides = copy.deepcopy(starting_overrides)

    history: list[dict] = []
    best = None  # {"overrides":..., "score":..., "result":..., "iteration":...}

    for iteration in range(1, max_iterations + 1):
        _save_overrides(current_overrides)
        result = _build_render_check()
        score = result["score"]
        n_violations = len(result["violations"])

        record = {
            "iteration": iteration,
            "overrides": copy.deepcopy(current_overrides),
            "score": score,
            "n_violations": n_violations,
            "violation_types": sorted({v["type"] for v in result["violations"]}),
        }

        if best is None or score > best["score"]:
            history.append({**record, "action": "accepted"})
            best = {"overrides": copy.deepcopy(current_overrides), "score": score, "result": result, "iteration": iteration}
        else:
            # 改善なし（悪化 or 同一）: 直前の最良状態へロールバックして停止する
            history.append({**record, "action": "rejected_rollback"})
            _save_overrides(best["overrides"])
            _build_render_check()  # 成果物(HTML/PDF)を最良状態に戻す
            history.append({"iteration": iteration, "action": "rollback_to_iteration", "target_iteration": best["iteration"]})
            break

        if n_violations == 0:
            break

        suggestions = _suggest_fixes(result["violations"])
        next_overrides = _merge_overrides(current_overrides, suggestions)
        if next_overrides == current_overrides:
            # 提案できる修正がこれ以上ない（overflow等、ID単位で直せない違反のみ残っている）
            history.append({"iteration": iteration, "action": "no_further_autofix_available"})
            break
        current_overrides = next_overrides

    # 最終的に最良状態のオーバーライドが反映された成果物を確定させる
    _save_overrides(best["overrides"])
    final_result = _build_render_check()
    pages = pdf_render.render_pdf_to_page_images(_ACTIVE_PDF_PATH, _ACTIVE_RENDERS_DIR)

    return {
        "history": history,
        "best_iteration": best["iteration"],
        "final_overrides": best["overrides"],
        "final_result": final_result,
        "n_rendered_pages": len(pages),
    }


def build_pdf_only(
    md_path: Path,
    pdf_path: Path,
    build_dir: Path,
) -> dict:
    """既存Markdownと成果物をHTML化し、検査・自動修復・ページ画像化なしでPDFを描画する。"""
    html_path, registry_path = report_build.build(
        md_path=md_path,
        build_dir=build_dir,
    )
    pdf_render.render_html_to_pdf(html_path, pdf_path)
    return {
        "pdf": str(pdf_path),
        "html": str(html_path),
        "source_registry": str(registry_path),
        "mode": "pdf-only",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """レポートビルドCLIの引数を解析する。"""
    parser = argparse.ArgumentParser(description="Build and validate PDF report(s).")
    parser.add_argument(
        "--target",
        "-t",
        choices=tuple(REPORT_STEMS),
        default="both",
        help=(
            "Report to build: 'main' (SUMMARY_REPORT), "
            "'extra' (SUMMARY_REPORT_extra), or 'both' (default)."
        ),
    )
    parser.add_argument(
        "--pdf-only",
        "-p",
        action="store_true",
        help=(
            "Build HTML and PDF only from existing Markdown/results; "
            "skip layout checks, auto-fixes, and page-image rendering."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """選択された本編・発展版を構築し、検査結果を保存する。"""
    args = parse_args(argv)
    reports = {}
    for stem in REPORT_STEMS[args.target]:
        print(f"Building {stem}.pdf ...")
        if args.pdf_only:
            reports[stem] = build_pdf_only(
                md_path=OUTPUTS_DIR / f"{stem}.md",
                pdf_path=OUTPUTS_DIR / f"{stem}.pdf",
                build_dir=RENDERS_ROOT / "_build" / stem.lower(),
            )
            continue

        reports[stem] = run_pipeline(
            md_path=OUTPUTS_DIR / f"{stem}.md",
            pdf_path=OUTPUTS_DIR / f"{stem}.pdf",
            build_dir=RENDERS_ROOT / "_build" / stem.lower(),
            renders_dir=RENDERS_ROOT / stem.lower(),
        )
        report_path = OUTPUTS_DIR / "reports" / f"layout_{stem.lower()}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(reports[stem], ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    print(json.dumps(reports, ensure_ascii=False, indent=2, default=str))
    if args.pdf_only:
        return 0
    return 1 if any(report["final_result"]["violations"] for report in reports.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
