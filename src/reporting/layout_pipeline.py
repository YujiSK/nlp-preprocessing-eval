"""SUMMARY_REPORT / SUMMARY_REPORT_extraの安全なbuild -> render -> check制御。

通常ビルドは現在のmanual/generated設定を読み取り専用で適用し、YAMLを書き換えない。
`--auto-repair`を明示した場合だけ、違反からgenerated設定をメモリ上で探索し、
最終採用した対象レポートのgenerated部分だけを原子的に保存する。
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import logging
import os
import shutil
import tempfile
import warnings
from pathlib import Path

import yaml

from . import layout_checker as checker
from . import pdf_renderer as pdf_render
from . import report_builder as report_build

warnings.filterwarnings("ignore")
logging.getLogger("pdfminer").setLevel(logging.ERROR)

TASK9_ROOT = Path(__file__).resolve().parents[2]
OVERRIDES_PATH = TASK9_ROOT / "configs" / "layout_overrides.yml"
OUTPUTS_DIR = TASK9_ROOT / "outputs"
RENDERS_ROOT = OUTPUTS_DIR / "renders"
MAX_ITERATIONS = 3
REPORT_STEMS = {
    "main": ("SUMMARY_REPORT",),
    "extra": ("SUMMARY_REPORT_extra",),
    "both": ("SUMMARY_REPORT", "SUMMARY_REPORT_extra"),
}
TARGET_BY_STEM = {
    "SUMMARY_REPORT": "main",
    "SUMMARY_REPORT_extra": "extra",
}


def _empty_overrides() -> dict:
    return {"page_break_before": [], "keep_together": []}


def _normalize_overrides(value: dict | None) -> dict:
    value = value or {}
    return {
        "page_break_before": sorted(set(value.get("page_break_before") or [])),
        "keep_together": sorted(set(value.get("keep_together") or [])),
    }


def _default_override_config() -> dict:
    return {
        "manual": {
            "main": _empty_overrides(),
            "extra": _empty_overrides(),
        },
        "generated": {
            "main": _empty_overrides(),
            "extra": _empty_overrides(),
        },
    }


def _load_override_config(path: Path = OVERRIDES_PATH) -> dict:
    """新スキーマを読み込む。旧ルート直下リストはmanualとして非破壊移行する。"""
    if not path.exists():
        return _default_override_config()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = _default_override_config()

    if "manual" in raw or "generated" in raw:
        manual = raw.get("manual") or {}
        if "main" in manual or "extra" in manual:
            for target in ("main", "extra"):
                config["manual"][target] = _normalize_overrides(manual.get(target))
        else:
            # 初期リファクタリング版のmanual共通リストも両targetへ安全に移行する。
            shared_manual = _normalize_overrides(manual)
            for target in ("main", "extra"):
                config["manual"][target] = copy.deepcopy(shared_manual)
        generated = raw.get("generated") or {}
        for target in ("main", "extra"):
            config["generated"][target] = _normalize_overrides(generated.get(target))
    else:
        # 旧ルート直下リストは両レポートのmanualとして非破壊移行する。
        shared_manual = _normalize_overrides(raw)
        for target in ("main", "extra"):
            config["manual"][target] = copy.deepcopy(shared_manual)

    return config


def _merge_overrides(base: dict, additions: dict) -> dict:
    return {
        "page_break_before": sorted(
            set(base.get("page_break_before", []))
            | set(additions.get("page_break_before", []))
        ),
        "keep_together": sorted(
            set(base.get("keep_together", []))
            | set(additions.get("keep_together", []))
        ),
    }


def _effective_overrides(config: dict, target: str, generated: dict | None = None) -> dict:
    generated_value = (
        config["generated"][target] if generated is None else _normalize_overrides(generated)
    )
    return _merge_overrides(config["manual"][target], generated_value)


def _config_yaml(config: dict) -> str:
    header = (
        "# レポート本文を書き換えずにPDFレイアウトを調整する設定。\n"
        "# manual.main / manual.extraは人が管理し、自動修復処理は絶対に変更しない。\n"
        "# generated.main / generated.extraは--auto-repair指定時だけ対象別に更新される。\n\n"
    )
    normalized = {
        "manual": {
            target: _normalize_overrides((config.get("manual") or {}).get(target))
            for target in ("main", "extra")
        },
        "generated": {
            target: _normalize_overrides((config.get("generated") or {}).get(target))
            for target in ("main", "extra")
        },
    }
    return header + yaml.safe_dump(
        normalized,
        allow_unicode=True,
        sort_keys=False,
    )


def _save_generated_overrides(
    target: str,
    generated_overrides: dict,
    path: Path = OVERRIDES_PATH,
) -> None:
    """manualと他targetを再読込で保護し、対象generatedだけを原子的に保存する。"""
    if target not in ("main", "extra"):
        raise ValueError(f"Unknown override target: {target}")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_name = f"nlp-preprocessing-eval-{path.resolve().as_posix().replace('/', '_')}.lock"
    lock_path = Path(tempfile.gettempdir()) / lock_name

    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        latest = _load_override_config(path)
        latest["generated"][target] = _normalize_overrides(generated_overrides)
        temp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_file.write(_config_yaml(latest))
                temp_file.flush()
                os.fsync(temp_file.fileno())
                temp_name = temp_file.name
            os.replace(temp_name, path)
            temp_name = None
        finally:
            if temp_name is not None:
                Path(temp_name).unlink(missing_ok=True)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _suggest_fixes(violations: list[dict]) -> dict:
    page_break_before: set[str] = set()
    keep_together: set[str] = set()
    for violation in violations:
        if violation["type"] == "orphan_heading":
            page_break_before.add(violation["id"])
        elif violation["type"] in (
            "figure_caption_image_split",
            "short_block_split",
        ):
            keep_together.add(violation["id"])
    return {
        "page_break_before": sorted(page_break_before),
        "keep_together": sorted(keep_together),
    }


def _build_render_check(
    *,
    md_path: Path,
    pdf_path: Path,
    build_dir: Path,
    overrides: dict,
) -> dict:
    html_path, registry_path = report_build.build(
        md_path=md_path,
        build_dir=build_dir,
        overrides=overrides,
    )
    pdf_render.render_html_to_pdf(html_path, pdf_path)
    return checker.run_checks(pdf_path, registry_path)


def _cleanup_transient_artifacts(stem: str) -> None:
    """監査に不要な旧HTML複製とページPNGだけを限定的に除去する。"""
    for legacy_html in (
        "_final_report_render.html",
        "_summary_report_render.html",
        "_summary_report_extra_render.html",
    ):
        (RENDERS_ROOT / legacy_html).unlink(missing_ok=True)
    for legacy_page in RENDERS_ROOT.glob("page-*.png"):
        legacy_page.unlink()
    for legacy_build_file in (
        RENDERS_ROOT / "_build" / "FINAL_REPORT.render.html",
        RENDERS_ROOT / "_build" / "source_registry.json",
    ):
        legacy_build_file.unlink(missing_ok=True)

    for report_stem in TARGET_BY_STEM:
        page_dir = RENDERS_ROOT / report_stem.lower()
        if page_dir.exists():
            shutil.rmtree(page_dir)


def run_without_repair(
    *,
    target: str,
    md_path: Path,
    pdf_path: Path,
    build_dir: Path,
) -> dict:
    """設定を一切保存せず、現在の有効設定で1回だけビルド・検査する。"""
    config = _load_override_config()
    effective = _effective_overrides(config, target)
    result = _build_render_check(
        md_path=md_path,
        pdf_path=pdf_path,
        build_dir=build_dir,
        overrides=effective,
    )
    _cleanup_transient_artifacts(md_path.stem)
    return {
        "mode": "no-repair",
        "effective_overrides": effective,
        "final_result": result,
    }


def run_pipeline(
    *,
    target: str,
    md_path: Path,
    pdf_path: Path,
    build_dir: Path,
    max_iterations: int = MAX_ITERATIONS,
) -> dict:
    """明示的な自動修復モード。探索はメモリ上で行い、最良generatedだけを保存する。"""
    config = _load_override_config()
    current_generated = copy.deepcopy(config["generated"][target])
    best: dict | None = None
    history: list[dict] = []

    for iteration in range(1, max_iterations + 1):
        effective = _effective_overrides(config, target, current_generated)
        result = _build_render_check(
            md_path=md_path,
            pdf_path=pdf_path,
            build_dir=build_dir,
            overrides=effective,
        )
        record = {
            "iteration": iteration,
            "generated_overrides": copy.deepcopy(current_generated),
            "effective_overrides": effective,
            "status": result["status"],
            "score": result["score"],
            "n_violations": len(result["violations"]),
            "n_unresolved": len(result["unresolved"]),
            "violation_types": sorted({v["type"] for v in result["violations"]}),
        }

        if best is None or result["score"] > best["result"]["score"]:
            history.append({**record, "action": "accepted"})
            best = {
                "generated": copy.deepcopy(current_generated),
                "effective": effective,
                "result": result,
                "iteration": iteration,
            }
        else:
            history.append({**record, "action": "rejected_rollback"})
            history.append(
                {
                    "iteration": iteration,
                    "action": "rollback_to_iteration",
                    "target_iteration": best["iteration"],
                }
            )
            break

        if result["status"] == "PASS":
            break

        suggestions = _suggest_fixes(result["violations"])
        next_generated = _merge_overrides(current_generated, suggestions)
        if next_generated == current_generated:
            action = (
                "indeterminate_no_safe_autofix"
                if result["unresolved"]
                else "no_further_autofix_available"
            )
            history.append({"iteration": iteration, "action": action})
            break
        current_generated = next_generated

    if best is None:
        raise RuntimeError("Auto-repair loop produced no candidate")

    final_result = _build_render_check(
        md_path=md_path,
        pdf_path=pdf_path,
        build_dir=build_dir,
        overrides=best["effective"],
    )
    _save_generated_overrides(target, best["generated"])
    _cleanup_transient_artifacts(md_path.stem)

    return {
        "mode": "auto-repair",
        "history": history,
        "best_iteration": best["iteration"],
        "manual_overrides": config["manual"][target],
        "generated_overrides": best["generated"],
        "effective_overrides": best["effective"],
        "final_result": final_result,
    }


def build_pdf_only(
    *,
    target: str,
    md_path: Path,
    pdf_path: Path,
    build_dir: Path,
) -> dict:
    """既存結果からHTML/PDFだけを生成し、検査・設定保存を行わない。"""
    config = _load_override_config()
    effective = _effective_overrides(config, target)
    html_path, registry_path = report_build.build(
        md_path=md_path,
        build_dir=build_dir,
        overrides=effective,
    )
    pdf_render.render_html_to_pdf(html_path, pdf_path)
    _cleanup_transient_artifacts(md_path.stem)
    return {
        "mode": "pdf-only",
        "pdf": str(pdf_path),
        "html": str(html_path),
        "source_registry": str(registry_path),
        "effective_overrides": effective,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and validate PDF report(s).")
    parser.add_argument(
        "--target",
        "-t",
        choices=tuple(REPORT_STEMS),
        default="both",
        help=(
            "Target report: 'main' (SUMMARY_REPORT), "
            "'extra' (SUMMARY_REPORT_extra), or 'both' (default)."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--auto-repair",
        action="store_true",
        help="Run the auto-repair loop and save only generated overrides for the target.",
    )
    mode.add_argument(
        "--no-repair",
        action="store_true",
        help="Build and validate once without changing layout_overrides.yml (default).",
    )
    mode.add_argument(
        "--pdf-only",
        "-p",
        action="store_true",
        help=(
            "Build HTML and PDF only from existing Markdown/results; "
            "skip validation, auto-repair, and page-image rendering."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reports = {}
    for stem in REPORT_STEMS[args.target]:
        target = TARGET_BY_STEM[stem]
        common = {
            "target": target,
            "md_path": OUTPUTS_DIR / f"{stem}.md",
            "pdf_path": OUTPUTS_DIR / f"{stem}.pdf",
            "build_dir": RENDERS_ROOT / "_build" / stem.lower(),
        }
        print(f"Building {stem}.pdf ...")
        if args.pdf_only:
            reports[stem] = build_pdf_only(**common)
            continue
        if args.auto_repair:
            reports[stem] = run_pipeline(**common)
        else:
            reports[stem] = run_without_repair(**common)

        report_path = OUTPUTS_DIR / "reports" / f"layout_{stem.lower()}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(reports[stem], ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    print(json.dumps(reports, ensure_ascii=False, indent=2, default=str))
    if args.pdf_only:
        return 0
    return 1 if any(
        report["final_result"]["status"] != "PASS" for report in reports.values()
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())
