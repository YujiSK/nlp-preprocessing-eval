"""src/reporting 配下のPDF検証・統合ロジックをテストする。

- 検出ロジック（孤立見出し・図とキャプションの分離・短いコード/表の分割・はみ出し）は、
  実際のPDF生成を伴わない合成 PageData / registry を用いた単体テストで検証する。
- 実際のビルド〜PDF化〜検査の一連の流れは、tests/fixtures/sample_layout.md を用いた
  結合テストで確認する（トレイリング見出し、図とキャプション、長短のコード・表を含む）。
- 修復ループ（最大3回・ロールバック）は、`_build_render_check` をモックして
  イテレーションごとのスコア推移を制御し、停止条件を検証する。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

TASK9_ROOT = Path(__file__).resolve().parent.parent
if str(TASK9_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK9_ROOT))

from src.reporting import layout_checker as checker
from src.reporting import layout_pipeline as pipeline
from src.reporting import pdf_renderer as pdf_render
from src.reporting import report_builder as report_build

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def make_page(number, text, images=0, words=None, width=420.0, height=595.0):
    compact = checker._normalize(text)
    return checker.PageData(
        number=number,
        width=width,
        height=height,
        compact_text=compact,
        n_images=images,
        words=words or [],
        trigrams=frozenset(checker._trigrams(compact)),
    )


# ---------------------------------------------------------------------------
# 孤立見出し（orphan_heading）
# ---------------------------------------------------------------------------


def test_orphan_heading_detected_when_pages_differ():
    pages = [
        make_page(1, "本文A" * 20 + "見出しテスト小見出し"),
        make_page(2, "見出しの直後に続くはずの本文段落テキストです"),
    ]
    registry = [
        {
            "id": "heading-x",
            "type": "heading",
            "level": "h3",
            "text": "見出しテスト小見出し",
            "next_block_text": "見出しの直後に続くはずの本文段落テキストです",
        }
    ]
    resolved = checker.resolve_registry_pages(pages, registry)
    violations, unresolved = checker.detect_orphan_headings(pages, registry, resolved)

    assert unresolved == []
    assert len(violations) == 1
    assert violations[0]["type"] == "orphan_heading"
    assert violations[0]["pages"] == [1, 2]


def test_orphan_heading_ok_when_same_page():
    pages = [make_page(1, "見出しテスト小見出し 見出しの直後に続くはずの本文段落テキストです")]
    registry = [
        {
            "id": "heading-x",
            "type": "heading",
            "level": "h3",
            "text": "見出しテスト小見出し",
            "next_block_text": "見出しの直後に続くはずの本文段落テキストです",
        }
    ]
    resolved = checker.resolve_registry_pages(pages, registry)
    violations, unresolved = checker.detect_orphan_headings(pages, registry, resolved)

    assert violations == []
    assert unresolved == []


def test_trailing_heading_with_no_next_block_is_skipped():
    """文書末尾の見出し（next_block_textが空）は比較対象がないため違反として扱わない。"""
    pages = [make_page(1, "末尾の小見出しテキスト")]
    registry = [
        {
            "id": "heading-last",
            "type": "heading",
            "level": "h3",
            "text": "末尾の小見出しテキスト",
            "next_block_text": "",
        }
    ]
    resolved = checker.resolve_registry_pages(pages, registry)
    violations, unresolved = checker.detect_orphan_headings(pages, registry, resolved)

    assert violations == []
    assert unresolved == []


# ---------------------------------------------------------------------------
# 図とキャプションの分離（figure_caption_image_split）
# ---------------------------------------------------------------------------


def test_figure_caption_split_detected_when_no_image_on_caption_page():
    pages = [
        make_page(1, "何らかの本文", images=1),
        make_page(2, "図1.1サンプルキャプションテキストです", images=0),
    ]
    registry = [{"id": "figure-1-1", "type": "figure", "caption": "図1.1サンプルキャプションテキストです", "img_src": "x.png"}]
    resolved = checker.resolve_registry_pages(pages, registry)
    violations, unresolved = checker.detect_figure_caption_split(pages, registry, resolved)

    assert unresolved == []
    assert len(violations) == 1
    assert violations[0]["type"] == "figure_caption_image_split"


def test_figure_caption_ok_when_image_present_on_same_page():
    pages = [make_page(1, "図1.1サンプルキャプションテキストです", images=1)]
    registry = [{"id": "figure-1-1", "type": "figure", "caption": "図1.1サンプルキャプションテキストです", "img_src": "x.png"}]
    resolved = checker.resolve_registry_pages(pages, registry)
    violations, unresolved = checker.detect_figure_caption_split(pages, registry, resolved)

    assert violations == []
    assert unresolved == []


# ---------------------------------------------------------------------------
# 短いコード・表の分割（short_block_split）
# ---------------------------------------------------------------------------


def test_short_table_split_detected():
    pages = [
        make_page(1, "列Aヘッダーと列Bヘッダーの先頭行データ"),
        make_page(2, "テーブル末尾行のデータテキスト内容です"),
    ]
    registry = [
        {
            "id": "table-1-1",
            "type": "table",
            "n_rows": 3,
            "size_class": "small-table",
            "first_row_text": "列Aヘッダーと列Bヘッダーの先頭行データ",
            "last_row_text": "テーブル末尾行のデータテキスト内容です",
        }
    ]
    resolved = checker.resolve_registry_pages(pages, registry)
    violations, unresolved = checker.detect_short_block_split(pages, registry, resolved)

    assert unresolved == []
    assert len(violations) == 1
    assert violations[0]["type"] == "short_block_split"


def test_long_table_split_is_excluded_from_check():
    """long-table（分割許可）は対象外のため、ページが分かれていても違反にしない。"""
    pages = [
        make_page(1, "長い表の先頭行テキストです"),
        make_page(2, "長い表の末尾行テキストです"),
    ]
    registry = [
        {
            "id": "table-1-2",
            "type": "table",
            "n_rows": 20,
            "size_class": "long-table",
            "first_row_text": "長い表の先頭行テキストです",
            "last_row_text": "長い表の末尾行テキストです",
        }
    ]
    resolved = checker.resolve_registry_pages(pages, registry)
    violations, unresolved = checker.detect_short_block_split(pages, registry, resolved)

    assert violations == []
    assert unresolved == []


def test_short_code_split_detected():
    pages = [
        make_page(1, "importsklearnfromsklearnimportPipeline"),
        make_page(2, "print(finalresultvalue)"),
    ]
    registry = [
        {
            "id": "code-1-1",
            "type": "code",
            "n_lines": 5,
            "size_class": "short-code",
            "first_line": "importsklearnfromsklearnimportPipeline",
            "last_line": "print(finalresultvalue)",
        }
    ]
    resolved = checker.resolve_registry_pages(pages, registry)
    violations, unresolved = checker.detect_short_block_split(pages, registry, resolved)

    assert len(violations) == 1
    assert violations[0]["type"] == "short_block_split"


# ---------------------------------------------------------------------------
# はみ出し（overflow）
# ---------------------------------------------------------------------------


def test_overflow_detected_for_word_beyond_bottom_margin():
    margin_pt = 18 * checker.MM_TO_PT
    page = make_page(
        1,
        "本文",
        words=[{"text": "overflow_word", "x0": 100, "x1": 150, "top": 10, "bottom": 595 - margin_pt + 10}],
    )
    violations = checker.detect_overflow([page])
    assert len(violations) == 1
    assert violations[0]["type"] == "overflow"


def test_no_overflow_for_word_within_margins():
    page = make_page(1, "本文", words=[{"text": "ok_word", "x0": 100, "x1": 150, "top": 10, "bottom": 300}])
    violations = checker.detect_overflow([page])
    assert violations == []


# ---------------------------------------------------------------------------
# 結合テスト: tests/fixtures/sample_layout.md を実際にビルド〜PDF化〜検査する
# ---------------------------------------------------------------------------


@pytest.fixture
def fixture_workdir():
    """pdf_render.render_html_to_pdf は outputs/ 配下のHTMLしか扱えないため、
    outputs/_test_fixture_tmp/ にフィクスチャ一式をコピーして作業する。
    """
    workdir = report_build.TASK9_ROOT / "outputs" / "_test_fixture_tmp"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    shutil.copy(FIXTURES_DIR / "sample_layout.md", workdir / "sample_layout.md")
    shutil.copy(FIXTURES_DIR / "sample_image.png", workdir / "sample_image.png")
    try:
        yield workdir
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_fixture_end_to_end_zero_violations(fixture_workdir):
    """トレイリング見出し・図/キャプション・長短コード/表を含むサンプル文書が、
    ビルド〜PDF化〜検査の一連の流れをエラーなく通過し、想定通り違反0件になることを確認する。
    """
    md_path = fixture_workdir / "sample_layout.md"
    registry_build_dir = fixture_workdir / "_build"
    missing_overrides = fixture_workdir / "no_overrides.yml"

    html_path, registry_path = report_build.build(
        md_path=md_path,
        overrides_path=missing_overrides,
        css_path=report_build.DEFAULT_CSS_PATH,
        build_dir=registry_build_dir,
    )

    pdf_path = fixture_workdir / "sample_layout.pdf"
    pdf_render.render_html_to_pdf(html_path, pdf_path)

    result = checker.run_checks(pdf_path, registry_path)

    assert result["page_count"] >= 1
    assert result["n_registry_entries"] >= 6  # heading x複数 + figure + table x2 + code x2
    assert result["unresolved"] == []
    assert result["violations"] == [], result["violations"]


# ---------------------------------------------------------------------------
# 修復ループ（最大3回・ロールバック）
# ---------------------------------------------------------------------------


def _fake_result(score, violation_types):
    return {
        "pdf": "dummy.pdf",
        "page_count": 1,
        "n_registry_entries": 1,
        "violations": [{"type": t, "id": f"x-{i}", "detail": "", "pages": [1]} for i, t in enumerate(violation_types)],
        "unresolved": [],
        "score": score,
    }


def test_repair_loop_stops_when_zero_violations(monkeypatch, tmp_path):
    calls = {"n": 0}
    sequence = [
        _fake_result(-2, ["orphan_heading"]),
        _fake_result(0, []),
    ]

    def fake_build_render_check():
        calls["n"] += 1
        return sequence[min(calls["n"] - 1, len(sequence) - 1)]

    monkeypatch.setattr(pipeline, "_build_render_check", fake_build_render_check)
    monkeypatch.setattr(pipeline, "OVERRIDES_PATH", tmp_path / "layout_overrides.yml")
    monkeypatch.setattr(pipeline.pdf_render, "render_pdf_to_page_images", lambda pdf_path, out_dir: [])

    summary = pipeline.run_pipeline(max_iterations=3)

    assert summary["final_result"]["score"] == 0
    assert calls["n"] == 3  # iter1(-2) -> iter2(0, stop) -> final confirmation build


def test_repair_loop_rolls_back_when_no_improvement(monkeypatch, tmp_path):
    calls = {"n": 0}
    # iteration1: score -2 (accepted, best so far) -> iteration2: score -4 (worse, rollback)
    sequence = [
        _fake_result(-2, ["orphan_heading"]),
        _fake_result(-4, ["orphan_heading", "figure_caption_image_split"]),
        _fake_result(-2, ["orphan_heading"]),  # rollback re-build
        _fake_result(-2, ["orphan_heading"]),  # final confirmation build
    ]

    def fake_build_render_check():
        calls["n"] += 1
        return sequence[min(calls["n"] - 1, len(sequence) - 1)]

    monkeypatch.setattr(pipeline, "_build_render_check", fake_build_render_check)
    monkeypatch.setattr(pipeline, "OVERRIDES_PATH", tmp_path / "layout_overrides.yml")
    monkeypatch.setattr(pipeline.pdf_render, "render_pdf_to_page_images", lambda pdf_path, out_dir: [])

    summary = pipeline.run_pipeline(max_iterations=3)

    actions = [h.get("action") for h in summary["history"]]
    assert "rejected_rollback" in actions
    assert "rollback_to_iteration" in actions
    assert summary["final_result"]["score"] == -2
    assert summary["best_iteration"] == 1


@pytest.mark.parametrize(
    ("target", "expected_stems"),
    [
        ("main", ["SUMMARY_REPORT"]),
        ("extra", ["SUMMARY_REPORT_extra"]),
        ("both", ["SUMMARY_REPORT", "SUMMARY_REPORT_extra"]),
    ],
)
def test_pipeline_main_selects_requested_reports(monkeypatch, tmp_path, target, expected_stems):
    """--targetに応じて指定レポートだけをビルドする。"""
    calls = []

    def fake_run_pipeline(**kwargs):
        calls.append(kwargs["md_path"].stem)
        return {"final_result": {"violations": []}}

    monkeypatch.setattr(pipeline, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "RENDERS_ROOT", tmp_path / "renders")
    monkeypatch.setattr(pipeline, "run_pipeline", fake_run_pipeline)

    assert pipeline.main(["--target", target]) == 0
    assert calls == expected_stems


def test_pipeline_parse_args_rejects_unknown_target():
    """未定義のビルド対象はargparseで拒否する。"""
    with pytest.raises(SystemExit):
        pipeline.parse_args(["--target", "unknown"])


def test_pipeline_parse_args_accepts_pdf_only_short_option():
    """-pでPDF描画専用モードを選択できる。"""
    args = pipeline.parse_args(["--target", "extra", "-p"])
    assert args.target == "extra"
    assert args.pdf_only is True


def test_pipeline_main_pdf_only_skips_validation(monkeypatch, tmp_path):
    """PDF専用モードでは検査パイプラインを呼ばず、描画専用経路だけを実行する。"""
    calls = []

    def fake_build_pdf_only(**kwargs):
        calls.append(kwargs["md_path"].stem)
        return {"mode": "pdf-only"}

    def fail_run_pipeline(**kwargs):
        raise AssertionError("run_pipeline must not run in --pdf-only mode")

    monkeypatch.setattr(pipeline, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "RENDERS_ROOT", tmp_path / "renders")
    monkeypatch.setattr(pipeline, "build_pdf_only", fake_build_pdf_only)
    monkeypatch.setattr(pipeline, "run_pipeline", fail_run_pipeline)

    assert pipeline.main(["--target", "both", "--pdf-only"]) == 0
    assert calls == ["SUMMARY_REPORT", "SUMMARY_REPORT_extra"]
