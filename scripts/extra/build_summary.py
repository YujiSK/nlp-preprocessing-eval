"""SUMMARY_REPORT.mdへ発展Appendixを再現可能に結合する。"""

from __future__ import annotations

import re
from pathlib import Path

TASK9_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = TASK9_ROOT / "outputs"
BASE_REPORT = OUTPUTS / "SUMMARY_REPORT.md"
EXTRA_REPORT = OUTPUTS / "SUMMARY_REPORT_extra.md"

EXTRA_EXPERIMENTS = [
    ("exp_a_extra", "APPENDIX_EXP_A_PERMUTATION.md"),
    ("exp_b_extra", "APPENDIX_EXP_B_COVERAGE.md"),
    ("exp_c_extra", "APPENDIX_EXP_C_THRESHOLD.md"),
    ("exp_d_extra", "APPENDIX_EXP_D_ABLATION.md"),
]


def adapt_extra_experiment(directory: str, filename: str) -> str:
    """単体Appendixを、統合版の付録C配下に置く実験セクションへ変換する。"""
    path = OUTPUTS / directory / filename
    text = path.read_text(encoding="utf-8").strip()
    lines = text.splitlines()

    title = lines[0].lstrip("#").strip()
    title = re.sub(r"^APPENDIX_EXP_[A-D]_[A-Z_]+\s*—\s*", "", title)
    title = re.sub(r"^Appendix\s+[A-D]\s*—\s*", "", title, flags=re.IGNORECASE)
    lines[0] = f"### {title}"

    # 単体Appendix内のh2を、統合版では実験タイトル配下のh4へ下げる。
    for index in range(1, len(lines)):
        if lines[index].startswith("## "):
            lines[index] = "##" + lines[index]

    text = "\n".join(lines)
    # Appendix単体では同一ディレクトリ参照、統合版ではoutputs/からの相対参照へ変換。
    text = re.sub(
        r"\]\((?!https?://|/)([^/)]+\.png)\)",
        rf"]({directory}/\1)",
        text,
    )
    return text


def main() -> None:
    base = BASE_REPORT.read_text(encoding="utf-8").rstrip()
    sections = [
        adapt_extra_experiment(directory, filename)
        for directory, filename in EXTRA_EXPERIMENTS
    ]
    appendix_c = "## 付録C：発展実験\n\n" + "\n\n---\n\n".join(sections)
    integrated = base + "\n\n---\n\n" + appendix_c + "\n"
    EXTRA_REPORT.write_text(integrated, encoding="utf-8")
    print(f"wrote {EXTRA_REPORT}")
    print(f"base chars={len(base):,}, integrated chars={len(integrated):,}")


if __name__ == "__main__":
    main()
