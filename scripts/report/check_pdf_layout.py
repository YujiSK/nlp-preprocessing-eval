"""生成済みの本編・発展版PDFをまとめて再検査する。"""

from __future__ import annotations

import sys
from pathlib import Path

TASK9_ROOT = Path(__file__).resolve().parents[2]
if str(TASK9_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK9_ROOT))

from src.reporting.layout_checker import main_all_reports


if __name__ == "__main__":
    raise SystemExit(main_all_reports())
