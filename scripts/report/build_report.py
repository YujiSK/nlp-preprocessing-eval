"""安全な既定動作でレポートを構築する、reporting層の薄いCLIラッパー。"""

from __future__ import annotations

import sys
from pathlib import Path

TASK9_ROOT = Path(__file__).resolve().parents[2]
if str(TASK9_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK9_ROOT))

from src.reporting.layout_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
