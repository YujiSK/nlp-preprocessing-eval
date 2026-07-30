import sys
from pathlib import Path

TASK9_ROOT = Path(__file__).resolve().parent.parent
if str(TASK9_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK9_ROOT))
