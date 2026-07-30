"""共通ユーティリティ: 出力先パス、共有CV分割、処理時間計測、Long形式CSV保存、環境情報の記録。

execution_plan.md 第4.2章（CV分割の共通化）、第4.6章（処理コスト計測）、
第6章（出力・保存規約）、第7章（再現性・環境固定）に対応する。
"""

from __future__ import annotations

import json
import platform
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

TASK9_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_ROOT = TASK9_ROOT / "outputs"

RANDOM_STATE = 42
N_SPLITS = 5


def ensure_output_dir(experiment_id: str) -> Path:
    """`outputs/exp_{id}` を作成して返す（第6.3章：出力先は自動生成し手動作成に依存しない）。"""
    exp_dir = OUTPUTS_ROOT / f"exp_{experiment_id.lower()}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def get_outer_cv(n_splits: int = N_SPLITS, random_state: int = RANDOM_STATE) -> StratifiedKFold:
    """全モデル・全条件で共通利用する外側CV（第4.2章）。"""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def get_outer_splits(
    X, y, n_splits: int = N_SPLITS, random_state: int = RANDOM_STATE
) -> list[tuple[np.ndarray, np.ndarray]]:
    """外側CVのFold Indexを1回だけ生成し、リストとして固定する。

    呼び出し側はこのリストを全モデル・全条件（Before/After）で使い回すこと。
    条件ごとに異なる分割を生成すると、前処理差ではなくFoldの偶然性が結果に混入する。
    """
    cv = get_outer_cv(n_splits=n_splits, random_state=random_state)
    return list(cv.split(X, y))


class _Timer:
    def __init__(self) -> None:
        self.seconds: float | None = None


@contextmanager
def timer():
    """`with timer() as t: ...` のブロック終了後に `t.seconds` で経過秒数を取得する。"""
    t = _Timer()
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.seconds = time.perf_counter() - start


def append_long_records(records: list[dict], csv_path: Path) -> pd.DataFrame:
    """Long形式（experiment, condition, model, fold, metric, value...）でCSVへ追記保存する（第6.3章）。"""
    df = pd.DataFrame(records)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return df


_ENV_PACKAGES = [
    "sklearn",
    "pandas",
    "numpy",
    "imblearn",
    "neologdn",
    "sudachipy",
    "MeCab",
    "janome",
    "matplotlib",
    "seaborn",
]


def collect_environment_info() -> dict:
    """第7章の再現性情報（実行環境・主要ライブラリのバージョン）を収集する。"""
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    for pkg in _ENV_PACKAGES:
        try:
            module = __import__(pkg)
            info[pkg] = getattr(module, "__version__", "unknown")
        except ImportError:
            info[pkg] = "not installed"
    return info


def save_environment_info(experiment_id: str) -> Path:
    """実行環境情報を `outputs/exp_{id}/exp_{id}_environment.json` として保存する。"""
    exp_dir = ensure_output_dir(experiment_id)
    path = exp_dir / f"exp_{experiment_id.lower()}_environment.json"
    path.write_text(json.dumps(collect_environment_info(), indent=2, ensure_ascii=False))
    return path
