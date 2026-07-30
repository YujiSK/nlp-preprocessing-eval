"""実験Dの時系列・グループ構造を監査し、追加分割の要否判断材料を保存する。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

TASK9_ROOT = Path(__file__).resolve().parents[2]
if str(TASK9_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK9_ROOT))

from src.experiments.preprocessing import deduplicate_by_raw_text, load_livedoor_corpus


def main() -> None:
    df = load_livedoor_corpus(TASK9_ROOT / "data_cache" / "text")
    df, removed = deduplicate_by_raw_text(df)
    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    duplicate_urls = int(df["url"].duplicated(keep=False).sum())
    duplicate_filenames = int(df["filename"].duplicated(keep=False).sum())
    audit = pd.DataFrame(
        [
            {"metric": "n_documents", "value": len(df), "decision": ""},
            {"metric": "exact_text_duplicates_removed", "value": removed, "decision": ""},
            {"metric": "duplicate_url_rows", "value": duplicate_urls, "decision": ""},
            {"metric": "duplicate_filename_rows", "value": duplicate_filenames, "decision": ""},
            {"metric": "unparseable_date_rows", "value": int(parsed_dates.isna().sum()), "decision": ""},
            {"metric": "min_date", "value": parsed_dates.min().isoformat(), "decision": ""},
            {"metric": "max_date", "value": parsed_dates.max().isoformat(), "decision": ""},
            {
                "metric": "group_split_required",
                "value": "no",
                "decision": "URL・filename重複および明示的group IDがないため",
            },
            {
                "metric": "time_split_in_primary_ablation",
                "value": "no",
                "decision": (
                    "固定コーパス内の前処理要因比較を目的とし全条件で同一StratifiedKFoldを優先。"
                    "将来時点への性能一般化には別途時系列評価が必要"
                ),
            },
        ]
    )
    out = TASK9_ROOT / "outputs" / "exp_d_extra" / "expD_temporal_group_audit.csv"
    audit.to_csv(out, index=False)
    print(audit.to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
