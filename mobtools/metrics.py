"""Putting the YOLO and FCOS results side by side.

The two trainers report in different formats - Ultralytics writes a CSV row per
epoch, MMDetection writes one JSON object per line - so comparing them at all
means reading both and picking the final epoch out of each.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

YOLO_MAP = "metrics/mAP50-95(B)"
YOLO_MAP50 = "metrics/mAP50(B)"
FCOS_MAP = "coco/bbox_mAP"
FCOS_MAP50 = "coco/bbox_mAP_50"


def read_yolo_final_epoch(results_csv: str | Path) -> tuple[float, float]:
    """The last epoch's mAP and mAP@50 from an Ultralytics results.csv."""
    frame = pd.read_csv(results_csv)
    if frame.empty:
        raise ValueError(f"{results_csv} has no rows - did training finish?")
    missing = [column for column in (YOLO_MAP, YOLO_MAP50) if column not in frame]
    if missing:
        raise KeyError(f"{results_csv} is missing {missing}; Ultralytics renamed them?")
    return float(frame[YOLO_MAP].iloc[-1]), float(frame[YOLO_MAP50].iloc[-1])


def read_fcos_final_epoch(log_path: str | Path) -> tuple[float, float]:
    """The last validation entry's mAP and mAP@50 from an MMDetection log.

    The log interleaves training and validation lines; only validation lines
    carry `bbox_mAP`, so the rest are skipped rather than parsed and discarded.
    """
    entries = []
    for line in Path(log_path).read_text(encoding="utf-8").splitlines():
        if FCOS_MAP not in line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a truncated final line is normal in an interrupted run
    if not entries:
        raise ValueError(f"{log_path} contains no validation entries")
    last = entries[-1]
    return float(last[FCOS_MAP]), float(last[FCOS_MAP50])


def compare_metrics(
    yolo_csv: str | Path, fcos_log: str | Path, save_path: str | Path
) -> pd.DataFrame:
    """Collect both models' final metrics into one table and save it.

    Previously this indexed the last row without checking there was one, so an
    interrupted training run failed with an opaque IndexError rather than
    saying which file was empty.
    """
    yolo_map, yolo_map50 = read_yolo_final_epoch(yolo_csv)
    fcos_map, fcos_map50 = read_fcos_final_epoch(fcos_log)

    comparison = pd.DataFrame(
        {
            "Model": ["YOLOv8s", "FCOS"],
            "mAP": [yolo_map, fcos_map],
            "mAP_50": [yolo_map50, fcos_map50],
        }
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(save_path, index=False)
    return comparison
