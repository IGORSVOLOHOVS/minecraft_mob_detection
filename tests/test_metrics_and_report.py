"""Tests for the comparison table and the PDF report.

Both previously failed in ways that pointed at the wrong thing: an empty
results file raised IndexError from inside pandas, and a missing metrics CSV
produced a report with a silently absent section.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from mobtools.metrics import compare_metrics, read_fcos_final_epoch, read_yolo_final_epoch
from mobtools.report import generate_report

YOLO_ROWS = [
    {"epoch": 1, "metrics/mAP50-95(B)": 0.10, "metrics/mAP50(B)": 0.20},
    {"epoch": 2, "metrics/mAP50-95(B)": 0.31, "metrics/mAP50(B)": 0.62},
]
FCOS_LINES = [
    {"mode": "train", "loss": 1.0},
    {"coco/bbox_mAP": 0.11, "coco/bbox_mAP_50": 0.22},
    {"coco/bbox_mAP": 0.27, "coco/bbox_mAP_50": 0.55},
]


@pytest.fixture
def yolo_csv(tmp_path: Path) -> Path:
    path = tmp_path / "results.csv"
    pd.DataFrame(YOLO_ROWS).to_csv(path, index=False)
    return path


@pytest.fixture
def fcos_log(tmp_path: Path) -> Path:
    path = tmp_path / "fcos.log.json"
    path.write_text("\n".join(json.dumps(line) for line in FCOS_LINES), encoding="utf-8")
    return path


def test_the_last_epoch_is_the_one_reported(yolo_csv: Path) -> None:
    assert read_yolo_final_epoch(yolo_csv) == (0.31, 0.62)


def test_training_lines_are_skipped(fcos_log: Path) -> None:
    """Only validation entries carry bbox_mAP; the rest must not be parsed as one."""
    assert read_fcos_final_epoch(fcos_log) == (0.27, 0.55)


def test_a_truncated_final_line_does_not_break_parsing(tmp_path: Path) -> None:
    """An interrupted run leaves half a line; that is not a reason to fail."""
    path = tmp_path / "fcos.log.json"
    body = "\n".join(json.dumps(line) for line in FCOS_LINES)
    path.write_text(body + '\n{"coco/bbox_mAP": 0.9, "coco/bbox_m', encoding="utf-8")

    assert read_fcos_final_epoch(path) == (0.27, 0.55)


def test_an_empty_results_file_says_so(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    pd.DataFrame(columns=["epoch", "metrics/mAP50-95(B)", "metrics/mAP50(B)"]).to_csv(
        path, index=False
    )

    with pytest.raises(ValueError, match="no rows"):
        read_yolo_final_epoch(path)


def test_a_log_without_validation_says_so(tmp_path: Path) -> None:
    path = tmp_path / "fcos.log.json"
    path.write_text(json.dumps({"mode": "train", "loss": 1.0}), encoding="utf-8")

    with pytest.raises(ValueError, match="no validation entries"):
        read_fcos_final_epoch(path)


def test_comparison_holds_both_models(yolo_csv: Path, fcos_log: Path, tmp_path: Path) -> None:
    table = compare_metrics(yolo_csv, fcos_log, tmp_path / "out" / "compare.csv")

    assert list(table["Model"]) == ["YOLOv8s", "FCOS"]
    assert list(table["mAP"]) == [0.31, 0.27]
    assert (tmp_path / "out" / "compare.csv").is_file()


def _chart(tmp_path: Path) -> Path:
    path = tmp_path / "chart.png"
    Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8)).save(path)
    return path


def test_report_is_written(tmp_path: Path, yolo_csv: Path, fcos_log: Path) -> None:
    compare_metrics(yolo_csv, fcos_log, tmp_path / "compare.csv")

    written = generate_report(
        _chart(tmp_path), tmp_path / "compare.csv", tmp_path / "artifacts" / "report.pdf"
    )

    assert written.is_file()
    assert written.read_bytes().startswith(b"%PDF")


def test_report_survives_a_missing_comparison(tmp_path: Path) -> None:
    """The chart alone is still a readable report."""
    written = generate_report(_chart(tmp_path), tmp_path / "not-here.csv", tmp_path / "report.pdf")

    assert written.is_file()


def test_a_missing_chart_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="distribution chart"):
        generate_report(tmp_path / "nowhere.png", tmp_path / "x.csv", tmp_path / "r.pdf")
