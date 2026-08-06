"""Helpers for the Minecraft mob detection coursework.

The rest of this repository is a fork of mmdetection. This package is the part
written for the project: preparing the dataset, comparing the two detectors and
producing the report.
"""

from .dataset import (
    MINECRAFT_CLASSES,
    analyze_coco_distribution,
    coco_to_yolo,
    prepare_yolo_dataset,
    visualize_bboxes,
)
from .metrics import compare_metrics
from .report import PDFReport, generate_report

__all__ = [
    "MINECRAFT_CLASSES",
    "PDFReport",
    "analyze_coco_distribution",
    "coco_to_yolo",
    "compare_metrics",
    "generate_report",
    "prepare_yolo_dataset",
    "visualize_bboxes",
]
