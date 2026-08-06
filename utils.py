"""Compatibility shim: the helpers now live in the `mobtools` package.

`notebook.ipynb` imports these names from `utils`, and a notebook that has
already been run and committed should not have to be edited to keep working.
New code should import from `mobtools` directly.
"""

from mobtools import (
    MINECRAFT_CLASSES,
    PDFReport,
    analyze_coco_distribution,
    coco_to_yolo,
    compare_metrics,
    generate_report,
    prepare_yolo_dataset,
    visualize_bboxes,
)

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
