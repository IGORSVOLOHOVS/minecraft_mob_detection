"""The branches the first two suites leave out, and the `utils` shim the notebook imports.

Everything here runs on files the test writes itself: no dataset, no weights and no
network are needed, and importing `mobtools` must not pull in the training stack.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import mobtools
import utils
from mobtools import dataset, report
from mobtools.dataset import prepare_yolo_dataset, visualize_bboxes
from mobtools.metrics import read_yolo_final_epoch

ROOT = Path(__file__).resolve().parent.parent
COCO = {
    "categories": [{"id": 2, "name": "creeper"}],
    "images": [
        {"id": 10, "file_name": "first.jpg", "width": 200, "height": 100},
        {"id": 11, "file_name": "second.jpg", "width": 200, "height": 100},
    ],
    # The box of image 11 must not be drawn on image 10.
    "annotations": [
        {"id": 1, "image_id": 10, "category_id": 2, "bbox": [0, 0, 100, 50]},
        {"id": 2, "image_id": 11, "category_id": 2, "bbox": [150, 60, 40, 30]},
    ],
}


@pytest.fixture
def annotations(tmp_path: Path) -> Path:
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(COCO), encoding="utf-8")
    return path


def black_image(path: Path) -> Path:
    Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8)).save(path)
    return path


def test_the_shim_hands_the_notebook_the_same_objects() -> None:
    assert utils.__all__ == mobtools.__all__
    for name in utils.__all__:
        assert getattr(utils, name) is getattr(mobtools, name), name


def test_importing_the_helpers_loads_no_training_stack() -> None:
    # A fresh interpreter, so another test's imports cannot hide one made by mobtools.
    probe = (
        "import sys, mobtools; "
        "print(sorted(m for m in ('torch', 'ultralytics', 'mmdet', 'mmcv') if m in sys.modules))"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True, check=True
    )
    assert done.stdout.strip() == "[]"


def test_a_wrong_id_for_a_known_file_is_corrected_and_said(
    annotations: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    image = black_image(tmp_path / "first.jpg")
    drawn = visualize_bboxes(image, annotations, img_id=99, show=False)

    assert "using id 10 for 'first.jpg', not 99" in capsys.readouterr().out
    assert drawn[:50, :100].any(), "the box of image 10 was drawn"
    assert not drawn[60:, 150:].any(), "the box of image 11 was not"


def test_an_unreadable_image_file_is_reported(annotations: Path, tmp_path: Path) -> None:
    broken = tmp_path / "first.jpg"
    broken.write_text("not an image", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Could not load image"):
        visualize_bboxes(broken, annotations, show=False)


def test_show_hands_the_picture_to_matplotlib(
    annotations: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shown: list[bool] = []
    monkeypatch.setattr(dataset.plt, "show", lambda: shown.append(True))

    visualize_bboxes(black_image(tmp_path / "first.jpg"), annotations, show=True)

    assert shown == [True]
    dataset.plt.close("all")


def test_a_split_without_annotations_still_copies_its_images(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "coco" / "train"
    source.mkdir(parents=True)
    black_image(source / "first.jpg")

    prepare_yolo_dataset(tmp_path / "coco", tmp_path / "yolo")

    assert f"Warning: {source / 'annotations.json'} not found" in capsys.readouterr().out
    assert (tmp_path / "yolo" / "train" / "images" / "first.jpg").is_file()
    assert not any((tmp_path / "yolo" / "train" / "labels").iterdir())


def test_renamed_ultralytics_columns_are_named(tmp_path: Path) -> None:
    results = tmp_path / "results.csv"
    pd.DataFrame([{"epoch": 1, "metrics/mAP50(B)": 0.2}]).to_csv(results, index=False)

    with pytest.raises(KeyError, match=r"mAP50-95\(B\)"):
        read_yolo_final_epoch(results)


def test_without_the_bundled_fonts_the_report_falls_back_and_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(report, "FONT_DIR", tmp_path)

    pdf = report.PDFReport()

    assert pdf.body_font == report.FALLBACK_FONT
    assert "Cyrillic will not render" in capsys.readouterr().out
