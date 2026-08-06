"""Tests for the dataset helpers.

The COCO-to-YOLO conversion is the piece where a silent mistake is expensive:
a wrong normalisation trains the detector on boxes in the wrong place, and
nothing complains. These pin the arithmetic against hand-computed values.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mobtools.dataset import (
    MINECRAFT_CLASSES,
    WRAPPER_CATEGORY,
    analyze_coco_distribution,
    coco_to_yolo,
    prepare_yolo_dataset,
    visualize_bboxes,
)

# One 200x100 image with two boxes, one 100x100 image with none. The wrapper
# category is present, as it is in the real export.
COCO = {
    "categories": [
        {"id": 1, "name": WRAPPER_CATEGORY},
        {"id": 2, "name": "creeper"},
        {"id": 3, "name": "zombie"},
    ],
    "images": [
        {"id": 10, "file_name": "first.jpg", "width": 200, "height": 100},
        {"id": 11, "file_name": "second.jpg", "width": 100, "height": 100},
    ],
    "annotations": [
        {"id": 1, "image_id": 10, "category_id": 2, "bbox": [0, 0, 100, 50]},
        {"id": 2, "image_id": 10, "category_id": 3, "bbox": [100, 50, 100, 50]},
        {"id": 3, "image_id": 10, "category_id": 1, "bbox": [0, 0, 200, 100]},
    ],
}


@pytest.fixture
def annotations(tmp_path: Path) -> Path:
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(COCO), encoding="utf-8")
    return path


def test_distribution_counts_every_class(annotations: Path, tmp_path: Path) -> None:
    counts = analyze_coco_distribution(annotations, tmp_path / "chart.png")

    assert counts == {WRAPPER_CATEGORY: 1, "creeper": 1, "zombie": 1}
    assert (tmp_path / "chart.png").is_file()


def test_distribution_is_ordered_largest_first(tmp_path: Path) -> None:
    coco = json.loads(json.dumps(COCO))
    coco["annotations"] += [
        {"id": 4, "image_id": 10, "category_id": 3, "bbox": [0, 0, 10, 10]},
        {"id": 5, "image_id": 10, "category_id": 3, "bbox": [0, 0, 10, 10]},
    ]
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(coco), encoding="utf-8")

    counts = analyze_coco_distribution(path, tmp_path / "chart.png")

    assert next(iter(counts)) == "zombie"


def test_yolo_normalisation_is_centre_and_fraction(annotations: Path, tmp_path: Path) -> None:
    """A 100x50 box at the origin of a 200x100 image: centre (0.25, 0.25), size (0.5, 0.5)."""
    coco_to_yolo(annotations, tmp_path / "labels")

    lines = (tmp_path / "labels" / "first.txt").read_text(encoding="utf-8").splitlines()

    assert lines[0] == "0 0.250000 0.250000 0.500000 0.500000"
    assert lines[1] == "1 0.750000 0.750000 0.500000 0.500000"


def test_the_wrapper_category_is_dropped(annotations: Path, tmp_path: Path) -> None:
    """Keeping it would train an eighteenth class competing with the real ones."""
    coco_to_yolo(annotations, tmp_path / "labels")

    lines = (tmp_path / "labels" / "first.txt").read_text(encoding="utf-8").splitlines()
    indices = {line.split()[0] for line in lines}

    assert len(lines) == 2
    assert indices == {"0", "1"}


def test_an_image_with_no_annotations_gets_an_empty_file(annotations: Path, tmp_path: Path) -> None:
    """YOLO reads a missing file as unlabelled and an empty one as an empty scene."""
    coco_to_yolo(annotations, tmp_path / "labels")

    second = tmp_path / "labels" / "second.txt"

    assert second.is_file()
    assert second.read_text(encoding="utf-8") == ""


def test_prepare_builds_the_layout_yolo_expects(tmp_path: Path) -> None:
    source = tmp_path / "src" / "train"
    source.mkdir(parents=True)
    (source / "annotations.json").write_text(json.dumps(COCO), encoding="utf-8")
    Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(source / "first.jpg")

    data_yaml = prepare_yolo_dataset(tmp_path / "src", tmp_path / "dst")

    assert (tmp_path / "dst" / "train" / "images" / "first.jpg").is_file()
    assert (tmp_path / "dst" / "train" / "labels" / "first.txt").is_file()
    text = data_yaml.read_text(encoding="utf-8")
    assert f"nc: {len(MINECRAFT_CLASSES)}" in text
    assert "train: train/images" in text


def test_a_missing_split_is_skipped_not_fatal(tmp_path: Path) -> None:
    """A dataset without a test split is an ordinary thing to be handed."""
    (tmp_path / "src").mkdir()

    data_yaml = prepare_yolo_dataset(tmp_path / "src", tmp_path / "dst")

    assert data_yaml.is_file()


def test_visualising_returns_the_annotated_image(annotations: Path, tmp_path: Path) -> None:
    image_path = tmp_path / "first.jpg"
    Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8)).save(image_path)

    drawn = visualize_bboxes(image_path, annotations, save_path=tmp_path / "out.jpg", show=False)

    assert drawn.shape == (100, 200, 3)
    assert drawn.any(), "boxes were drawn, so the image cannot still be all black"
    assert (tmp_path / "out.jpg").is_file()


def test_an_unknown_image_is_reported(annotations: Path, tmp_path: Path) -> None:
    missing = tmp_path / "nowhere.jpg"
    Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(missing)

    with pytest.raises(ValueError, match="Could not find image"):
        visualize_bboxes(missing, annotations, show=False)
