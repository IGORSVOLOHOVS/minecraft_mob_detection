"""Reading a COCO dataset and turning it into the layout YOLO expects.

The dataset ships as COCO JSON; Ultralytics wants one `.txt` per image plus a
`data.yaml`. Everything in this module is about crossing that gap, plus the two
inspection helpers used to check the data before training on it.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

# The export carries a wrapper category covering every mob, alongside the real
# per-mob categories. Kept in the annotations it would train the detector to
# predict "some mob" as an eighteenth class competing with the seventeen real
# ones, so it is dropped during conversion.
WRAPPER_CATEGORY = "minecraft-mobs"

MINECRAFT_CLASSES: tuple[str, ...] = (
    "bee",
    "chicken",
    "cow",
    "creeper",
    "enderman",
    "fox",
    "frog",
    "ghast",
    "goat",
    "llama",
    "pig",
    "sheep",
    "skeleton",
    "spider",
    "turtle",
    "wolf",
    "zombie",
)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
SPLITS = ("train", "valid", "test")
BOX_COLOUR = (0, 255, 0)


def load_coco(json_path: str | Path) -> dict[str, Any]:
    """Read a COCO annotations file."""
    return json.loads(Path(json_path).read_text(encoding="utf-8"))


def class_names_by_id(coco: dict[str, Any]) -> dict[int, str]:
    return {category["id"]: category["name"] for category in coco["categories"]}


def analyze_coco_distribution(json_path: str | Path, save_path: str | Path) -> dict[str, int]:
    """Count annotations per class and save a bar chart, largest class first.

    Returns the counts so a caller can assert on balance rather than squint at
    the picture. An empty class still appears, with a count of zero — that is
    the case worth seeing.
    """
    coco = load_coco(json_path)
    names = class_names_by_id(coco)

    counts = dict.fromkeys(names.values(), 0)
    for annotation in coco["annotations"]:
        counts[names[annotation["category_id"]]] += 1
    ordered = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(12, 6))
    try:
        plt.bar(list(ordered), list(ordered.values()))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path)
    finally:
        # Without this every call leaks a figure, and a notebook that analyses
        # three splits ends up warning about too many open figures.
        plt.close(figure)
    return ordered


def visualize_bboxes(
    img_path: str | Path,
    json_path: str | Path,
    img_id: int | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> np.ndarray:
    """Draw a COCO image's boxes and labels, and return the annotated image.

    The image is found by file name rather than by the caller's `img_id`, which
    is only a hint: passing the wrong id for a known file used to draw another
    image's boxes silently. Returning the array lets a caller save or compare it
    without a display; `show=False` suppresses the notebook rendering.
    """
    coco = load_coco(json_path)
    names = class_names_by_id(coco)

    filename = Path(img_path).name
    found = next((image for image in coco["images"] if filename in image["file_name"]), None)
    if found is not None:
        if img_id is not None and img_id != found["id"]:
            print(f"INFO: using id {found['id']} for '{filename}', not {img_id}")
        img_id = found["id"]
    elif img_id is None:
        raise ValueError(f"Could not find image '{filename}' in annotations.")

    image = cv2.imread(str(img_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")

    for annotation in coco["annotations"]:
        if annotation["image_id"] != img_id:
            continue
        x, y, width, height = (int(value) for value in annotation["bbox"])
        label = names.get(annotation["category_id"], f"ID:{annotation['category_id']}")
        cv2.rectangle(image, (x, y), (x + width, y + height), BOX_COLOUR, 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BOX_COLOUR, 2)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), image)

    if show:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
    return image


def coco_to_yolo(json_path: str | Path, output_dir: str | Path) -> int:
    """Write one YOLO `.txt` per image; return how many were written.

    YOLO wants class index, then centre-x, centre-y, width and height as
    fractions of the image. COCO gives a top-left corner and a size in pixels,
    so both the origin and the units change here.

    Images with no annotations still get an empty file: YOLO reads a missing
    label file as "unlabelled" and an empty one as "nothing in this image",
    which is what an empty scene actually means.
    """
    coco = load_coco(json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    real_categories = sorted(
        (c for c in coco["categories"] if c["name"] != WRAPPER_CATEGORY),
        key=lambda category: category["id"],
    )
    class_index = {category["id"]: index for index, category in enumerate(real_categories)}

    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image[annotation["image_id"]].append(annotation)

    for image in coco["images"]:
        width, height = float(image["width"]), float(image["height"])
        lines = []
        for annotation in by_image.get(image["id"], []):
            if annotation["category_id"] not in class_index:
                continue
            x, y, box_width, box_height = annotation["bbox"]
            lines.append(
                f"{class_index[annotation['category_id']]} "
                f"{(x + box_width / 2) / width:.6f} "
                f"{(y + box_height / 2) / height:.6f} "
                f"{box_width / width:.6f} {box_height / height:.6f}"
            )
        target = output_dir / f"{Path(image['file_name']).stem}.txt"
        target.write_text("\n".join(lines), encoding="utf-8")

    return len(coco["images"])


def prepare_yolo_dataset(
    src_root: str | Path,
    dst_root: str | Path,
    class_names: tuple[str, ...] | list[str] | None = None,
) -> Path:
    """Build a complete YOLO dataset from the COCO splits; return the data.yaml.

    A missing split is reported and skipped rather than raising: a dataset with
    no test split is a normal thing to hand this function.
    """
    src_root, dst_root = Path(src_root), Path(dst_root)
    names = tuple(class_names) if class_names is not None else MINECRAFT_CLASSES

    for split in SPLITS:
        source = src_root / split
        if not source.is_dir():
            print(f"Warning: {source} not found, skipping this split")
            continue
        print(f"Processing {split} split...")

        images_out = dst_root / split / "images"
        labels_out = dst_root / split / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        annotations = source / "annotations.json"
        if annotations.is_file():
            converted = coco_to_yolo(annotations, labels_out)
            print(f"  converted {converted} images to YOLO labels")
        else:
            print(f"Warning: {annotations} not found")

        for entry in source.iterdir():
            if entry.suffix.lower() in IMAGE_SUFFIXES:
                shutil.copy2(entry, images_out / entry.name)

    # The per-split loop above creates this, unless every split was missing -
    # in which case data.yaml still has to land somewhere.
    dst_root.mkdir(parents=True, exist_ok=True)
    data_yaml = dst_root / "data.yaml"
    data_yaml.write_text(
        f"path: {os.path.abspath(dst_root)}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "\n"
        f"nc: {len(names)}\n"
        f"names: {list(names)}\n",
        encoding="utf-8",
    )
    print(f"Created {data_yaml}")
    return data_yaml
