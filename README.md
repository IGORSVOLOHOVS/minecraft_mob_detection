# minecraft-mob-detection

Coursework that fine-tunes two object detectors, FCOS (MMDetection) and YOLOv8 (Ultralytics), on a COCO dataset of 17 Minecraft mobs, and compares them on accuracy (mAP), inference speed and the look of their predictions on test images and video.

The repository has two halves:

- **This project's own code.** `notebook.ipynb` walks through the data, the training of both detectors, inference and the comparison. `mobtools/` holds the helpers it calls: `dataset.py` reads the COCO export, plots the class balance, draws boxes and converts the annotations to the layout YOLO expects; `metrics.py` reads the final epoch of each trainer's log into one table; `report.py` writes the PDF report. `utils.py` re-exports them under the names the committed notebook imports.
- **Vendored mmdetection.** `configs/`, `tools/` and `demo/` are OpenMMLab's code, kept so FCOS can be trained without a separate checkout. `configs/fcos/fcos_minecraft.py` is the configuration the notebook trains FCOS with. `THIRD_PARTY.md` says which path belongs to whom.

`artifacts/` holds the results of the run behind `report.pdf`: training logs and curves, inference examples and `metrics/metrics_comparison.csv`. The dataset, the video and the trained weights are not in the repository.

## Use

```python
from mobtools import analyze_coco_distribution, compare_metrics, generate_report, prepare_yolo_dataset

prepare_yolo_dataset("datasets/minecraft", "datasets/minecraft_yolo")
analyze_coco_distribution("datasets/minecraft/train/annotations.json", "artifacts/distribution.png")
compare_metrics("artifacts/yolo/results.csv", "artifacts/fcos/20260311_015948/vis_data/scalars.json", "artifacts/metrics/metrics_comparison.csv")
generate_report("artifacts/distribution.png", "artifacts/metrics/metrics_comparison.csv")
```

`docs/install.md` says what to install, for the helpers and for the notebook, and `docs/manual.md` describes each helper. `.\recipe.ps1 test` runs the tests, mypy and ruff over `mobtools/`, `utils.py` and `tests/`. The tests write their own small COCO files, images and logs, so they need neither the dataset nor any weights, and they check that importing `mobtools` loads no part of the training stack.

## Licence

The code written for this project - `mobtools/`, `tests/`, `utils.py` and the notebook - is MIT, see `LICENSE`.

`tools/`, `demo/` and `configs/` are vendored from [mmdetection](https://github.com/open-mmlab/mmdetection) and remain under the Apache License 2.0; the copy required by that licence is in `LICENSES/Apache-2.0-OpenMMLab.txt`.

## Lab requirements

This repository complies with [revision 4.0.0](http://127.0.0.1:3000/igors-lab/software-requirements/src/tag/v4.0.0/BEST_REQUIREMENTS.md) of the lab requirements, except R-02 (#8), R-09 (#9), R-12 (#10), R-37 (#11), R-41 (#12), R-42 (#13).
