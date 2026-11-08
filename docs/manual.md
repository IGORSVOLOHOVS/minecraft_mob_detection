# Manual

Every helper is in `mobtools` and is also importable from `utils`, which is what `notebook.ipynb` does. Paths may be strings or `Path`s.

## Dataset

- `analyze_coco_distribution(json_path, save_path)` counts the annotations per class of a COCO file, saves a bar chart largest class first, and returns the counts as a dict. A class with no annotation appears with 0.
- `visualize_bboxes(img_path, json_path, img_id=None, save_path=None, show=True)` draws the boxes and labels of one image and returns the image as an array. The image is found in the annotations by its file name; an `img_id` that disagrees is corrected with an `INFO` line. An image that is not in the annotations raises `ValueError`, a file that cannot be read raises `FileNotFoundError`. `save_path` writes the drawing, `show=False` keeps it off the screen.
- `coco_to_yolo(json_path, output_dir)` writes one YOLO label file per image, class index then centre and size as fractions of the image, and returns how many images it wrote. The wrapper category `minecraft-mobs` is dropped, and an image with no box gets an empty file.
- `prepare_yolo_dataset(src_root, dst_root, class_names=None)` builds a YOLO dataset from the `train`, `valid` and `test` splits under `src_root`: images copied, labels converted, and a `data.yaml` with the 17 class names of `MINECRAFT_CLASSES` unless others are given. It returns the path of `data.yaml`. A missing split or a missing `annotations.json` is reported and skipped.

## Metrics

- `compare_metrics(yolo_csv, fcos_log, save_path)` takes the last epoch's mAP and mAP@50 from Ultralytics' `results.csv` and the last validation entry of MMDetection's JSON log, saves them as one CSV table and returns it as a DataFrame. An empty results file, renamed columns or a log without validation entries raise an error that names the file.

## Report

- `generate_report(distribution_image, metrics_csv, output_path="artifacts/report.pdf")` writes an A4 PDF in Russian, the language of the coursework: the class distribution chart, then the comparison of the two models. Without the metrics CSV the comparison section is left out; without the chart it raises `FileNotFoundError`. It returns the path it wrote.
