# Install

## The helpers and their tests

1. Install [uv](https://docs.astral.sh/uv/) and Python 3.13.
2. Clone this repository and run `uv sync --frozen` at its root. That installs `numpy`, `opencv-python-headless`, `matplotlib`, `pandas` and `fpdf2` at the versions `uv.lock` pins, and the test tools.
3. `.\recipe.ps1 test` runs pytest, mypy and ruff. Nothing is downloaded beyond the packages: the tests make their own data.

`.\recipe.ps1 build` packs `mobtools` and `utils.py` into a wheel in `dist\`. `.\recipe.ps1 run -c "import mobtools"` starts Python with the helpers importable.

The PDF report needs `ttf/DejaVuSans.ttf` and `ttf/DejaVuSans-Bold.ttf`, which are in the repository, and must be written from the repository root. Elsewhere it falls back to Helvetica and warns that Cyrillic will not render.

## The notebook

Training is not part of the lab's pipeline: it needs a GPU, downloads the pretrained FCOS and YOLOv8 weights, and the OpenMMLab packages want Python 3.10.

1. Create a Python 3.10 environment: `python -m venv .venv`, then `.\.venv\Scripts\Activate.ps1`.
2. Install PyTorch for your CUDA version first, for example `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`.
3. Then the rest: `pip install -U openmim`, `mim install mmengine`, `mim install "mmcv>=2.0.0"`, `pip install mmdet ultralytics jupyterlab pycocotools opencv-python pandas matplotlib fpdf2`.
4. Put the dataset in `datasets/minecraft/` with the folders `train`, `valid` and `test`, each holding its images and `annotations.json`, and the video for inference at `datasets/minecraft/video.mp4`. `datasets/` is gitignored.
