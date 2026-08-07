# Сканирование кубического мира: детекция персонажей в Minecraft с FCOS и YOLO

Данный проект посвящен исследованию возможностей современных моделей детекции объектов (FCOS и YOLOv8) на примере игрового мира Minecraft. В ходе работы произведено дообучение детекторов на специфическом игровом датасете (формат COCO, 17 классов мобов), а также сравнение их по точности (mAP), скорости инференса (FPS) и визуальному качеству предсказаний на тестовых изображениях и видеопотоке.

## Структура проекта

* `datasets/minecraft/` — исходный датасет COCO и видео для инференса (не отслеживается в git).
* `configs/` — файлы конфигурации для MMDetection (FCOS) и Ultralytics (YOLO).
* `artifacts/` — директория с результатами:
  * `fcos/` и `yolo/` — логи обучения и сохраненные веса (чекпоинты).
  * `inference/` — примеры работы моделей на тестовых изображениях.
  * `metrics/` — CSV-отчеты со сравнением метрик (mAP, FPS).
  * `videos/` — результаты инференса на видеопотоке.
* `notebook.ipynb` — основной Jupyter Notebook с EDA, кодом запуска обучения, инференса и визуализации.
* `utils.py` — вспомогательные функции для подготовки и обработки данных.
* `requirements.txt` — список необходимых Python-библиотек.
* `report.pdf` — итоговый аналитический отчет с выводами по спринту.

---

## Рабочий процесс и Запуск

Процесс выполнения проекта разделен на логические фазы. Прежде чем начать, убедитесь, что ваш базовый фундамент (сон, питание) в порядке, так как обучение моделей потребует концентрации и времени.

### Фаза 1: Исследование (Где я? Что вокруг меня?)
На этом этапе необходимо подготовить рабочую среду (строго под Python 3.10) и осмотреть сырые данные. **Важно соблюдать порядок установки**, так как зависимости OpenMMLab требуют уже установленного PyTorch.

1. Создайте и активируйте виртуальное окружение:

**Для Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

```

*(Для Linux/macOS: `source .venv/bin/activate`)*

2. установите PyTorch. Команда зависит от версии CUDA на вашей ВМ (пример для CUDA 11.8):

```powershell
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

```

3. Убедившись, что PyTorch установлен без ошибок, установите базовые зависимости и пакеты MMDetection с YOLO:

```powershell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install mmdet ultralytics jupyterlab pycocotools opencv-python pandas matplotlib

```

4. Разместите скачанный датасет в `datasets/minecraft/` (внутри должны быть папки `train`, `val`, `test` и файл `annotations.json`). Разместите видео для инференса по пути `datasets/minecraft/video.mp4`.

## Licence

The code written for this project - `mobtools/`, `tests/`, `scripts/`,
`utils.py` and the notebook - is MIT, see `LICENSE`.

`tools/`, `demo/` and `configs/` are vendored from
[mmdetection](https://github.com/open-mmlab/mmdetection) and remain under the
Apache License 2.0; the copy required by that licence is in
`LICENSES/Apache-2.0-OpenMMLab.txt`. `THIRD_PARTY.md` says which path belongs to
whom.
