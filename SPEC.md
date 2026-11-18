# Спецификация требований: minecraft_mob_detection

Восстановлена по коду ветки `dev`, коммит `f5d087f`, 2026-11-18. Описывает реализованное
поведение; заявленное без кода вынесено в 3.6, неизвестное — в раздел 9.

<!-- jira-sync:start -->
Jira: KAN-80

## Tasks

This list is one ticket in Jira project KAN, [KAN-80](https://igorsvolohovs.atlassian.net/browse/KAN-80). Each line is a checkbox there: tick it in Jira and the sync ticks it here; add a checkbox there and it is appended here. Done items are the features described in section 3; open items are what the project declares but the code does not do yet.

- [x] Анализ датасета (`mobtools/dataset.py`) (section 3.1) <!-- spec-mobtools-dataset-py -->
- [x] Преобразование COCO → YOLO (`mobtools/dataset.py`) (section 3.2) <!-- spec-coco-yolo-mobtools-dataset-py -->
- [x] Сравнение метрик (`mobtools/metrics.py`) (section 3.3) <!-- spec-mobtools-metrics-py -->
- [x] PDF-отчёт (`mobtools/report.py`) (section 3.4) <!-- spec-pdf-mobtools-report-py -->
- [x] Сценарий, конфиг FCOS и сопровождение (section 3.5) <!-- spec-fcos-9fae9f96dd -->
- [ ] Сравнение по скорости инференса (FPS) (declared, not implemented) <!-- spec-fps-818477d718 -->
- [ ] Результаты инференса на видеопотоке в `artifacts/videos/` (declared, not implemented) <!-- spec-artifacts-videos-1e4797d9f3 -->
- [ ] `requirements.txt` (declared, not implemented) <!-- spec-requirements-txt -->
- [ ] Сохранённые веса в `artifacts/fcos/`, `artifacts/yolo/` (declared, not implemented) <!-- spec-artifacts-fcos-artifacts-yolo -->
- [ ] Конфигурация Ultralytics в `configs/` (declared, not implemented) <!-- spec-ultralytics-configs -->
- [ ] Последующие фазы работы (declared, not implemented) <!-- spec-4fd8de359d -->
- [ ] Страницы «Пример визуализации» в `report.pdf` (declared, not implemented) <!-- spec-report-pdf-135bb9371e -->

<!-- jira-sync:end -->

## 1. Назначение и границы

Учебный проект («coursework», `mobtools/__init__.py:1`) по сравнению двух детекторов, YOLOv8s
и FCOS, на датасете мобов Minecraft (17 классов, COCO).

В границах:
- пакет `mobtools` (`dataset.py`, `metrics.py`, `report.py`) и прослойка `utils.py`;
- `notebook.ipynb` (порядок вызовов);
- конфиг `configs/fcos/fcos_minecraft.py`;
- тесты `tests/`, CI `.github/workflows/python-ci.yml`, `scripts/check_before_push.py`;
- закоммиченные результаты в `artifacts/` как данные.

Вне границ:
- вендорный mmdetection: `tools/`, `demo/`, остальные `configs/` (`THIRD_PARTY.md`);
- сам датасет (`datasets/` игнорируется, `.gitignore:3`) и веса моделей (в git их нет;
  `.gitignore:1` закрывает только `artifacts/yolo/weights/`);
- само обучение и видеоинференс: ячейки 6-9 `notebook.ipynb` лишь вызывают Ultralytics,
  `tools/train.py` и `demo/video_demo.py` (см. FR-21);
- ветка `origin/release` и её 15 коммитов, которых нет в `dev`.

## 2. Контекст

| Участник / система | Роль | Источник |
| --- | --- | --- |
| Автор курсовой | запускает ноутбук, обучает модели, получает отчёт | `mobtools/__init__.py:1`, `report.py:4` |
| Ultralytics YOLO | обучение YOLOv8s по `data.yaml`, пишет `results.csv` | `artifacts/yolo/args.yaml:3-4`, `notebook.ipynb` ячейка 6 |
| MMDetection / MMEngine | обучение FCOS, JSON-лог валидации, `DetInferencer` | `configs/fcos/fcos_minecraft.py`, `notebook.ipynb` ячейки 5, 7, 9 |
| download.openmmlab.com | предобученные веса FCOS (COCO) | `configs/fcos/fcos_minecraft.py:77` |
| GitHub Actions | ruff и pytest на push / PR в `dev` | `.github/workflows/python-ci.yml:6-34` |

Среда: Python 3.10 (`pyproject.toml:8`, `python-ci.yml:19`). Ячейка инференса требует CUDA
(`device='cuda:0'`, `notebook.ipynb` ячейка 5). Закоммиченный прогон FCOS выполнен на Windows,
Python 3.10.0, PyTorch 2.0.0+cu118, MMEngine 0.10.7, одна GPU
(`artifacts/fcos/20260311_015948/20260311_015948.log:1-39`) — это запись, а не требование.

## 3. Функциональные требования

Нумерация ячеек ноутбука — с 1.

### 3.1. Анализ датасета (`mobtools/dataset.py`)

- **FR-1.** Система должна подсчитывать число аннотаций для каждой категории COCO, включая
  категории без аннотаций, и возвращать словарь, упорядоченный по убыванию числа.
  Source: `mobtools/dataset.py:61-74`. Verification: `test_distribution_counts_every_class`,
  `test_distribution_is_ordered_largest_first`.
- **FR-2.** Система должна сохранять столбчатую диаграмму этих чисел в заданный файл, создавая
  родительские каталоги, и закрывать фигуру matplotlib после сохранения.
  Source: `mobtools/dataset.py:76-87`. Verification: `test_distribution_counts_every_class`
  (только факт записи файла).
- **FR-3.** Система должна находить изображение в COCO по вхождению имени файла в `file_name`,
  использовать найденный id вместо переданного `img_id` и печатать сообщение INFO при их
  расхождении. Source: `mobtools/dataset.py:108-113`. Verification: no test (для расхождения).
- **FR-4.** Система должна возбуждать `ValueError`, если изображение не найдено в аннотациях и
  `img_id` не передан, и `FileNotFoundError`, если файл изображения не читается.
  Source: `mobtools/dataset.py:114-119`. Verification: `test_an_unknown_image_is_reported`
  (`ValueError`); для `FileNotFoundError` no test.
- **FR-5.** Система должна рисовать для каждой аннотации изображения зелёный прямоугольник
  толщиной 2 и подпись с именем класса (или `ID:<n>` для неизвестной категории), при
  `save_path` сохранять результат, при `show=True` выводить его через matplotlib и возвращать
  массив изображения. Source: `mobtools/dataset.py:121-138`.
  Verification: `test_visualising_returns_the_annotated_image`.

### 3.2. Преобразование COCO → YOLO (`mobtools/dataset.py`)

- **FR-6.** Система должна исключать категорию `minecraft-mobs` и назначать индексы классов
  0..N-1 остальным категориям в порядке возрастания их id.
  Source: `mobtools/dataset.py:25`, `:156-160`, `:170-171`.
  Verification: `test_the_wrapper_category_is_dropped`.
- **FR-7.** Система должна записывать каждую рамку строкой
  `<класс> <cx> <cy> <w> <h>`, где центр и размер — доли ширины и высоты изображения с шестью
  знаками после точки. Source: `mobtools/dataset.py:172-178`.
  Verification: `test_yolo_normalisation_is_centre_and_fraction`.
- **FR-8.** Система должна создавать по одному `.txt` на каждое изображение COCO с именем по
  основе имени файла, в том числе пустой файл для изображения без аннотаций, и возвращать число
  изображений. Source: `mobtools/dataset.py:166-182`.
  Verification: `test_an_image_with_no_annotations_gets_an_empty_file`.
- **FR-9.** Система должна обходить сплиты `train`, `valid`, `test`, пропускать отсутствующий
  сплит с предупреждением, создавать в целевом каталоге `<сплит>/images` и `<сплит>/labels`,
  конвертировать `<сплит>/annotations.json` (или предупреждать о его отсутствии) и копировать
  файлы `.jpg`, `.jpeg`, `.png` без учёта регистра расширения.
  Source: `mobtools/dataset.py:198-219`. Verification: `test_prepare_builds_the_layout_yolo_expects`,
  `test_a_missing_split_is_skipped_not_fatal`.
- **FR-10.** Система должна записывать `data.yaml` с абсолютным `path`, путями
  `train/images`, `valid/images`, `test/images`, `nc` и `names` (по умолчанию 17 имён
  `MINECRAFT_CLASSES`) даже при отсутствии всех сплитов и возвращать путь к нему.
  Source: `mobtools/dataset.py:196`, `:221-236`. Verification:
  `test_prepare_builds_the_layout_yolo_expects`, `test_a_missing_split_is_skipped_not_fatal`.

### 3.3. Сравнение метрик (`mobtools/metrics.py`)

- **FR-11.** Система должна возвращать значения `metrics/mAP50-95(B)` и `metrics/mAP50(B)`
  последней строки `results.csv` Ultralytics.
  Source: `mobtools/metrics.py:15-16`, `:21-29`. Verification: `test_the_last_epoch_is_the_one_reported`.
- **FR-12.** Система должна возбуждать `ValueError` для `results.csv` без строк и `KeyError` с
  перечнем отсутствующих колонок. Source: `mobtools/metrics.py:24-28`.
  Verification: `test_an_empty_results_file_says_so`; для `KeyError` no test.
- **FR-13.** Система должна читать из лога MMDetection только строки, содержащие
  `coco/bbox_mAP`, пропускать строки, не разбираемые как JSON, и возвращать `coco/bbox_mAP` и
  `coco/bbox_mAP_50` последней разобранной записи; при отсутствии таких записей — возбуждать
  `ValueError`. Source: `mobtools/metrics.py:38-49`. Verification:
  `test_training_lines_are_skipped`, `test_a_truncated_final_line_does_not_break_parsing`,
  `test_a_log_without_validation_says_so`.
- **FR-14.** Система должна сводить метрики в таблицу с колонками `Model` (`YOLOv8s`, `FCOS`),
  `mAP`, `mAP_50`, сохранять её в CSV без индекса, создавая каталог, и возвращать таблицу.
  Source: `mobtools/metrics.py:52-74`. Verification: `test_comparison_holds_both_models`.

### 3.4. PDF-отчёт (`mobtools/report.py`)

- **FR-15.** Система должна использовать шрифт DejaVu из `ttf/DejaVuSans.ttf` и
  `ttf/DejaVuSans-Bold.ttf`, если оба файла есть, иначе Helvetica с предупреждением.
  Source: `mobtools/report.py:15-17`, `:42-54`. Verification: no test (ветка отката).
- **FR-16.** Система должна печатать на каждой странице заголовок отчёта и дату генерации
  `ДД.ММ.ГГГГ`, а внизу — «Страница N». Source: `mobtools/report.py:56-85`. Verification: no test.
- **FR-17.** Система должна возбуждать `FileNotFoundError`, если нет графика распределения;
  иначе начинать отчёт страницей «Распределение классов в датасете» с изображением шириной 100
  по центру и подписью. Source: `mobtools/report.py:97-111`, `:124-133`.
  Verification: `test_a_missing_chart_is_reported`, `test_report_is_written`.
- **FR-18.** Система должна при наличии CSV сравнения добавлять страницу «Сравнение метрик
  YOLOv8s и FCOS» со строкой на модель (mAP и mAP_50 с тремя знаками), а при его отсутствии —
  печатать предупреждение и не добавлять страницу. Source: `mobtools/report.py:135-146`.
  Verification: `test_report_is_written`, `test_report_survives_a_missing_comparison`.
- **FR-19.** Система должна записывать PDF в `output_path` (по умолчанию `artifacts/report.pdf`),
  создавая каталог, и возвращать путь. Source: `mobtools/report.py:117`, `:148-152`.
  Verification: `test_report_is_written`.

### 3.5. Сценарий, конфиг FCOS и сопровождение

- **FR-20.** Модуль `utils` должен реэкспортировать восемь публичных имён `mobtools` для
  ноутбука. Source: `utils.py:8-28`. Verification: no test.
- **FR-21.** Ноутбук должен создавать `artifacts/inference`, `artifacts/videos`,
  `artifacts/metrics`, строить YOLO-датасет в `yolo_datasets/minecraft`, сохранять
  `artifacts/distribution.png` и `artifacts/inference/test_eda.jpg` и запускать FCOS
  `DetInferencer` на одном тестовом изображении с выводом в `artifacts/inference/` (ячейки 1-5);
  обучать YOLOv8s (`yolov8s.pt`, 12 эпох, imgsz 512, проект `artifacts`, имя `yolo`) и FCOS
  (`tools/train.py configs/fcos/fcos_minecraft.py --work-dir artifacts/fcos`) (ячейки 6-7);
  запускать YOLO с `artifacts/yolo/weights/best.pt` на `datasets/minecraft/test/` и
  `datasets/minecraft/video.mp4` и FCOS `demo/video_demo.py` с `artifacts/fcos/latest.pth` на том
  же видео (ячейки 8-9); вызывать `compare_metrics` и `generate_report` (ячейки 10-11).
  Source: `notebook.ipynb` ячейки 1-11. Verification: no test.
- **FR-22.** Конфиг должен задавать дообучение FCOS R50-caffe FPN GN-head на 17 классах:
  вход 512×512 с сохранением пропорций, горизонтальный flip 0.5, batch 2, 100 эпох с валидацией
  COCO bbox каждую эпоху на `valid/annotations.json`, SGD lr 0.00125, линейный разогрев 500
  итераций, снижение LR ×0.1 на эпохах 70 и 90, старт с весов COCO.
  Source: `configs/fcos/fcos_minecraft.py:1-77`. Verification: no test; закоммиченный лог
  содержит 100 записей валидации (`artifacts/fcos/20260311_015948/vis_data/scalars.json`).
- **FR-23.** `scripts/check_before_push.py` — стандартный скрипт из
  IGORSVOLOHOVS/repo-quality-template: закоммиченный файл совпадает с шаблонным побайтно
  (в рабочей копии отличаются только окончания строк), здесь не специфицируется. Отклонение
  проекта одно: в `.github/workflows/python-ci.yml` нет mypy, поэтому локально повторяются только
  `ruff check .`, `ruff format --check .` и `pytest`. Verification: no test.

### 3.6. Заявлено, но не реализовано

| Заявление | Источник | Что в коде |
| --- | --- | --- |
| Сравнение по скорости инференса (FPS) | `README.md:3`, `:12` | кода замера нет (есть лишь штатный вывод времени Ultralytics и `video_demo.py` в ячейках 6, 8, 9); в `metrics_comparison.csv` только mAP и mAP_50 |
| Результаты инференса на видеопотоке в `artifacts/videos/` | `README.md:3`, `:13`, `:56` | код есть (ячейки 8-9), но `artifacts/videos/` игнорируется (`.gitignore:2`), результатов в git нет |
| `requirements.txt` | `README.md:16` | файла нет |
| Сохранённые веса в `artifacts/fcos/`, `artifacts/yolo/` | `README.md:10` | в git их нет; `artifacts/yolo/weights/` игнорируется (`.gitignore:1`), для `.pth` в `artifacts/fcos/` правила нет |
| Конфигурация Ultralytics в `configs/` | `README.md:8` | в `configs/yolo/` конфиги YOLOv3 из mmdetection |
| Последующие фазы работы | `README.md:23` | описана только «Фаза 1» |
| Страницы «Пример визуализации» в `report.pdf` | `report.pdf`, с. 3-6 | текущий `generate_report` создаёт не более двух страниц |

## 4. Интерфейсы

Python API (`mobtools/__init__.py:18-27`):

| Функция | Вход | Выход |
| --- | --- | --- |
| `analyze_coco_distribution(json_path, save_path)` | COCO JSON, путь PNG | `dict[str, int]` |
| `visualize_bboxes(img_path, json_path, img_id=None, save_path=None, show=True)` | изображение, COCO JSON | `np.ndarray` (BGR) |
| `coco_to_yolo(json_path, output_dir)` | COCO JSON | число изображений |
| `prepare_yolo_dataset(src_root, dst_root, class_names=None)` | каталог сплитов | путь `data.yaml` |
| `compare_metrics(yolo_csv, fcos_log, save_path)` | `results.csv`, JSON Lines | `pd.DataFrame` |
| `generate_report(distribution_image, metrics_csv, output_path="artifacts/report.pdf")` | PNG, CSV | путь PDF |
| `PDFReport` | подкласс `fpdf.FPDF` | — |

CLI:

| Команда | Флаги | Источник |
| --- | --- | --- |
| `python scripts/check_before_push.py` | `--install-hook`, `--quiet` | `scripts/check_before_push.py:327-331` |

Форматы файлов:

| Файл | Формат | Источник |
| --- | --- | --- |
| `<сплит>/annotations.json` | COCO: `categories[id,name]`, `images[id,file_name,width,height]`, `annotations[image_id,category_id,bbox]` | `dataset.py:58`, `:121-124`, `:163-172` |
| `labels/<stem>.txt` | строки `класс cx cy w h`, доли с 6 знаками | `dataset.py:173-180` |
| `data.yaml` | `path`, `train`, `val`, `test`, `nc`, `names` | `dataset.py:225-233` |
| `results.csv` | колонки Ultralytics, читаются `metrics/mAP50-95(B)`, `metrics/mAP50(B)` | `metrics.py:15-16` |
| лог FCOS | JSON Lines, ключи `coco/bbox_mAP`, `coco/bbox_mAP_50` | `metrics.py:17-18` |
| `metrics_comparison.csv` | `Model,mAP,mAP_50` | `metrics.py:64-73` |
| `inference/preds/*.json` | `labels`, `scores`, `bboxes` (100 записей в закоммиченном файле) | вывод `DetInferencer` |

## 5. Данные

- Вход: `datasets/minecraft/{train,valid,test}/` с изображениями и `annotations.json`
  (`dataset.py:48`, `:210`; `fcos_minecraft.py:11`, `:31-32`, `:46-47`). В git не хранится.
- Промежуточные: `yolo_datasets/minecraft/` (`notebook.ipynb` ячейка 2), не в `.gitignore`.
- Результаты в git: `artifacts/yolo/results.csv` (12 эпох), `artifacts/fcos/20260311_015948/`
  (100 эпох валидации, `scalars.json` и `20260311_015948.json` совпадают побайтно),
  `artifacts/metrics/metrics_comparison.csv`, `artifacts/distribution.png`,
  `artifacts/inference/`, `artifacts/report.pdf`, `report.pdf`.
- Классы: 17 имён `MINECRAFT_CLASSES` (`dataset.py:27-45`), тот же кортеж в
  `fcos_minecraft.py:10`.

## 6. Нефункциональные требования

- **NFR-1.** Код проекта совместим с Python 3.10. Source: `pyproject.toml:8`, `python-ci.yml:19`,
  `README.md:26`.
- **NFR-2.** Код `mobtools/`, `tests/`, `utils.py` проходит `ruff check` с набором правил
  `E W F I N UP B C4 SIM RUF ANN` и `ruff format --check`, длина строки 100; вендорный код и
  `scripts/` не проверяются. Source: `pyproject.toml:6-19`, `python-ci.yml:27-31`.
- **NFR-3.** Тесты импортируют пакет одинаково при `pytest` и `python -m pytest`.
  Source: `pyproject.toml:27`.
- **NFR-4.** Сообщения об ошибках называют файл и причину (пустой CSV, нет колонок, нет записей
  валидации, нет графика, нет изображения). Source: `metrics.py:25`, `:28`, `:47`;
  `report.py:126`; `dataset.py:115`, `:119`.
- **NFR-5.** Текстовые файлы читаются и пишутся в UTF-8. Source: `dataset.py:54`, `:180`, `:233`;
  `metrics.py:39`.
- **NFR-6.** Собственный код распространяется под MIT, вендорный mmdetection — под Apache-2.0 с
  копией лицензии. Source: `README.md:58-67`, `THIRD_PARTY.md`, `LICENSES/Apache-2.0-OpenMMLab.txt`.

## 7. Ограничения и зависимости

Версии нигде не закреплены: нет lock-файла и `requirements.txt`.

| Зависимость | Ограничение | Источник |
| --- | --- | --- |
| pandas, matplotlib, opencv-python(-headless), numpy, fpdf2, pillow | без версий | `python-ci.yml:25` |
| ruff, pytest | без версий | `python-ci.yml:25` |
| torch, torchvision, torchaudio | индекс cu118 | `README.md:42` |
| openmim, mmengine, mmdet, ultralytics, jupyterlab, pycocotools | без версий | `README.md:49-52` |
| mmcv | `>=2.0.0` | `README.md:51` |
| Веса FCOS COCO | URL | `fcos_minecraft.py:77` |
| `yolov8s.pt` | — | `artifacts/yolo/args.yaml:3` |
| Шрифты DejaVu | файлы в `ttf/`, лицензия в репозитории не указана | `report.py:42-43` |

`fpdf2` обязателен для импорта `mobtools` и `utils` (`mobtools/__init__.py:16`,
`report.py:13`), но отсутствует в списке установки `README.md:52`.

## 8. Приёмка

1. Установить зависимости по `python-ci.yml:24-25`, выполнить `ruff check .`,
   `ruff format --check .`, `pytest` (`python-ci.yml:28-34`) — все три должны завершиться с
   кодом 0. Это покрывает FR-1…FR-19 там, где указан тест.
2. Либо `python scripts/check_before_push.py`: итог «all clear» означает, что команды CI прошли
   локально (`scripts/check_before_push.py:314-316`).
3. FR-20…FR-22 проверяются вручную: импорт `utils`, выполнение ячеек `notebook.ipynb` с
   датасетом в `datasets/minecraft/` и GPU, загрузка `configs/fcos/fcos_minecraft.py` в MMEngine.
   Сейчас ожидается отказ последней проверки: отсутствует
   `configs/_base_/datasets/coco_detection.py` (его скрывает `.gitignore:3`).
4. Сверка данных: `compare_metrics` на `artifacts/yolo/results.csv` и
   `artifacts/fcos/20260311_015948/vis_data/scalars.json` по коду должен дать для FCOS
   0.297 / 0.476, тогда как в `artifacts/metrics/metrics_comparison.csv` 0.041 / 0.089. Пока
   расхождение не объяснено, закоммиченный CSV этой спецификации не соответствует.

## 9. Открытые вопросы

1. Для какого курса и какого заказчика проект, есть ли целевые значения mAP или FPS — в файлах
   не сказано.
2. Источник и лицензия датасета. Размеры сплитов известны только из сохранённого вывода ячейки 2
   (код `0536db1`): train 2307, valid 422, test 155 изображений. Имена файлов вида
   `*_jpg.rf.<hash>.jpg` похожи на экспорт Roboflow — это вывод, не установленный факт.
3. Частично отвечено ноутбуком. YOLOv8s: команда в ячейке 6, её сохранённый вывод —
   Ultralytics 8.4.21, `name=yolo6`, `optimizer=auto`, как в `args.yaml:16`, `:19`; но в исходнике
   ячейки теперь `optimizer='AdamW'` — по-видимому, код правили после прогона. FCOS: команда в
   ячейке 7 (`tools/train.py configs/fcos/fcos_minecraft.py --work-dir artifacts/fcos`), но её сохранённый
   вывод — другой прогон (`fcos_minecraft_20260311_014247`), упавший на 11-й эпохе с
   `PermissionError` при записи чекпойнта. Какой командой получен закоммиченный прогон
   `20260311_015948`, не записано.
4. Откуда в `metrics_comparison.csv` и `artifacts/report.pdf` значения FCOS 0.041 / 0.089, которых
   нет в закоммиченном логе.
5. Каким кодом созданы страницы 3-6 корневого `report.pdf`: ни текущий код, ни `0536db1:utils.py`
   их не создают; файл добавлен коммитом `d0d787c` («Add files via upload»).
6. Отвечено сохранённым выводом ячейки 5: `size mismatch for bbox_head.conv_cls.weight` — в
   чекпойнте 80 классов, в модели 17. Это COCO-веса из `fcos_minecraft.py:77`, голова
   классификации не загружена; закоммиченное предсказание в `artifacts/inference/` сделано не
   дообученной моделью. Открыто: почему инференс не шёл с весами из `artifacts/fcos/`.
7. Совпадает ли в реальном экспорте порядок `category_id` с алфавитным порядком
   `MINECRAFT_CLASSES` (FR-6 против FR-10). Косвенный признак расхождения: в train у `wolf`
   4 аннотации (вывод ячейки 3), а валидация YOLO насчитывает под этим именем 138 объектов
   (вывод ячейки 6). Не доказательство.
8. Что делает fpdf2 с кириллицей при откате на Helvetica: docstring `report.py:33-37` говорит о
   пустых заголовках; не проверено, fpdf2 здесь не установлен.
9. Почему YOLOv8s обучался 12 эпох, а FCOS 100.
10. FPS: запланирован или исключён из работы (видеоинференс в ноутбуке есть, ячейки 8-9).
11. `THIRD_PARTY.md` относит весь `configs/` к OpenMMLab, но `configs/fcos/fcos_minecraft.py`
    написан под этот датасет (17 классов, `datasets/minecraft/`). Под какой лицензией он?
12. Отвечено: прогон «Python CI» на `f5d087f` (push в `dev`) на GitHub успешен, шаги Lint,
    Format check, Test пройдены (`gh run view`, проверено 2026-11-18).
13. Какая ветка основная: `origin/HEAD` указывает на `release`, CI срабатывает только на `dev`,
    и `release` на 15 коммитов впереди `dev`.
