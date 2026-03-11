import json
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os
import shutil
import datetime
from fpdf import FPDF, XPos, YPos
from ultralytics import YOLO

"""
@brief Анализирует распределение классов в COCO датасете и сохраняет график.
@param[in] json_path Путь к файлу аннотаций COCO (annotations.json).
@param[in] save_path Путь для сохранения графика распределения.
@return dict Словарь с количеством объектов каждого класса.
"""
def analyze_coco_distribution(json_path, save_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    classes = {cat['id']: cat['name'] for cat in data['categories']}
    counts = {name: 0 for name in classes.values()}
    for ann in data['annotations']:
        counts[classes[ann['category_id']]] += 1
    sorted_counts_items = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    sorted_counts = dict(sorted_counts_items)
    plt.figure(figsize=(12, 6))
    plt.bar(list(sorted_counts.keys()), list(sorted_counts.values()))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    return sorted_counts

"""
@brief Визуализирует bounding box'ы на тестовом изображении из COCO.
@param[in] img_path Путь к изображению.
@param[in] json_path Путь к файлу аннотаций.
@param[in] img_id ID изображения в COCO формате.
@param[in] save_path Путь для сохранения результата.
@return None
"""
def visualize_bboxes(img_path, json_path, img_id=None, save_path=None):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    classes = {cat['id']: cat['name'] for cat in data['categories']}
    
    filename = os.path.basename(img_path)
    img_info = next((img for img in data['images'] if filename in img['file_name']), None)
    
    if img_info:
        actual_id = img_info['id']
        if img_id is not None and img_id != actual_id:
            print(f"INFO: Overriding img_id {img_id} with actual ID {actual_id} for '{filename}'")
        img_id = actual_id
    elif img_id is None:
        raise ValueError(f"Could not find image '{filename}' in annotations.")

    anns = [a for a in data['annotations'] if a['image_id'] == img_id]
    
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
    
    color = (0, 255, 0) 
    
    for ann in anns:
        x, y, w, h = map(int, ann['bbox'])
        category_id = ann['category_id']
        cls_name = classes.get(category_id, f"ID:{category_id}")
        
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, cls_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img)
    
    # Notebook display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()


"""
@brief Конвертирует аннотации из формата COCO в формат YOLO.
@param[in] json_path Путь к файлу аннотаций COCO (annotations.json).
@param[in] output_dir Директория для сохранения .txt файлов YOLO.
@return None
"""
def coco_to_yolo(json_path, output_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    categories = [cat for cat in data['categories'] if cat['name'] != 'minecraft-mobs']
    cat_map = {cat['id']: i for i, cat in enumerate(sorted(categories, key=lambda x: x['id']))}
    
    images = {img['id']: img for img in data['images']}
    
    img_anns = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_anns:
            img_anns[img_id] = []
        img_anns[img_id].append(ann)
    
    for img_id, img_info in images.items():
        img_w, img_h = img_info['width'], img_info['height']
        file_name = img_info['file_name']
        base_name = os.path.splitext(file_name)[0]
        
        yolo_lines = []
        for ann in img_anns.get(img_id, []):
            cat_id = ann['category_id']
            if cat_id not in cat_map:
                continue
            
            yolo_cls = cat_map[cat_id]
            x, y, w, h = ann['bbox']
            
            x_center = (x + w / 2.0) / img_w
            y_center = (y + h / 2.0) / img_h
            w_norm = w / float(img_w)
            h_norm = h / float(img_h)
            
            yolo_lines.append(f"{yolo_cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        with open(txt_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
    
    print(f"Successfully converted {len(images)} images to YOLO format in '{output_dir}'")

"""
@brief Подготавливает полный YOLO датасет (train/valid/test) из COCO структуры.
@param[in] src_root Корневая папка исходного датасета (с папками train, valid, test).
@param[in] dst_root Целевая папка для YOLO датасета.
@param[in] class_names Список имен классов. Если None, берется список по умолчанию для Minecraft.
"""
def prepare_yolo_dataset(src_root, dst_root, class_names=None):
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        print(f"Processing {split} split...")
        split_src = os.path.join(src_root, split)
        split_dst_images = os.path.join(dst_root, split, 'images')
        split_dst_labels = os.path.join(dst_root, split, 'labels')
        
        os.makedirs(split_dst_images, exist_ok=True)
        os.makedirs(split_dst_labels, exist_ok=True)
        
        # 1. Convert annotations
        json_path = os.path.join(split_src, 'annotations.json')
        if os.path.exists(json_path):
            coco_to_yolo(json_path, split_dst_labels)
        else:
            print(f"Warning: {json_path} not found")
            
        for filename in os.listdir(split_src):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                shutil.copy2(
                    os.path.join(split_src, filename),
                    os.path.join(split_dst_images, filename)
                )
    
    if class_names is None:
        class_names = ['bee', 'chicken', 'cow', 'creeper', 'enderman', 'fox', 'frog', 'ghast', 'goat', 'llama', 'pig', 'sheep', 'skeleton', 'spider', 'turtle', 'wolf', 'zombie']
        
    yaml_content = f"""path: {os.path.abspath(dst_root)}
train: train/images
val: valid/images
test: test/images

nc: {len(class_names)}
names: {class_names}
"""
    with open(os.path.join(dst_root, 'data.yaml'), 'w') as f:
        f.write(yaml_content)
    print(f"Created data.yaml in {dst_root}")

"""
@brief Собирает метрики из CSV файла YOLO и логов MMDetection для сравнения.
@param[in] yolo_csv Путь к файлу results.csv от YOLO.
@param[in] fcos_log Путь к лог-файлу JSON от FCOS.
@param[in] save_path Путь для сохранения итоговой таблицы.
@return DataFrame Сводная таблица метрик.
"""
def compare_metrics(yolo_csv, fcos_log, save_path):
    yolo_df = pd.read_csv(yolo_csv)
    yolo_map = yolo_df['metrics/mAP50-95(B)'].iloc[-1]
    yolo_map50 = yolo_df['metrics/mAP50(B)'].iloc[-1]
    with open(fcos_log, 'r') as f:
        fcos_data = [json.loads(line) for line in f if 'bbox_mAP' in line]
    fcos_map = fcos_data[-1]['coco/bbox_mAP']
    fcos_map50 = fcos_data[-1]['coco/bbox_mAP_50']
    comp = pd.DataFrame({
        'Model': ['YOLOv8s', 'FCOS'],
        'mAP': [yolo_map, fcos_map],
        'mAP_50': [yolo_map50, fcos_map50]
    })
    comp.to_csv(save_path, index=False)
    return comp

class PDFReport(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Пробуем использовать красивые шрифты, если они доступны (их мы скачивали в нашу папку ttf/)
        regular_font_path = os.path.join('ttf', 'DejaVuSans.ttf')
        bold_font_path = os.path.join('ttf', 'DejaVuSans-Bold.ttf')

        if os.path.exists(regular_font_path) and os.path.exists(bold_font_path):
            self.add_font('DejaVu', '', regular_font_path)
            self.add_font('DejaVu', 'B', bold_font_path)

            self.font_family = 'DejaVu'
        else:
            # Если шрифты не найдены, используем стандартный
            self.font_family = 'Arial'

    def header(self):
        """Создаёт шапку для каждой страницы"""
        self.set_font(self.font_family, 'B', 15)
        self.cell(0, 10, 'Аналитический отчёт по дорожной обстановке', border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.set_font(self.font_family, '', 8)
        self.cell(0, 5, f'Дата генерации: {datetime.date.today().strftime("%d.%m.%Y")}', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        """Добавляет номера страниц в подвале"""
        self.set_y(-15)
        self.set_font(self.font_family, 'B', 8)
        self.cell(0, 10, f'Страница {self.page_no()}', border=0, new_x=XPos.RIGHT, new_y=YPos.TOP, align='C')

    def chapter_title(self, title):
        """Создаёт заголовок раздела"""
        self.set_font(self.font_family, 'B', 12)
        self.cell(0, 10, title, border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        self.ln(5)

    def chapter_body(self, body):
        """Добавляет основной текст"""
        self.set_font(self.font_family, '', 10)
        self.multi_cell(0, 5, body)
        self.ln()
    
    def add_image_section(self, title, image_path, stats_text):
        """Добавляет секцию с изображением и статистикой"""
        self.add_page()
        self.chapter_title(title)

        # Центрируем изображение на странице
        image_width = 100
        page_width = self.w - 2 * self.l_margin
        x_position = (page_width - image_width) / 2 + self.l_margin
        self.image(image_path, x=x_position, y=None, w=image_width)
        self.ln(5)
        self.set_font(self.font_family, '', 10) 
        self.multi_cell(0, 5, stats_text)


def generate_report(distribution_image, metrics_csv, output_path="artifacts/report.pdf"):
    pdf = PDFReport()
    
    # Секция с распределением классов
    pdf.add_image_section(
        title="Распределение классов в датасете",
        image_path=distribution_image,
        stats_text="На графике показано распределение объектов различных классов в тренировочном COCO датасете. Анализ помогает оценить сбалансированность данных."
    )
    
    # Секция с метриками
    if os.path.exists(metrics_csv):
        df = pd.read_csv(metrics_csv)
        pdf.add_page()
        pdf.chapter_title("Сравнение метрик YOLOv8s и FCOS")
        
        body = "Различия метрик выявления объектов (наиболее высокая mAP_50 у лучшей модели):\n\n"
        for _, row in df.iterrows():
            body += f"- Модель: {row['Model']}, mAP: {row['mAP']:.3f}, mAP_50: {row['mAP_50']:.3f}\n"
        pdf.chapter_body(body)
        
    pdf.output(output_path)
    print(f"Отчет успешно сгенерирован в: {output_path}")