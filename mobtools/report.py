"""The PDF summary: class distribution, then the model comparison.

Text in the report itself stays in Russian, because that is the language of the
coursework it belongs to. The code around it is documented in English.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd
from fpdf import FPDF, XPos, YPos

FONT_DIR = Path("ttf")
BUNDLED_FONT = "DejaVu"
FALLBACK_FONT = "Helvetica"

TITLE = "Аналитический отчёт: детекция мобов Minecraft"
DISTRIBUTION_CAPTION = (
    "На графике показано распределение объектов различных классов в "
    "тренировочном COCO датасете. Анализ помогает оценить сбалансированность "
    "данных."
)
COMPARISON_CAPTION = (
    "Различия метрик выявления объектов (наиболее высокая mAP_50 у лучшей модели):\n\n"
)


class PDFReport(FPDF):
    """An A4 report with a running header and page numbers.

    Cyrillic needs a font that has it: the core PDF fonts do not, so the report
    uses the DejaVu files under `ttf/` when they are present and falls back to
    Helvetica otherwise - which renders the headings as blanks, hence the
    warning.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)

        regular = FONT_DIR / "DejaVuSans.ttf"
        bold = FONT_DIR / "DejaVuSans-Bold.ttf"
        # Deliberately not `self.font_family`: FPDF sets that in its own
        # __init__ to track the currently selected font, and overwriting it
        # here left the library's idea of the current font out of step with
        # the document's.
        if regular.is_file() and bold.is_file():
            self.add_font(BUNDLED_FONT, "", str(regular))
            self.add_font(BUNDLED_FONT, "B", str(bold))
            self.body_font = BUNDLED_FONT
        else:
            print(f"Warning: {FONT_DIR}/DejaVuSans*.ttf missing; Cyrillic will not render")
            self.body_font = FALLBACK_FONT

    def header(self) -> None:
        """The title and generation date, repeated on every page."""
        self.set_font(self.body_font, "B", 15)
        self.cell(0, 10, TITLE, border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.set_font(self.body_font, "", 8)
        generated = datetime.date.today().strftime("%d.%m.%Y")
        self.cell(
            0,
            5,
            f"Дата генерации: {generated}",
            border=0,
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
            align="C",
        )
        self.ln(10)

    def footer(self) -> None:
        """The page number, 15 mm from the bottom."""
        self.set_y(-15)
        self.set_font(self.body_font, "B", 8)
        self.cell(
            0,
            10,
            f"Страница {self.page_no()}",
            border=0,
            new_x=XPos.RIGHT,
            new_y=YPos.TOP,
            align="C",
        )

    def chapter_title(self, title: str) -> None:
        self.set_font(self.body_font, "B", 12)
        self.cell(0, 10, title, border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
        self.ln(5)

    def chapter_body(self, body: str) -> None:
        self.set_font(self.body_font, "", 10)
        self.multi_cell(0, 5, body)
        self.ln()

    def add_image_section(self, title: str, image_path: str | Path, caption: str) -> None:
        """A page holding one centred image with text beneath it."""
        self.add_page()
        self.chapter_title(title)

        image_width = 100
        usable_width = self.w - 2 * self.l_margin
        self.image(
            str(image_path),
            x=(usable_width - image_width) / 2 + self.l_margin,
            y=None,
            w=image_width,
        )
        self.ln(5)
        self.chapter_body(caption)


def generate_report(
    distribution_image: str | Path,
    metrics_csv: str | Path,
    output_path: str | Path = "artifacts/report.pdf",
) -> Path:
    """Write the PDF and return where it landed.

    The metrics section is skipped when the comparison has not been produced
    yet, so the distribution chart alone still yields a readable report.
    """
    distribution_image = Path(distribution_image)
    if not distribution_image.is_file():
        raise FileNotFoundError(f"No distribution chart at {distribution_image}")

    pdf = PDFReport()
    pdf.add_image_section(
        title="Распределение классов в датасете",
        image_path=distribution_image,
        caption=DISTRIBUTION_CAPTION,
    )

    metrics_csv = Path(metrics_csv)
    if metrics_csv.is_file():
        frame = pd.read_csv(metrics_csv)
        pdf.add_page()
        pdf.chapter_title("Сравнение метрик YOLOv8s и FCOS")
        rows = "".join(
            f"- Модель: {row['Model']}, mAP: {row['mAP']:.3f}, mAP_50: {row['mAP_50']:.3f}\n"
            for _, row in frame.iterrows()
        )
        pdf.chapter_body(COMPARISON_CAPTION + rows)
    else:
        print(f"Warning: {metrics_csv} not found; the report has no comparison section")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))
    print(f"Отчет успешно сгенерирован в: {output_path}")
    return output_path
