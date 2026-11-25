"""Render the same review as Markdown and a paginated PDF."""

from collections import Counter
from dataclasses import dataclass
from html import escape
from io import BytesIO
import os
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont, TTFError
from reportlab.platypus import LongTable, Paragraph, SimpleDocTemplate, TableStyle

from review import CATEGORIES, ITEM_CRITERIA, KINDS, SET_CRITERIA, SEVERITIES, STATUSES, Assessment, Review, ordered_findings
from source import Document


TITLE = "Статическое ревью требований"
SEVERITY_ORDER = {key: index for index, key in enumerate(SEVERITIES)}


@dataclass(frozen=True)
class Passport:
    time: str
    model: str
    model_versions: str
    revision: str
    requests: int


@dataclass(frozen=True)
class Text:
    style: str
    value: str


@dataclass(frozen=True)
class Table:
    header: tuple[str, ...]
    rows: list[tuple[str, ...]]


def review_blocks(document: Document, review: Review, passport: Passport) -> list[Text | Table]:
    blocks: list[Text | Table] = []
    issues = ordered_findings(review)
    duplicates = dict(review.duplicates)
    defects = []
    labels = {}
    for index, (owner, issue) in enumerate(issues, 1):
        if index not in duplicates:
            label = f"D-{len(defects) + 1:03d}"
            labels[index] = label
            defects.append((label, owner, issue))
    issue_number = 0

    def add(value: str, style: str = "body") -> None:
        blocks.append(Text(style, value))

    def emit_findings(assessment: Assessment, criteria: dict) -> None:
        nonlocal issue_number
        for finding in assessment.findings:
            issue_number += 1
            if issue_number in duplicates:
                add(f"Связанный дефект: {labels[duplicates[issue_number]]} · {finding.title}.")
                continue
            label = labels[issue_number]
            add(f"{label} · {finding.title}", "finding")
            add(
                f"{SEVERITIES[finding.severity]} · {CATEGORIES[finding.category]} · "
                f"{criteria[finding.criterion][0]}"
            )
            add(finding.explanation)
            for evidence in finding.evidence:
                add(f"{evidence.location} · «{evidence.quote}»", "quote")
            add(f"Исправление: {finding.fix}")

    add(TITLE, "title")
    add("1. Паспорт прогона", "section")
    add(f"Документ: {document.title}")
    add(f"Файл: {document.name}")
    add(f"SHA-256 исходного файла: {document.digest}")
    add(f"Дата и время начала: {passport.time}")
    add(f"Модель: Google Gemini API / {passport.model}")
    if passport.model_versions:
        add(f"Версия модели по ответу API: {passport.model_versions}")
    add(f"Версия агента: {passport.revision}")
    add(f"Запросов к модели, включая повторные: {passport.requests}")

    add("2. Выделенные требования", "section")
    add("REQ — идентификатор прогона. «—» означает, что в исходнике нет отдельного ID для положения.")
    if review.requirements:
        blocks.append(Table(
            ("ID отчёта", "ID документа", "Тип и формулировка", "Источник"),
            [
                (
                    item.id, item.document_id or "—", f"{KINDS[item.kind]} · {item.summary}",
                    f"{item.source.location} · «{item.source.quote}»",
                )
                for item in review.requirements
            ],
        ))
    else:
        add("Агент не выделил в документе требований или критериев приёмки.")

    add("3. Проверка каждого требования", "section")
    add("«Без замечаний» означает, что агент не обнаружил дефекта по этому критерию в исходном тексте.")
    for item in review.requirements:
        add(f"{item.id} · {item.summary}", "subheading")
        add(f"{item.source.location} · «{item.source.quote}»", "quote")
        assessment = review.individual[item.id]
        for status, label in STATUSES.items():
            names = [ITEM_CRITERIA[key][0] for key, value in assessment.checks.items() if value == status]
            if names:
                add(f"{label}: {', '.join(names)}.")
        emit_findings(assessment, ITEM_CRITERIA)
    if not review.requirements:
        add("Индивидуальные проверки отсутствуют, поскольку каталог пуст.")

    add("4. Проверка набора целиком", "section")
    for key, value in review.whole.checks.items():
        add(f"{SET_CRITERIA[key][0]} — {STATUSES[value]}.")
    emit_findings(review.whole, SET_CRITERIA)

    add("5. Сводка", "section")
    counts = Counter(finding.severity for _, _, finding in defects)
    add(f"Выделено положений: {len(review.requirements)}.")
    add(
        f"Всего нарушений: {len(defects)}. Критичных: {counts['critical']}. "
        f"Существенных: {counts['major']}. Малых: {counts['minor']}."
    )
    add("Критичный дефект блокирует ключевой сценарий. Существенный меняет поведение или покрытие. Малый имеет локальное влияние.")
    add("Критичные дефекты", "subheading")
    critical = [(label, owner, issue) for label, owner, issue in defects if issue.severity == "critical"]
    if not critical:
        add("Критичных дефектов в этом прогоне не обнаружено.")
    for label, owner, issue in critical:
        places = " / ".join(source.location for source in issue.evidence)
        add(f"{label} · {owner} · {CATEGORIES[issue.category]} · {issue.title} ({places}).", "bullet")

    add("6. Готовность к разработке и тестированию", "section")
    if not review.requirements:
        add("Разработка и тестирование: готовность не установлена. Сначала зафиксируйте проверяемые требования.")
    elif critical:
        add("Разработка: отложить реализацию затронутых сценариев до устранения критичных дефектов.")
        add("Тестирование: можно подготовить черновики сценариев. Утвердить ожидаемые результаты и начать приёмку можно после исправлений.")
    elif counts["major"]:
        add("Разработка: можно начать работу над согласованными сценариями. Уточните правила, затронутые существенными дефектами.")
        add("Тестирование: можно проектировать проверки согласованных сценариев. В затронутых сценариях сначала утвердите ожидаемые результаты.")
    else:
        add("Разработка и проектирование тестов: можно начинать по результатам статического ревью.")
    priority = sorted(defects, key=lambda record: SEVERITY_ORDER[record[2].severity])[:5]
    if priority:
        add("Что исправить в первую очередь", "subheading")
        for label, _, issue in priority:
            add(f"{label} · {issue.fix}", "bullet")
    return blocks


def markdown_text(value: str) -> str:
    return re.sub(r"([\\`*_\[\]|~$])", r"\\\1", escape(value, quote=False))


def to_markdown(blocks: list[Text | Table]) -> str:
    prefixes = {"title": "# ", "section": "## ", "subheading": "### ", "finding": "#### ", "quote": "> ", "bullet": "- "}
    sections = []
    for block in blocks:
        if isinstance(block, Text):
            sections.append(prefixes.get(block.style, "") + markdown_text(block.value))
            continue

        def row(cells: tuple[str, ...]) -> str:
            return "| " + " | ".join(markdown_text(cell) for cell in cells) + " |"

        separators = tuple("---" for _ in block.header)
        sections.append("\n".join([row(block.header), row(separators), *(row(cells) for cells in block.rows)]))
    return "\n\n".join(sections) + "\n"


def pdf_fonts(font: Path | None) -> tuple[str, str]:
    windows_fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    candidates = [
        (windows_fonts / "arial.ttf", windows_fonts / "arialbd.ttf"),
        (windows_fonts / "NotoSans-Regular.ttf", windows_fonts / "NotoSans-Bold.ttf"),
        (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
        (Path("/System/Library/Fonts/Supplemental/Arial.ttf"), Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")),
    ]
    if font is not None:
        candidates = [(font, font)]
    for regular, bold in candidates:
        if regular.is_file() and bold.is_file():
            try:
                regular_face = TTFont("ReviewSans", str(regular))
                bold_face = TTFont("ReviewSansBold", str(bold))
            except TTFError as exc:
                if font is not None:
                    raise ValueError("передайте корректный TTF-шрифт через --font") from exc
                continue
            if not all(ord(char) in regular_face.face.charToGlyph for char in "АяЁ"):
                continue
            pdfmetrics.registerFont(regular_face)
            pdfmetrics.registerFont(bold_face)
            return regular_face.fontName, bold_face.fontName
    raise ValueError("не найден TTF-шрифт с кириллицей. Установите DejaVu Sans или передайте --font путь.ttf")


def to_pdf(blocks: list[Text | Table], fonts: tuple[str, str]) -> bytes:
    regular, bold = fonts
    navy = colors.HexColor("#132A40")
    ink = colors.HexColor("#213449")
    teal = colors.HexColor("#175A59")
    grey = colors.HexColor("#486477")
    styles = {
        "title": ParagraphStyle("title", fontName=bold, fontSize=24, leading=30, textColor=navy, spaceAfter=16, keepWithNext=True),
        "section": ParagraphStyle("section", fontName=bold, fontSize=16, leading=22, textColor=teal, spaceBefore=14, spaceAfter=8, keepWithNext=True),
        "subheading": ParagraphStyle("subheading", fontName=bold, fontSize=11, leading=16, textColor=navy, spaceBefore=8, spaceAfter=4, keepWithNext=True),
        "finding": ParagraphStyle("finding", fontName=bold, fontSize=10, leading=14, textColor=teal, spaceBefore=5, spaceAfter=3, keepWithNext=True),
        "body": ParagraphStyle("body", fontName=regular, fontSize=9.4, leading=13.8, textColor=ink, spaceAfter=5),
        "quote": ParagraphStyle("quote", fontName=regular, fontSize=8.5, leading=12, textColor=grey, leftIndent=10, spaceAfter=5),
        "bullet": ParagraphStyle("bullet", fontName=regular, fontSize=9.4, leading=13.8, textColor=ink, leftIndent=10, spaceAfter=5),
        "cell": ParagraphStyle("cell", fontName=regular, fontSize=8, leading=11.3, textColor=ink),
        "header": ParagraphStyle("header", fontName=bold, fontSize=8, leading=11.3, textColor=colors.white),
    }
    output = BytesIO()
    pdf = SimpleDocTemplate(
        output, pagesize=A4, leftMargin=46, rightMargin=46, topMargin=45, bottomMargin=48,
        title=TITLE, author="AI Requirements Validator",
    )
    story = []
    for block in blocks:
        if isinstance(block, Text):
            value = ("• " if block.style == "bullet" else "") + block.value
            story.append(Paragraph(escape(value), styles[block.style]))
            continue
        rows = [[Paragraph(escape(cell), styles["header"]) for cell in block.header]]
        rows += [[Paragraph(escape(cell), styles["cell"]) for cell in row] for row in block.rows]
        table = LongTable(rows, colWidths=[pdf.width * x for x in (.13, .13, .29, .45)], repeatRows=1, splitInRow=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), navy),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF4F6")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, 1), (-1, -1), .3, colors.HexColor("#D6E3E8")),
        ]))
        story.append(table)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D6E3E8"))
        canvas.line(46, 36, A4[0] - 46, 36)
        canvas.setFont(regular, 8)
        canvas.setFillColor(grey)
        canvas.drawString(46, 24, "AI Requirements Validator")
        canvas.drawRightString(A4[0] - 46, 24, f"Страница {doc.page}")
        canvas.restoreState()

    pdf.build(story, onFirstPage=footer, onLaterPages=footer)
    return output.getvalue()
