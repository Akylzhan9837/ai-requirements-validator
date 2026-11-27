"""Markdown (подмножество, которое генерирует report.py) → PDF через reportlab. Шрифты DejaVu лежат в репозитории (кириллица)."""
from __future__ import annotations

import html
import math
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import HRFlowable, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

FONT_DIR = Path(__file__).parent / "fonts"
_registered = False


def _register_fonts() -> None:
    global _registered
    if _registered:
        return
    for name, fn in [("DejaVu", "DejaVuSans.ttf"), ("DejaVu-Bold", "DejaVuSans-Bold.ttf"),
                     ("DejaVu-Oblique", "DejaVuSans-Oblique.ttf"), ("DejaVu-Mono", "DejaVuSansMono.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / fn)))
    pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Oblique",
                                  boldItalic="DejaVu-Bold")
    _registered = True


def _styles() -> dict[str, ParagraphStyle]:
    base = dict(fontName="DejaVu", fontSize=9, leading=12.5, alignment=TA_LEFT, textColor=colors.HexColor("#1a1a1a"))

    def S(name: str, **kw) -> ParagraphStyle:  # заголовки не отрываются от следующего блока
        return ParagraphStyle(name, **{**base, **kw})

    blue = colors.HexColor("#0b3d91")
    return {
        "p": S("p", spaceAfter=4),
        "h1": S("h1", keepWithNext=1, fontName="DejaVu-Bold", fontSize=16, leading=20, spaceAfter=8, textColor=blue),
        "h2": S("h2", keepWithNext=1, fontName="DejaVu-Bold", fontSize=13, leading=17, spaceBefore=12, spaceAfter=6, textColor=blue),
        "h3": S("h3", keepWithNext=1, fontName="DejaVu-Bold", fontSize=10.5, leading=14, spaceBefore=8, spaceAfter=4),
        "h4": S("h4", keepWithNext=1, fontName="DejaVu-Bold", fontSize=9.5, leading=13, spaceBefore=8, spaceAfter=3, textColor=colors.HexColor("#222222")),
        "quote": S("quote", fontName="DejaVu-Oblique", fontSize=8.5, leading=11.5, textColor=colors.HexColor("#444444"),
                   leftIndent=8, spaceAfter=5),
        "cell": S("cell", fontSize=7.3, leading=9.3),
        "cellh": S("cellh", fontName="DejaVu-Bold", fontSize=7.3, leading=9.3, textColor=colors.white),
        "li": S("li", spaceAfter=1.5),
    }


_LONG = re.compile(r"(\S{26})(?=\S)")


def _inline(text: str) -> str:
    t = html.escape(text, quote=False)
    t = _LONG.sub("\\1\u200b", t)
    t = re.sub(r"`([^`]+)`", r'<font face="DejaVu-Mono">\1</font>', t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = t.replace("\\|", "|")
    return t


def _split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    return [c.strip() for c in re.split(r"(?<!\\)\|", s)]


def _table(rows: list[list[str]], st: dict[str, ParagraphStyle], width: float) -> Table:
    ncol = len(rows[0])
    rows = [r + [""] * (ncol - len(r)) if len(r) < ncol else r[:ncol] for r in rows]
    char_w = 4.0  # ≈ ширина символа шрифта 7.3 pt
    weights, mins = [], []
    for j in range(ncol):
        col = [r[j] for r in rows]
        lens = [len(c) for c in col[1:]] or [4]
        avg = sum(lens) / len(lens)
        weights.append(min(max(avg * 0.8 + max(lens) * 0.2, 4), 80) ** 0.9)
        longest_word = max((len(w) for c in col for w in re.sub(r"[*`]", "", c).split()), default=3)
        mins.append(min(longest_word, 16) * char_w + 8)
    # сначала минимумы (чтобы слова не рвались), остаток — пропорционально весам
    free = max(width - sum(mins), 0)
    total = sum(weights)
    widths = [m + free * w / total for m, w in zip(mins, weights)]
    scale = width / sum(widths)
    widths = [w * scale for w in widths]
    data = [[Paragraph(_inline(c), st["cellh"] if i == 0 else st["cell"]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3d91")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c0cc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f6fb")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


class _NumberedCanvas(canvas.Canvas):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._saved: list[dict] = []

    def showPage(self):
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved)
        for state in self._saved:
            self.__dict__.update(state)
            self.setFont("DejaVu", 7.5)
            self.setFillColor(colors.HexColor("#777777"))
            self.drawRightString(A4[0] - 15 * mm, 8 * mm, f"Стр. {self._pageNumber} из {total}")
            self.drawString(15 * mm, 8 * mm, "Отчёт агента проверки требований")
            super().showPage()
        super().save()


def markdown_to_pdf(md: str, path: Path, title: str = "Отчёт о проверке требований") -> None:
    _register_fonts()
    st = _styles()
    width = A4[0] - 30 * mm
    story: list = []
    lines = md.splitlines()
    i = 0
    para: list[str] = []

    def flush() -> None:
        if para:
            story.append(Paragraph(_inline(" ".join(para)), st["p"]))
            para.clear()

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s:
            flush(); i += 1; continue
        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            flush()
            story.append(Paragraph(_inline(m.group(2)), st[f"h{len(m.group(1))}"]))
            i += 1; continue
        if s == "---":
            flush(); story.append(HRFlowable(width="100%", color=colors.HexColor("#b8c0cc"))); i += 1; continue
        if s.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = lines[i].strip()
                if not re.fullmatch(r"\|?(\s*:?-{2,}:?\s*\|)+\s*:?-*:?\s*\|?", row):
                    rows.append(_split_row(row))
                i += 1
            if rows:
                story.append(_table(rows, st, width)); story.append(Spacer(1, 6))
            continue
        if s.startswith(">"):
            flush()
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip()); i += 1
            story.append(Paragraph(_inline(" ".join(buf)), st["quote"])); continue
        if re.match(r"^[-*]\s+", s) or re.match(r"^\d+\.\s+", s):
            flush()
            numbered = bool(re.match(r"^\d+\.\s+", s))
            items = []
            while i < len(lines) and re.match(r"^([-*]|\d+\.)\s+", lines[i].strip()):
                items.append(re.sub(r"^([-*]|\d+\.)\s+", "", lines[i].strip())); i += 1
            story.append(ListFlowable([ListItem(Paragraph(_inline(t), st["li"]), leftIndent=14) for t in items],
                                      bulletType="1" if numbered else "bullet", start="1" if numbered else "•",
                                      bulletFontName="DejaVu", bulletFontSize=8, leftIndent=14))
            story.append(Spacer(1, 3)); continue
        para.append(s); i += 1
    flush()

    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm,
                            bottomMargin=16 * mm, title=title, author="Агент проверки требований")
    doc.build(story, canvasmaker=_NumberedCanvas)
