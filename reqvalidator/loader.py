"""Чтение документа требований (md / txt / pdf / docx) и привязка цитат к строкам."""
from __future__ import annotations

import hashlib
import re
from bisect import bisect_right
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

SUPPORTED = (".md", ".markdown", ".txt", ".pdf", ".docx")


class LoadError(Exception):
    pass


@dataclass
class Line:
    no: int                 # номер строки, с 1
    text: str
    page: Optional[int]     # номер страницы (для PDF)
    section: str            # путь по заголовкам


@dataclass
class Location:
    line_start: int
    line_end: int
    text: str               # найденный фрагмент (для нечёткого совпадения — текст из документа)
    exact: bool


def _norm_with_map(lines: list[Line]) -> tuple[str, list[int]]:
    """Склеивает документ в строку из одних букв/цифр; возвращает строку и карту символ→номер строки."""
    chars: list[str] = []
    owners: list[int] = []
    for ln in lines:
        for ch in ln.text.lower().replace("ё", "е"):
            if ch.isalnum():
                chars.append(ch)
                owners.append(ln.no)
    return "".join(chars), owners


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower().replace("ё", "е") if ch.isalnum())


class Document:
    def __init__(self, path: Path, fmt: str, lines: list[Line], title: str, raw_bytes: bytes):
        self.path = path
        self.fmt = fmt
        self.lines = lines
        self.title = title
        self.sha256 = hashlib.sha256(raw_bytes).hexdigest()
        self._norm_text, self._owners = _norm_with_map(lines)
        self._line_norm = {ln.no: _norm(ln.text) for ln in lines}

    # ----------------------------------------------------------- представление
    @property
    def text(self) -> str:
        return "\n".join(ln.text for ln in self.lines)

    def numbered(self, start: int = 1, end: Optional[int] = None) -> str:
        """Текст с метками [L<номер>] — LLM видит номера строк и может на них опираться."""
        end = end or len(self.lines)
        return "\n".join(f"[L{ln.no}] {ln.text}" for ln in self.lines[start - 1:end])

    def sections_chunks(self, max_chars: int) -> list[tuple[int, int]]:
        """Делит документ на диапазоны строк ≤ max_chars, по возможности на границах заголовков."""
        chunks: list[tuple[int, int]] = []
        start, size = 1, 0
        for ln in self.lines:
            is_heading = ln.text.lstrip().startswith("#")
            if size + len(ln.text) + 8 > max_chars and size > 0 and is_heading:
                chunks.append((start, ln.no - 1))
                start, size = ln.no, 0
            elif size + len(ln.text) + 8 > max_chars * 1.25 and size > 0:
                chunks.append((start, ln.no - 1))
                start, size = ln.no, 0
            size += len(ln.text) + 8
        if self.lines:
            chunks.append((start, self.lines[-1].no))
        return chunks

    # --------------------------------------------------------------- локация
    def locate(self, quote: str) -> Optional[Location]:
        q = _norm(quote)
        if len(q) < 4:
            return None
        idx = self._norm_text.find(q)
        if idx >= 0:
            return Location(self._owners[idx], self._owners[idx + len(q) - 1], quote.strip(), True)
        return self._fuzzy(q)

    def _fuzzy(self, q: str, threshold: float = 0.80) -> Optional[Location]:
        best: tuple[float, int, int] = (0.0, 0, 0)
        n = len(self.lines)
        max_window = 3 if n <= 3000 else 1
        for i in range(n):
            acc = ""
            for w in range(max_window):
                if i + w >= n:
                    break
                acc += self._line_norm[self.lines[i + w].no]
                if not acc:
                    continue
                if abs(len(acc) - len(q)) > max(len(q), len(acc)) * 0.5:
                    continue
                sm = SequenceMatcher(None, q, acc, autojunk=False)
                if sm.real_quick_ratio() < threshold or sm.quick_ratio() < threshold:
                    continue
                r = sm.ratio()
                if r > best[0]:
                    best = (r, i, i + w)
        if best[0] >= threshold:
            a, b = self.lines[best[1]], self.lines[best[2]]
            text = " ".join(l.text.strip() for l in self.lines[best[1]:best[2] + 1])
            return Location(a.no, b.no, text, False)
        return None

    def section_of(self, line_no: int) -> str:
        return self.lines[line_no - 1].section if 1 <= line_no <= len(self.lines) else ""

    def page_of(self, line_no: int) -> Optional[int]:
        return self.lines[line_no - 1].page if 1 <= line_no <= len(self.lines) else None

    def place(self, loc: Optional[Location]) -> str:
        if loc is None:
            return "место не определено"
        parts = []
        page = self.page_of(loc.line_start)
        if page:
            parts.append(f"стр. {page}")
        rng = f"{loc.line_start}" if loc.line_start == loc.line_end else f"{loc.line_start}–{loc.line_end}"
        parts.append(f"строка {rng}")
        sec = self.section_of(loc.line_start)
        place = ", ".join(parts)
        return f"{place} (раздел «{sec}»)" if sec and not page else place


# ------------------------------------------------------------------ загрузка
_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def _build_lines(raw_lines: list[str], pages: Optional[list[int]] = None, md_headings: bool = True) -> tuple[list[Line], str]:
    lines: list[Line] = []
    stack: list[tuple[int, str]] = []
    title = ""
    in_code = False
    for i, raw in enumerate(raw_lines, start=1):
        text = raw.rstrip("\n").rstrip()
        if text.lstrip().startswith("```"):
            in_code = not in_code
        m = _HEADING.match(text) if md_headings and not in_code else None
        if m:
            level, name = len(m.group(1)), m.group(2).strip().strip("*")
            if level == 1 and not title:
                title = name
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, name))
        section = " › ".join(n for lv, n in stack if lv > 1) or (title or "—")
        lines.append(Line(i, text, pages[i - 1] if pages else None, section))
    return lines, title


def _read_pdf(path: Path) -> tuple[list[str], list[int]]:
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover
        raise LoadError("Для чтения PDF нужен пакет pypdf (pip install -r requirements.txt)") from e
    reader = PdfReader(str(path))
    out: list[str] = []
    pages: list[int] = []
    for pno, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        for ln in txt.splitlines():
            out.append(ln)
            pages.append(pno)
    if not "".join(out).strip():
        raise LoadError(
            "В PDF нет текстового слоя (похоже на скан). Агент работает с текстовыми PDF — "
            "пропустите файл через OCR или подайте .md/.docx."
        )
    return out, pages


def _read_docx(path: Path) -> list[str]:
    try:
        import docx
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as e:  # pragma: no cover
        raise LoadError("Для чтения DOCX нужен пакет python-docx") from e
    d = docx.Document(str(path))
    out: list[str] = []
    for child in d.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, d)
            text = p.text.strip()
            if not text:
                continue
            style = (p.style.name or "") if p.style is not None else ""
            m = re.match(r"(?:Heading|Заголовок)\s*(\d)", style)
            if m:
                out.append("#" * int(m.group(1)) + " " + text)
            elif style.lower().startswith(("list", "спис")):
                out.append("* " + text)
            else:
                out.append(text)
        elif tag == "tbl":
            t = Table(child, d)
            for r_i, row in enumerate(t.rows):
                cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                out.append("| " + " | ".join(cells) + " |")
                if r_i == 0:
                    out.append("| " + " | ".join("---" for _ in cells) + " |")
    return out


def load_document(path: str | Path) -> Document:
    p = Path(path)
    if not p.is_file():
        raise LoadError(f"Файл не найден: {p}")
    ext = p.suffix.lower()
    if ext not in SUPPORTED:
        raise LoadError(f"Формат {ext or '(без расширения)'} не поддерживается. Допустимо: {', '.join(SUPPORTED)}")
    raw_bytes = p.read_bytes()

    pages: Optional[list[int]] = None
    if ext == ".pdf":
        raw_lines, pages = _read_pdf(p)
        lines, title = _build_lines(raw_lines, pages, md_headings=False)
    elif ext == ".docx":
        lines, title = _build_lines(_read_docx(p))
    else:
        try:
            text = raw_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw_bytes.decode("cp1251", errors="replace")
        lines, title = _build_lines(text.splitlines())

    if not any(l.text.strip() for l in lines):
        raise LoadError("Документ пуст")
    if not title:
        title = next((l.text.strip() for l in lines if l.text.strip()), p.stem)[:120]
    return Document(p, ext.lstrip("."), lines, title, raw_bytes)
