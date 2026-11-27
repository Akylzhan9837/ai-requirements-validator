"""Load a document and map evidence to source lines or PDF pages."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re

from pypdf import PdfReader
from pypdf.errors import PdfReadError


def compact(text: str) -> str:
    return " ".join(text.split())


def displayed_markdown(pieces: list[tuple[int, str]]) -> tuple[str, list[int]]:
    """Read displayed Markdown while retaining offsets into the source."""
    visible = []
    cursor = 0
    for _, piece in pieces:
        prefix = re.match(r"^(?:[-*+] |#{1,6} )", piece)
        start = prefix.end() if prefix else 0
        hidden = set()
        for marker in re.finditer(r"\*\*[^*\n]+\*\*|`[^`]+`", piece):
            size = 2 if marker[0].startswith("**") else 1
            hidden.update(range(marker.start(), marker.start() + size))
            hidden.update(range(marker.end() - size, marker.end()))
        for index, char in enumerate(piece):
            if index < start or index in hidden:
                continue
            if char == " " and (not visible or visible[-1][0] == " "):
                continue
            visible.append((char, cursor + index))
        cursor += len(piece)
        if visible and visible[-1][0] != " ":
            visible.append((" ", cursor))
        cursor += 1
    if visible and visible[-1][0] == " ":
        visible.pop()
    return "".join(char for char, _ in visible), [position for _, position in visible]


@dataclass(frozen=True)
class Line:
    text: str
    page: int | None = None
    page_line: int | None = None


@dataclass(frozen=True)
class Evidence:
    start: int
    end: int
    quote: str
    location: str


@dataclass(frozen=True)
class Document:
    title: str
    name: str
    digest: str
    lines: tuple[Line, ...]

    def numbered_text(self) -> str:
        return "\n".join(
            f"{number:04d} | {line.text}"
            for number, line in enumerate(self.lines, 1)
            if line.text.strip()
        )

    def evidence(self, item: dict) -> Evidence:
        if not isinstance(item, dict):
            raise ValueError("ссылка на источник должна быть объектом")
        start, end, quote = (item.get(key) for key in ("start", "end", "quote"))
        if type(start) is not int or type(end) is not int or not 1 <= start <= end <= len(self.lines):
            raise ValueError("ссылка содержит некорректный диапазон строк")
        if not isinstance(quote, str) or not compact(quote):
            raise ValueError(f"пустая цитата в строках {start}–{end}")

        needle = compact(quote)

        def search(first: int, last: int) -> tuple[list[tuple[int, str]], str, int]:
            pieces = [
                (number, compact(self.lines[number - 1].text))
                for number in range(first, last + 1)
                if self.lines[number - 1].text.strip()
            ]
            text = " ".join(value for _, value in pieces)
            return pieces, text, text.find(needle)

        pieces, haystack, offset = search(start, end)
        if offset < 0:
            pieces, haystack, offset = search(max(1, start - 3), min(len(self.lines), end + 3))
            if offset >= 0 and haystack.find(needle, offset + 1) >= 0:
                offset = -1
        limit = offset + len(needle)
        if offset < 0 and self.name.lower().endswith(".md"):
            visible, positions = displayed_markdown(pieces)
            candidate, _ = displayed_markdown([(0, needle)])
            index = visible.find(candidate) if candidate else -1
            if index >= 0 and visible.find(candidate, index + 1) < 0:
                offset, limit = positions[index], positions[index + len(candidate) - 1] + 1
        if offset < 0:
            raise ValueError(f"цитата не совпадает с источником в строках {start}–{end}")
        cursor = 0
        actual = []
        for number, text in pieces:
            if cursor < limit and cursor + len(text) > offset:
                actual.append(number)
            cursor += len(text) + 1
        first, last = actual[0], actual[-1]
        return Evidence(first, last, haystack[offset:limit], self.location(first, last))

    def location(self, start: int, end: int) -> str:
        first, last = self.lines[start - 1], self.lines[end - 1]
        if first.page is None:
            return f"строка {start}" if start == end else f"строки {start}–{end}"
        if start == end:
            return f"стр. {first.page}, строка {first.page_line}"
        if first.page == last.page:
            return f"стр. {first.page}, строки {first.page_line}–{last.page_line}"
        return f"стр. {first.page}, строка {first.page_line} — стр. {last.page}, строка {last.page_line}"


def read_document(source: Path) -> Document:
    suffix = source.suffix.lower()
    if suffix not in {".md", ".pdf"}:
        raise ValueError("поддерживаемые расширения входа: .md и .pdf")
    raw = source.read_bytes()

    if suffix == ".md":
        lines = [Line(text) for text in raw.decode("utf-8-sig").splitlines()]
        title = next(
            (match[1].strip() for line in lines if (match := re.match(r"^\s*#\s+(.+?)\s*#*\s*$", line.text))),
            source.stem,
        )
    else:
        try:
            reader = PdfReader(BytesIO(raw))
            if reader.is_encrypted:
                raise ValueError("снимите пароль с входного PDF")
            title = str(reader.metadata.title or "").strip() if reader.metadata else ""
            lines = []
            for page_number, page in enumerate(reader.pages, 1):
                page_lines = (page.extract_text() or "").splitlines()
                if not any(line.strip() for line in page_lines):
                    raise ValueError(
                        f"PDF: страница {page_number} без текстового слоя. "
                        "Пропустите документ через OCR или передайте Markdown."
                    )
                lines.extend(
                    Line(text, page_number, number)
                    for number, text in enumerate(page_lines, 1)
                )
        except (PdfReadError, NotImplementedError) as exc:
            raise ValueError("не удалось прочитать входной PDF") from exc
        title = title or next((line.text.strip() for line in lines if line.text.strip()), source.stem)

    if not any(line.text.strip() for line in lines):
        raise ValueError("во входном файле нет текста")
    if sum(len(line.text) for line in lines) > 100_000:
        raise ValueError("вход превышает 100 000 символов текста")
    return Document(compact(title), source.name, sha256(raw).hexdigest(), tuple(lines))
