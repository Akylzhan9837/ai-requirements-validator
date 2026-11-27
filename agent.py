"""Command-line entry point for the requirements review agent."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from tempfile import NamedTemporaryFile

from gemini import Gemini, GeminiError, settings
from reporting import Passport, pdf_fonts, review_blocks, to_markdown, to_pdf
from review import review_document
from source import read_document


ROOT = Path(__file__).resolve().parent
VERSION = "v0.2.1"


def version() -> str:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *args], encoding="utf-8",
            stderr=subprocess.DEVNULL, timeout=4,
        ).strip()

    try:
        if Path(git("rev-parse", "--show-toplevel")).resolve() == ROOT:
            revision = git("rev-parse", "HEAD")
            dirty = git("status", "--porcelain", "--untracked-files=no")
            return f"{revision} (локальные изменения)" if dirty else revision
    except (OSError, subprocess.SubprocessError):
        pass
    return VERSION


def write_reports(markdown: Path, pdf: Path, markdown_bytes: bytes, pdf_bytes: bytes) -> None:
    markdown.parent.mkdir(parents=True, exist_ok=True)
    staged = []
    try:
        for target, contents in ((markdown, markdown_bytes), (pdf, pdf_bytes)):
            with NamedTemporaryFile(dir=target.parent, prefix=".review-", delete=False) as stream:
                temporary = Path(stream.name)
                staged.append((temporary, target))
                stream.write(contents)
        for temporary, target in staged:
            temporary.replace(target)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Статическое ревью требований в Markdown или PDF.")
    parser.add_argument("source", type=Path, help="входной файл .md или .pdf")
    parser.add_argument("report", type=Path, help="путь к отчёту .md (рядом появится .pdf)")
    parser.add_argument("--font", type=Path, help="путь к TTF-шрифту с кириллицей для PDF")
    args = parser.parse_args()
    if args.report.suffix.lower() != ".md":
        parser.error("путь к отчёту должен оканчиваться на .md")
    pdf = args.report.with_suffix(".pdf")
    for target in (args.report, pdf):
        if target.resolve() == args.source.resolve() or (
            target.exists() and args.source.exists() and target.samefile(args.source)
        ):
            parser.error("путь к отчёту совпадает с входным файлом")

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        document = read_document(args.source)
        fonts = pdf_fonts(args.font)
        model = Gemini(*settings(ROOT))
        revision = version()
        progress(f"Документ: {document.title}. Модель: {model.model}.")
        review = review_document(document, model, progress)
        blocks = review_blocks(document, review, Passport(
            started, model.model, ", ".join(sorted(model.versions)), revision, model.requests,
        ))
        write_reports(args.report, pdf, to_markdown(blocks).encode("utf-8"), to_pdf(blocks, fonts))
    except (GeminiError, ValueError, OSError) as exc:
        progress(f"Ошибка прогона: {exc}")
        return 1

    print(f"Markdown: {args.report.resolve()}\nPDF: {pdf.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
