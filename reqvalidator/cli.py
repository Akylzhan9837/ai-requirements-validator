"""Командная строка: python agent.py <файл требований> <файл отчёта>"""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from .config import ROOT, ConfigError, agent_version, load_config
from .llm import GeminiClient, LLMError
from .loader import LoadError, load_document
from .pdf import markdown_to_pdf
from .pipeline import log, run_pipeline
from .report import build_markdown


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent.py",
        description="ИИ-агент проверки требований: читает документ с требованиями и пишет отчёт (Markdown + PDF).")
    p.add_argument("input", help="путь к документу требований (.md, .txt, .pdf, .docx)")
    p.add_argument("output", help="путь к отчёту (.md); PDF ляжет рядом с тем же именем")
    p.add_argument("--no-pdf", action="store_true", help="не создавать PDF-версию отчёта")
    p.add_argument("--model", help="модель Gemini (по умолчанию GEMINI_MODEL из .env)")
    p.add_argument("--batch-size", type=int, default=6, help="сколько требований проверять за один запрос (по умолчанию 6)")
    p.add_argument("--no-cache", action="store_true", help="не использовать кэш ответов LLM (.cache/llm)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        cfg = load_config(args.model)
    except ConfigError as e:
        print(f"Ошибка конфигурации: {e}", file=sys.stderr)
        return 4
    try:
        doc = load_document(args.input)
    except LoadError as e:
        print(f"Ошибка чтения документа: {e}", file=sys.stderr)
        return 2

    out = Path(args.output)
    md_path = out.with_suffix(".md") if out.suffix.lower() == ".pdf" else out
    pdf_path = md_path.with_suffix(".pdf")
    md_path.parent.mkdir(parents=True, exist_ok=True)

    log(f"Документ: «{doc.title}» ({doc.fmt}, {len(doc.lines)} строк). Модель: {cfg.model}")
    llm = GeminiClient(cfg.api_key, cfg.model, cfg.min_interval, cfg.max_retries,
                       cache_dir=None if args.no_cache else ROOT / ".cache" / "llm", verbose=args.verbose,
                       fallback_models=cfg.fallback_models)
    try:
        res = run_pipeline(doc, llm, batch_size=max(1, args.batch_size))
    except LLMError as e:
        print(f"\nОшибка работы с моделью: {e}", file=sys.stderr)
        return 3

    cmd = "python agent.py " + " ".join(shlex.quote(a) for a in (argv if argv is not None else sys.argv[1:]))
    md = build_markdown(res, model=cfg.model, agent_ver=agent_version(), stats=llm.stats, argv=cmd)
    md_path.write_text(md, encoding="utf-8")
    log(f"✔ Отчёт (Markdown): {md_path}")
    if not args.no_pdf:
        try:
            markdown_to_pdf(md, pdf_path, title=f"Отчёт о проверке требований: {doc.title}")
            log(f"✔ Отчёт (PDF): {pdf_path}")
        except Exception as e:  # PDF — дополнение; Markdown уже сохранён
            log(f"⚠ PDF не создан: {e}")
    crit = sum(f.severity == "critical" for r in res.requirements for f in res.results[r.rid].findings) + \
        sum(f.severity == "critical" for f in res.set_findings)
    log(f"Требований: {len(res.requirements)}; критичных замечаний: {crit}; вызовов LLM: {llm.stats.calls} (кэш: {llm.stats.cached})")
    return 0
