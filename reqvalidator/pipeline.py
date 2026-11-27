"""Конвейер проверки: нарезка → проверка каждого требования → проверка набора → итог."""
from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import prompts
from .criteria import BY_CODE, REQ_CRITERIA
from .lint import lint_document, lint_text
from .llm import LLM, LLMError
from .loader import Document
from .models import (CheckBatchOut, ExtractOut, ExtractedReq, FinalizeOut, Finding, FindingOut, Requirement,
                     ReqResult, SetCheckOut, SetFindingOut)

SEV_ORDER = {"critical": 0, "major": 1, "minor": 2}
MAX_CONTEXT_CHARS = 150_000
EXTRACT_CHUNK_CHARS = 60_000


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


@dataclass
class RunResult:
    doc: Document
    requirements: list[Requirement]
    results: dict[str, ReqResult]
    set_findings: list[Finding]
    rationale: str
    priorities: list[dict]
    notes: list[str] = field(default_factory=list)          # предупреждения для приложения отчёта
    dropped_findings: int = 0
    repaired_quotes: int = 0
    unchecked: list[str] = field(default_factory=list)
    started: float = 0.0
    finished: float = 0.0


# ================================================================ 1. нарезка
_ID_RE = re.compile(r"^([A-ZА-ЯЁ]{1,6}-\d{1,3})(?![\w-])", re.UNICODE)


def _doc_ids_in_text(doc: Document) -> set[str]:
    """Явные идентификаторы пунктов («Ф-1», «НФТ-2») в начале строк/ячеек — для проверки полноты нарезки."""
    ids: set[str] = set()
    for ln in doc.lines:
        s = re.sub(r"^[\s*|>\-•#]+", "", ln.text)
        m = _ID_RE.match(s)
        if m:
            ids.add(m.group(1))
    return ids


def _strip_markup(s: str) -> str:
    return re.sub(r"[*`]+", "", s).strip()


def extract_requirements(doc: Document, llm: LLM, res: RunResult) -> list[Requirement]:
    raw: list[ExtractedReq] = []
    chunks = doc.sections_chunks(EXTRACT_CHUNK_CHARS)
    for i, (a, b) in enumerate(chunks, start=1):
        log(f"[1/4] Нарезка требований: фрагмент {i}/{len(chunks)} (строки {a}–{b})")
        out = llm.generate(prompts.SYSTEM, prompts.extract_prompt(doc.title, doc.numbered(a, b), (a, b), len(doc.lines)),
                           ExtractOut, tag=f"extract-{i}")
        raw.extend(out.requirements)

    # страховка полноты: все явные идентификаторы документа должны быть найдены
    present = {(r.doc_id or "").strip() for r in raw}
    missing = sorted(i for i in _doc_ids_in_text(doc) if i not in present)
    if missing:
        log(f"      пропущены явные ID {missing} — дозапрос")
        out = llm.generate(prompts.SYSTEM, prompts.extract_missing_prompt(doc.numbered(), missing), ExtractOut,
                           tag="extract-missing")
        raw.extend(r for r in out.requirements if (r.doc_id or "").strip() in missing)
        still = [m for m in missing if m not in {(r.doc_id or "").strip() for r in raw}]
        if still:
            res.notes.append(f"Агент не смог выделить пункты с идентификаторами: {', '.join(still)}.")

    # привязка к тексту, дедупликация, порядок по документу
    located: list[tuple[int, int, Requirement]] = []
    seen: set[tuple[int, int, str]] = set()
    for r in raw:
        quote = _strip_markup(r.quote)
        loc = doc.locate(quote)
        if loc is None:
            res.notes.append(f"Требование «{r.summary}» не удалось привязать к тексту документа (цитата не найдена); "
                             "оно исключено из проверки как возможная галлюцинация модели.")
            continue
        key = (loc.line_start, loc.line_end, (r.doc_id or "").strip())
        if key in seen:
            continue
        seen.add(key)
        text = quote if loc.exact else loc.text
        req = Requirement(
            rid="", doc_id=(r.doc_id or "").strip() or None, id_assigned_by_agent=not (r.doc_id or "").strip(),
            section=doc.section_of(loc.line_start) or r.section, kind=r.kind, quote=text,
            summary=_strip_markup(r.summary), line_start=loc.line_start, line_end=loc.line_end,
            page=doc.page_of(loc.line_start), located=True)
        located.append((loc.line_start, loc.line_end, req))
    located.sort(key=lambda t: (t[0], t[1]))
    reqs = [t[2] for t in located]
    for n, r in enumerate(reqs, start=1):
        r.rid = f"R-{n:03d}"
    if not reqs:
        raise LLMError("Модель не выделила ни одного требования. Проверьте документ и модель.")
    log(f"      выделено требований: {len(reqs)}")
    return reqs


# ================================================= 2. проверка каждого требования
def _context(doc: Document) -> str:
    full = doc.numbered()
    if len(full) <= MAX_CONTEXT_CHARS:
        return full
    return full[:MAX_CONTEXT_CHARS] + "\n[... документ обрезан по лимиту контекста ...]"


def _ground_quote(doc: Document, quote: str) -> tuple[Optional[str], Optional[str], bool]:
    """→ (цитата для отчёта, место, цитата_исправлена). None, если в документе не найдено."""
    q = _strip_markup(quote)
    loc = doc.locate(q)
    if loc is None:
        return None, None, False
    return (q if loc.exact else loc.text), doc.place(loc), not loc.exact


def check_requirements(doc: Document, reqs: list[Requirement], llm: LLM, res: RunResult, batch_size: int) -> dict[str, ReqResult]:
    ctx = _context(doc)
    results: dict[str, ReqResult] = {r.rid: ReqResult(rid=r.rid, checked=False) for r in reqs}
    by_rid = {r.rid: r for r in reqs}
    batches = [reqs[i:i + batch_size] for i in range(0, len(reqs), batch_size)]

    def run(batch: list[Requirement], tag: str) -> set[str]:
        payload = [{"rid": r.rid, "doc_id": r.doc_id, "kind": r.kind, "section": r.section, "quote": r.quote,
                    "lint": lint_text(r.quote)} for r in batch]
        out = llm.generate(prompts.SYSTEM, prompts.check_prompt(doc.title, ctx, payload), CheckBatchOut, tag=tag)
        done: set[str] = set()
        for c in out.checks:
            rid = c.rid.strip()
            if rid not in by_rid or rid in done or rid not in {b.rid for b in batch}:
                continue
            req = by_rid[rid]
            rr = results[rid]
            rr.checked = True
            rr.not_applicable = sorted(set(c.not_applicable))
            rr.rewrite = (c.rewrite or "").strip() or None
            seen: set[tuple[str, str]] = set()
            for f in c.findings:
                quote, place, repaired = _ground_quote(doc, f.quote)
                if quote is None:                       # цитата не найдена → привязываем к самому требованию
                    quote, place, repaired = req.quote, doc.place(doc.locate(req.quote)), True
                key = (f.criterion, quote)
                if key in seen:
                    continue
                seen.add(key)
                if repaired:
                    res.repaired_quotes += 1
                rr.findings.append(Finding(
                    fid="", scope="requirement", criterion=f.criterion, severity=f.severity, rid=rid,
                    quotes=[quote], places=[place or "—"], problem=f.problem.strip(), fix=f.fix.strip(),
                    quote_repaired=repaired))
            rr.findings.sort(key=lambda x: SEV_ORDER[x.severity])
            done.add(rid)
        return done

    for i, batch in enumerate(batches, start=1):
        log(f"[2/4] Проверка требований: пакет {i}/{len(batches)} ({batch[0].rid}–{batch[-1].rid})")
        try:
            done = run(batch, f"check-{i}")
        except LLMError as e:
            log(f"      ⚠ пакет не обработан: {e}")
            done = set()
        missed = [r for r in batch if r.rid not in done]
        if missed:                                       # однократный повтор для пропущенных
            log(f"      повтор для пропущенных: {', '.join(r.rid for r in missed)}")
            try:
                run(missed, f"check-{i}-retry")
            except LLMError as e:
                log(f"      ⚠ повтор не удался: {e}")
    res.unchecked = [rid for rid, r in results.items() if not r.checked]
    if res.unchecked:
        res.notes.append("Не удалось проверить (ошибка/пустой ответ модели): " + ", ".join(res.unchecked) +
                         ". Запустите прогон повторно — полученные ответы берутся из кэша, запрашиваются только недостающие.")
    return results


# ============================================================ 3. проверка набора
def check_set(doc: Document, reqs: list[Requirement], results: dict[str, ReqResult], llm: LLM, res: RunResult) -> list[Finding]:
    log("[3/4] Проверка набора требований целиком")
    table = [{"rid": r.rid, "doc_id": r.doc_id, "kind": r.kind, "section": r.section, "summary": r.summary,
              "quote": r.quote[:300]} for r in reqs]
    short = []
    for rid, rr in results.items():
        for f in rr.findings:
            short.append({"rid": rid, "criterion": f.criterion, "severity": f.severity, "problem": f.problem[:160]})
    hints = lint_document([l.text for l in doc.lines])
    try:
        out = llm.generate(prompts.SYSTEM, prompts.set_prompt(doc.title, _context(doc), table, short, hints),
                           SetCheckOut, tag="set-check")
    except LLMError as e:
        res.notes.append(f"Проверка набора не выполнена: {e}")
        return []
    valid_rids = {r.rid for r in reqs}
    findings: list[Finding] = []
    seen: set[str] = set()
    for f in out.findings:
        quotes, places = [], []
        for q in f.quotes:
            quote, place, repaired = _ground_quote(doc, q)
            if quote is None:
                continue
            if repaired:
                res.repaired_quotes += 1
            if quote not in quotes:
                quotes.append(quote)
                places.append(place or "—")
        if not quotes:                                   # без привязки к тексту замечание не принимается
            res.dropped_findings += 1
            continue
        sig = f.criterion + "|" + "|".join(sorted(quotes))
        if sig in seen:
            continue
        seen.add(sig)
        findings.append(Finding(
            fid="", scope="set", criterion=f.criterion, severity=f.severity,
            involved=[x for x in dict.fromkeys(i.strip() for i in f.involved) if x in valid_rids],
            quotes=quotes, places=places, problem=f.problem.strip(), fix=f.fix.strip()))
    findings.sort(key=lambda x: (SEV_ORDER[x.severity], x.criterion))
    return findings


# ================================================================ 4. итог
def finalize(res: RunResult, llm: LLM) -> None:
    log("[4/4] Итог и приоритеты")
    # сквозные идентификаторы замечаний
    n = 0
    for r in res.requirements:
        for f in res.results[r.rid].findings:
            n += 1
            f.fid = f"F-{n:03d}"
    for i, f in enumerate(res.set_findings, start=1):
        f.fid = f"N-{i:03d}"

    all_f = [f for r in res.requirements for f in res.results[r.rid].findings] + res.set_findings
    counts = {"требований": len(res.requirements), "замечаний": len(all_f),
              "critical": sum(f.severity == "critical" for f in all_f),
              "major": sum(f.severity == "major" for f in all_f),
              "minor": sum(f.severity == "minor" for f in all_f)}
    short = [{"id": f.fid, "severity": f.severity, "criterion": f.criterion,
              "about": f.rid or ", ".join(f.involved) or "документ в целом", "problem": f.problem[:200]}
             for f in sorted(all_f, key=lambda x: SEV_ORDER[x.severity])][:80]
    valid = {f.fid for f in all_f}
    try:
        out = llm.generate(prompts.SYSTEM, prompts.finalize_prompt(res.doc.title, counts, short), FinalizeOut, tag="finalize")
        res.rationale = out.rationale.strip()
        res.priorities = [{"title": p.title.strip(), "why": p.why.strip(), "refs": [x for x in p.refs if x in valid]}
                          for p in out.priorities]
    except LLMError as e:
        res.notes.append(f"Итоговая формулировка не получена от модели ({e}); приоритеты построены автоматически по серьёзности.")
        res.rationale = ""
        res.priorities = []
    if not res.priorities:                               # детерминированный запасной вариант
        top = sorted(all_f, key=lambda x: (SEV_ORDER[x.severity], x.scope != "set"))[:7]
        res.priorities = [{"title": f"{f.fid}: {BY_CODE[f.criterion].name}", "why": f.problem, "refs": [f.fid]} for f in top]


# ============================================================ оркестрация
def run_pipeline(doc: Document, llm: LLM, batch_size: int = 6) -> RunResult:
    res = RunResult(doc=doc, requirements=[], results={}, set_findings=[], rationale="", priorities=[],
                    started=time.time())
    res.requirements = extract_requirements(doc, llm, res)
    res.results = check_requirements(doc, res.requirements, llm, res, batch_size)
    res.set_findings = check_set(doc, res.requirements, res.results, llm, res)
    finalize(res, llm)
    res.finished = time.time()
    return res
