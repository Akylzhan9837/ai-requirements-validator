"""Сборка отчёта в Markdown (детерминированный шаблон поверх результатов конвейера)."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Optional

from .criteria import BY_CODE, REQ_CRITERIA, SEVERITY_DEF, SEVERITY_RU, SET_CRITERIA
from .llm import LLMStats
from .models import Finding, Requirement
from .pipeline import SEV_ORDER, RunResult

KIND_RU = {
    "functional": "функц.", "nonfunctional": "нефункц.", "data": "данные",
    "integration": "интеграция", "interface": "UI/UX", "acceptance": "приёмка",
    "business_rule": "цель/правило", "other": "прочее",
}
CATEGORY = {  # группировка критичных дефектов для сводки
    "VERIF": "Непроверяемые", "CONS": "Противоречивые", "S_CONS": "Противоречивые",
    "COMPL": "Неполные", "S_COMPL": "Неполные", "S_VERIF": "Неполные", "UNAMB": "Двусмысленные",
}
CATEGORY_ORDER = ["Непроверяемые", "Противоречивые", "Неполные", "Двусмысленные", "Прочие критичные"]


def cell(s: object) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ").strip() or "—"


def short(s: str, n: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def crit_label(code: str) -> str:
    c = BY_CODE[code]
    return f"{c.num} {c.name}"


def req_status(res: RunResult, rid: str) -> str:
    rr = res.results[rid]
    if not rr.checked:
        return "Не проверено"
    if any(f.severity == "critical" for f in rr.findings):
        return "Критично"
    return "Есть замечания" if rr.findings else "ОК"


def verdict(res: RunResult, all_f: list[Finding]) -> tuple[str, str]:
    crit = sum(f.severity == "critical" for f in all_f)
    major = sum(f.severity == "major" for f in all_f)
    if crit:
        return ("НЕТ — начинать разработку и тестирование по этому документу нельзя",
                f"найдено критичных дефектов: {crit}; пока они не устранены, результат разработки нельзя однозначно проверить")
    if res.unchecked:
        return ("УСЛОВНО — проверка неполная", "часть требований не удалось проверить; повторите прогон")
    if major:
        return ("УСЛОВНО — можно начинать после уточнения существенных замечаний",
                f"критичных дефектов нет, но есть существенные замечания: {major}")
    return ("ДА — можно начинать разработку и тестирование", "критичных и существенных замечаний не найдено")


def build_markdown(res: RunResult, *, model: str, agent_ver: str, stats: LLMStats, argv: str) -> str:
    doc = res.doc
    reqs = res.requirements
    by_rid = {r.rid: r for r in reqs}
    req_f = [f for r in reqs for f in res.results[r.rid].findings]
    all_f = req_f + res.set_findings
    out: list[str] = []
    w = out.append

    # ------------------------------------------------------------------ шапка
    w(f"# Отчёт о проверке требований: {doc.title}\n")
    w("> Отчёт сформирован автоматически ИИ-агентом проверки требований по выданному документу. "
      "Замечания носят экспертный характер и требуют подтверждения автором документа.\n")

    # --------------------------------------------------------------- 1. паспорт
    started = datetime.fromtimestamp(res.started).astimezone()
    w("## 1. Паспорт прогона\n")
    w("| Поле | Значение |")
    w("| :--- | :--- |")
    for k, v in [
        ("Документ", doc.title),
        ("Файл", f"{doc.path.name} ({doc.fmt}, {len(doc.lines)} строк)"),
        ("SHA-256 документа", doc.sha256),
        ("Дата и время прогона", started.strftime("%d.%m.%Y %H:%M:%S %z")),
        ("Модель", f"{model} (Google Gemini API, temperature = 0)"),
        ("Версия агента", agent_ver),
        ("Команда запуска", argv),
        ("Вызовов LLM", f"{stats.calls + stats.cached} (из них из кэша прежних вызовов: {stats.cached}; повторов из-за лимитов/ошибок: {stats.retries})"),
        ("Токены (по новым вызовам)", f"вход {stats.prompt_tokens}, выход {stats.output_tokens}"),
        ("Длительность прогона", f"{res.finished - res.started:.0f} с"),
    ]:
        w(f"| {cell(k)} | {cell(v)} |")
    w("")

    # ----------------------------------------------------------- 2. нарезка
    w("## 2. Как агент нарезал требования\n")
    w("Агент читает документ целиком, выделяет каждое отдельное предписание (функциональные, нефункциональные, данные, интеграция, "
      "критерии приёмки, правила API) и привязывает его к тексту по дословной цитате: цитата, которой нет в документе, отбрасывается. "
      "Если у пункта есть собственный идентификатор — он сохранён; если нет — агент присвоил идентификатор вида R-NNN "
      "(в колонке «ID в документе» стоит «—, присвоен агентом»). Строки считаются по исходному файлу.\n")
    w("| ID в отчёте | ID в документе | Краткая формулировка | Вид | Место в документе | Цитата |")
    w("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in reqs:
        place = doc.place(doc.locate(r.quote))
        docid = r.doc_id if r.doc_id else "—, присвоен агентом"
        w(f"| {r.rid} | {cell(docid)} | {cell(r.summary)} | {KIND_RU.get(r.kind, r.kind)} | {cell(place)} | {cell('«' + short(r.quote, 220) + '»')} |")
    w("")

    # ---------------------------------------------------- 3. каждое требование
    w("## 3. Проверка каждого требования\n")
    w("### 3.1. Критерии и шкала серьёзности\n")
    w("Критерии взяты из ISO/IEC/IEEE 29148:2018 (п. 5.2.5) и IEEE Std 830-1998 (п. 4.3).\n")
    w("| № | Критерий | Что проверяется | Источник |")
    w("| :--- | :--- | :--- | :--- |")
    for c in REQ_CRITERIA:
        w(f"| {c.num} | {c.name} | {cell(c.description)} | {cell(c.source)} |")
    w("")
    w("| Серьёзность | Значение |")
    w("| :--- | :--- |")
    for k in ("critical", "major", "minor"):
        w(f"| {SEVERITY_RU[k]} | {cell(SEVERITY_DEF[k])} |")
    w("")

    w("### 3.2. Сводная матрица\n")
    w("Обозначения: ✓ — критерий выполнен, ✗ — есть нарушение, — — критерий неприменим к этому виду требования, ? — не проверено.\n")
    head = ["ID", "Док. ID"] + [c.num for c in REQ_CRITERIA] + ["Статус", "Замеч."]
    w("| " + " | ".join(head) + " |")
    w("| " + " | ".join([":---"] * 2 + [":---:"] * len(REQ_CRITERIA) + [":---", ":---:"]) + " |")
    for r in reqs:
        rr = res.results[r.rid]
        failed = {f.criterion for f in rr.findings}
        marks = []
        for c in REQ_CRITERIA:
            if not rr.checked:
                marks.append("?")
            elif c.code in failed:
                marks.append("✗")
            elif c.code in rr.not_applicable:
                marks.append("—")
            else:
                marks.append("✓")
        w("| " + " | ".join([r.rid, cell(r.doc_id or "—")] + marks + [req_status(res, r.rid), str(len(rr.findings))]) + " |")
    w("")

    w("### 3.3. Подробно по требованиям\n")
    set_by_rid: dict[str, list[str]] = {}
    for f in res.set_findings:
        for rid in f.involved:
            set_by_rid.setdefault(rid, []).append(f.fid)
    for r in reqs:
        rr = res.results[r.rid]
        title = f"{r.rid}" + (f" · {r.doc_id}" if r.doc_id else "") + f" — {short(r.summary, 90)}"
        w(f"#### {title}\n")
        w(f"**Статус:** {req_status(res, r.rid)} · **Вид:** {KIND_RU.get(r.kind, r.kind)} · **Место:** {doc.place(doc.locate(r.quote))}\n")
        w(f"> «{short(r.quote, 600)}»\n")
        if not rr.checked:
            w("Требование не проверено из-за ошибки обращения к модели.\n")
            continue
        if not rr.findings:
            w("Нарушений критериев не найдено.\n")
        for f in rr.findings:
            w(f"**{f.fid} · {SEVERITY_RU[f.severity]} · {crit_label(f.criterion)}**\n")
            w(f"- **Фрагмент:** «{short(f.quotes[0], 300)}» ({f.places[0]}){' — фрагмент уточнён агентом по тексту документа' if f.quote_repaired else ''}")
            w(f"- **Проблема:** {f.problem}")
            w(f"- **Как исправить:** {f.fix}\n")
        if rr.rewrite and rr.findings:
            w(f"**Предлагаемая формулировка:** {rr.rewrite}\n")
        if r.rid in set_by_rid:
            w(f"**См. также замечания к набору:** {', '.join(set_by_rid[r.rid])}\n")

    # ---------------------------------------------------------- 4. набор
    w("## 4. Проверка всего набора требований\n")
    w("Агент сопоставляет требования между собой, с критериями приёмки, диаграммами и частями документа; ищет противоречия, пропуски, "
      "расхождения терминов, непокрытые требования, проблемы структуры и версионности. "
      "Критерии: ISO/IEC/IEEE 29148:2018 (п. 5.2.6) и IEEE Std 830-1998 (п. 4.3).\n")
    w("| № | Критерий набора | Замечаний | Наивысшая серьёзность |")
    w("| :--- | :--- | :---: | :--- |")
    for c in SET_CRITERIA:
        fs = [f for f in res.set_findings if f.criterion == c.code]
        top = SEVERITY_RU[min((f.severity for f in fs), key=SEV_ORDER.get)] if fs else "—"
        w(f"| {c.num} | {c.name} | {len(fs)} | {top} |")
    w("")
    if not res.set_findings:
        w("Замечаний к набору в целом не найдено (или проверка набора не выполнена — см. приложение).\n")
    for f in res.set_findings:
        w(f"#### {f.fid} · {SEVERITY_RU[f.severity]} · {crit_label(f.criterion)}\n")
        if f.involved:
            parts = []
            for rid in f.involved:
                rq = by_rid[rid]
                parts.append(f"{rid}" + (f" ({rq.doc_id})" if rq.doc_id else ""))
            w(f"**Затронуты требования:** {', '.join(parts)}\n")
        for q, p in zip(f.quotes, f.places):
            w(f"> «{short(q, 400)}» — {p}\n")
        w(f"- **Проблема:** {f.problem}")
        w(f"- **Как исправить:** {f.fix}\n")

    # --------------------------------------------------------- 5. сводка
    w("## 5. Сводка\n")
    kinds = Counter(KIND_RU.get(r.kind, r.kind) for r in reqs)
    with_f = sum(1 for r in reqs if res.results[r.rid].findings)
    cnt = Counter(f.severity for f in all_f)
    w("| Показатель | Значение |")
    w("| :--- | :--- |")
    rows = [
        ("Число требований", f"{len(reqs)} ({', '.join(f'{k}: {v}' for k, v in kinds.most_common())})"),
        ("Требований с нарушениями", f"{with_f} из {len(reqs)}"),
        ("Требований без нарушений", str(sum(1 for r in reqs if res.results[r.rid].checked and not res.results[r.rid].findings))),
        ("Не проверено из-за ошибки модели", str(len(res.unchecked))),
        ("Число нарушений всего", f"{len(all_f)} (к требованиям: {len(req_f)}, к набору: {len(res.set_findings)})"),
        ("По серьёзности", f"критично: {cnt['critical']}, существенно: {cnt['major']}, незначительно: {cnt['minor']}"),
    ]
    for k, v in rows:
        w(f"| {k} | {cell(v)} |")
    w("")

    w("### Нарушения по критериям\n")
    by_code = Counter(f.criterion for f in all_f)
    w("| Критерий | Замечаний |")
    w("| :--- | :---: |")
    for c in REQ_CRITERIA + SET_CRITERIA:
        if by_code.get(c.code):
            w(f"| {c.num} {c.name} | {by_code[c.code]} |")
    w("")

    w("### Критичные дефекты\n")
    crit = [f for f in all_f if f.severity == "critical"]
    if not crit:
        w("Критичных дефектов не найдено.\n")
    else:
        w("| Категория | Замечание | К чему относится | Суть |")
        w("| :--- | :--- | :--- | :--- |")
        groups: dict[str, list[Finding]] = {k: [] for k in CATEGORY_ORDER}
        for f in crit:
            groups[CATEGORY.get(f.criterion, "Прочие критичные")].append(f)
        for cat in CATEGORY_ORDER:
            for f in groups[cat]:
                if f.scope == "requirement":
                    rq = by_rid[f.rid]
                    about = f"{f.rid}" + (f" ({rq.doc_id})" if rq.doc_id else "")
                else:
                    about = ", ".join(f.involved) or "документ в целом"
                w(f"| {cat} | {f.fid} | {cell(about)} | {cell(short(f.problem, 260))} |")
        w("")

    # --------------------------------------------------------- 6. итог
    w("## 6. Итог\n")
    head, why = verdict(res, all_f)
    w(f"**Можно ли начинать разработку и тестирование: {head}.** {why[0].upper() + why[1:]}.\n")
    if res.rationale:
        w(res.rationale + "\n")
    w("### Что исправить в первую очередь\n")
    for i, p in enumerate(res.priorities, start=1):
        refs = f" (см. {', '.join(p['refs'])})" if p["refs"] else ""
        w(f"{i}. **{p['title']}**{refs} — {p['why']}")
    w("")

    # ---------------------------------------------------- приложение
    w("## Приложение. Ограничения и контроль качества прогона\n")
    w("- Отчёт сформирован языковой моделью; агент проверяет, что каждая цитата реально присутствует в тексте документа, "
      "но смысловые выводы модели следует подтверждать у автора требований.")
    w(f"- Замечаний к набору отброшено из-за отсутствия цитаты в документе: {res.dropped_findings}.")
    w(f"- Цитат уточнено агентом по тексту документа (модель дала неточную цитату): {res.repaired_quotes}.")
    for n in res.notes:
        w(f"- {n}")
    w("")
    return "\n".join(out)
