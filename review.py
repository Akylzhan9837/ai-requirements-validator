"""Extract requirements and validate item and whole-document reviews."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, TypeVar

from gemini import Gemini, ModelResponseError
from source import Document, Evidence, compact


ITEM_CRITERIA = {
    "necessary": ("Необходимость", "Прослеживается назначение или обоснованное ограничение."),
    "appropriate": ("Уместность", "Уровень детализации оправдан. Проектное решение допустимо в контракте или при обоснованном ограничении."),
    "unambiguous": ("Однозначность", "Для терминов, условий и результата есть единая интерпретация."),
    "complete": ("Полнота", "Указаны необходимые условия, данные и ожидаемый результат."),
    "singular": ("Атомарность", "Выделено одно независимо проверяемое обязательство."),
    "feasible": ("Выполнимость", "Формулировка осуществима в рамках явно заданных ограничений."),
    "verifiable": ("Проверяемость", "Можно определить наблюдаемый результат и критерий прохождения."),
    "correct": ("Корректность", "Правило соответствует заявленной цели и условиям исходного документа."),
    "conforming": ("Единообразие", "Формулировка использует определённые в документе обозначения и применимый шаблон."),
    "traceable": ("Прослеживаемость", "Положение можно однозначно найти и связать с относящимися к нему положениями или приёмкой."),
}
SET_CRITERIA = {
    "complete": ("Полнота набора", "Покрыты нужные потоки, границы, ошибки, данные и приёмка в заявленной области."),
    "consistent": ("Согласованность", "Правила, интеграции, условия приёмки и термины совместимы во всех частях."),
    "feasible": ("Совместная выполнимость", "Ограничения и зависимости допускают совместную реализацию."),
    "verifiable": ("Проверяемость набора", "Набор позволяет вывести сквозные сценарии с однозначными ожидаемыми результатами."),
    "ranked": ("Ранжируемость", "Понятны важность, обязательность и порядок реализации там, где от них зависит объём поставки."),
    "maintainable": ("Модифицируемость и связи", "Структура, идентификаторы, версия и ссылки позволяют согласованно менять документ."),
}
CATEGORIES = {
    "conflict": "Противоречие",
    "unverifiable": "Непроверяемость",
    "incomplete": "Неполнота",
    "ambiguous": "Двусмысленность",
    "design": "Необоснованное проектное ограничение",
    "other": "Другое",
}
SEVERITIES = {"critical": "Критичный", "major": "Существенный", "minor": "Малый"}
STATUSES = {"pass": "Без замечаний", "fail": "Дефект", "na": "Не применимо или недостаточно данных"}
UNCHECKED_LABEL = "Модель не подтвердила оценку"
KINDS = {
    "business": "Бизнес-правило",
    "functional": "Функциональное",
    "quality": "Нефункциональное",
    "data": "Данные",
    "integration": "Интеграция / API",
    "acceptance": "Приёмка",
    "constraint": "Ограничение",
}
CRITICAL_CATEGORIES = {"conflict", "unverifiable", "incomplete", "ambiguous"}
BATCH_SIZE = 12


@dataclass(frozen=True)
class Requirement:
    id: str
    document_id: str
    kind: str
    summary: str
    source: Evidence

    def model_data(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "document_id": self.document_id,
            "summary": self.summary, "start": self.source.start, "end": self.source.end,
            "quote": self.source.quote,
        }


@dataclass(frozen=True)
class Finding:
    criterion: str
    category: str
    severity: str
    title: str
    explanation: str
    fix: str
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class Assessment:
    checks: dict[str, str]
    findings: tuple[Finding, ...]


@dataclass(frozen=True)
class Review:
    requirements: tuple[Requirement, ...]
    individual: dict[str, Assessment]
    whole: Assessment
    duplicates: tuple[tuple[int, int], ...] = ()


EVIDENCE_RULES = """
Нумерация в document дана слева от |, а дальше идёт исходная строка.
Ссылка содержит start, end (включительно) и quote — дословный непрерывный фрагмент
исходника внутри диапазона. Убирай служебные номера строк, но сохраняй исходные
знаки, включая Markdown внутри фрагмента. При переносах можно соединить строки
пробелом. Без многоточий, парафразов и склейки из несмежных отрывков.
Для Markdown-таблицы цитируй ячейку либо последовательный фрагмент строки таблицы.
"""
EXTRACT = """
Выдели обязательства по всему документу. Включи все явные правила, ограничения,
проверяемое поведение UI и процессов, нефункциональные условия, данные, интеграцию,
контракт API и каждый самостоятельный сценарий приёмки. Учитывай требования в
таблицах, списках, описании методов и непронумерованных абзацах. Рассматривай
сценарий приёмки как условие, действие и результат в одном элементе.
Разделяй независимо проверяемые правила, включая разные предложения под одним ID.
Сохраняй исходный ID у каждой части, если он действительно присвоен требованию.
В каждом разделе сопоставь каталог со всеми нормативными строками. Для API
выделяй правила методов, входных полей, выхода, статусов и ошибок, включая
таблицы и примеры, если на них держится контракт. Сохрани
самостоятельные ограничения, условия ошибок и правила пользовательского пути.
Номера строк, коды HTTP, названия полей и номера шагов не являются ID требований.
Для отсутствующего ID возвращай пустую строку. Если одна обязанность повторена
дословно, достаточно одного вхождения, но сохраняй отличающиеся условия и примеры
приёмки. Не добавляй требования из заголовков, пустых секций, незаполненных
чек-листов, справочных полей, пояснений и описаний без обязательств.
Цитата должна содержать смысл обязательства, включая условия и границы, когда они
даны. Краткая формулировка отражает смысл исходника и ничего к нему не добавляет.
Проверь охват всех частей и верни требования в порядке появления.
""" + EVIDENCE_RULES

REVIEW_RULES = """
Оцени каждый критерий значением pass (не обнаружен дефект по тексту), fail (есть
связанный дефект), na (не применим или данных недостаточно для вывода). Для
каждого fail нужен минимум один дефект. Все findings относятся к fail.
Выбери один ведущий критерий для одного исправления. Не размножай одну проблему.
Пиши конкретный сценарий и его наблюдаемое последствие. В fix задай решение,
которое должен принять автор, или конкретный способ уточнить текст. Не присваивай
проекту новые числовые значения, обязательства или архитектуру от себя.
Противоречие — два несовместимых утверждения при одинаковых применимых условиях.
Дай отдельную дословную цитату для каждой стороны. Явно заданное исключение или
разные контексты могут снять конфликт. Отличающиеся представления данных на
разных границах сами по себе допустимы. Проверь полный источник перед выводом о пропуске: нужное правило
может быть в другой части, таблице или примере. Если смысл зависит от различия
между соседними правилами, явно укажи это различие. Для пропуска цитируй положение или
подраздел, который создаёт необходимость в отсутствующем правиле; укажи,
какой именно случай невозможно реализовать или проверить. Не считай пропуском
необязательные детали реализации, например алгоритм для уже определённого
наблюдаемого поведения. Не додумывай регуляторные и продуктовые требования.
severity=critical только если противоречие, неполнота, непроверяемость или
двусмысленность блокирует однозначную разработку или приёмку ключевого сценария.
major существенно меняет поведение или покрытие, но не блокирует ключевой поток.
minor — локальный дефект с небольшим влиянием.
""" + EVIDENCE_RULES

REVIEW_ITEMS = """
Проверь каждый элемент из targets. Верни ровно один результат для каждого target.id.
Document содержит полный контекст, включая остальные части. Локальный дефект
подкрепи цитатой из проверяемого положения, лучше коротким фрагментом из
точной цитаты target.quote. Проверь определения терминов, измеримость условий,
границы, исключения и результат для прямо описанного действия.
Межпунктовые конфликты, пропуски
сквозных сценариев и согласование нескольких частей оставь для обзора набора.
Не заявляй локальный конфликт без двух различных цитат. Используй точную цитату
из target.quote для стороны, относящейся к проверяемому требованию.

""" + REVIEW_RULES

REVIEW_SET = """
Проверь весь document вместе, используя каталог requirements. Сопоставь правила,
ограничения, интеграции, примеры и критерии приёмки. Ищи расхождения между
родительской и дочерней спецификациями, а также противоречия внутри каждой.
Сначала сравни правила, относящиеся к одному событию, объекту или состоянию,
включая правила из соседних пунктов одного раздела. Для каждой пары общих
ограничений и специальных условий попробуй построить конкретную последовательность,
в которой они действуют одновременно: повторное событие, изменение состояния,
исчерпание лимита, переход между периодами. Если для одного наблюдаемого результата
одно правило требует действие, а другое его запрещает, проверь, есть ли явно
заданное исключение или приоритет в той же области. При отсутствии ясного
приоритета зафиксируй конфликт или двусмысленность и опиши последовательность.
Отдельно сверяй цель с измеримым событием и сроком, приёмку с правилами, а в API
сравни описание операции, способ вызова, фактическое действие и статусы ошибок.
Верификация набора обязательна. Сопоставь каждую строку таблицы приёмки с
относящимся к ней правилом и ожидаемым наблюдаемым результатом. Отдельно проверь
переходы через границы периода, разные даты одного события и граничные значения.
Если ожидаемый результат прямо расходится с базовым правилом, используй
conflict и critical для ключевого сценария. В такой карточке нужны две
цитаты: из правила и из ожидаемого результата.
Проверь покрытие проверками для нижней и верхней границ, смены настройки,
повторных событий и негативных сценариев. Если документ задаёт поведение,
но приёмочный сценарий для него отсутствует, сгруппируй недостающее покрытие
в одну карточку verifiable с цитатой из соответствующего правила и приёмки.
Для API проверь метод, фактическое действие и примеры ответа, в том числе
расхождение статуса и полезной нагрузки. Разные представления одного значения
на входе и выходе могут быть намеренным контрактом. Отсутствие необязательных
реализационных деталей само по себе не повод считать сценарий непроверяемым.
Свяжи каждый системный дефект с конкретными фрагментами документа. individual
содержит заголовки уже найденных локальных дефектов, чтобы не дублировать их.
""" + REVIEW_RULES


FINAL_REVIEW = """
Проведи независимую проверку взаимодействия правил. В первую очередь проверь
все пары общего ограничения на действие для одной сущности или периода и
локального правила, требующего того же действия после изменения параметра,
состояния или повторного события. Построй последовательность, где действие
уже произошло, условие меняется, затем выполняется триггер. Сопоставь область
действия: сброс локальной истории не означает автоматическую отмену ограничения
более широкого уровня. Для явного исключения текст должен охватывать тот же
объект, действие и период, что и общее правило. Если обе трактовки правдоподобны,
зафиксируй двусмысленность приоритета. Если оба правила применимы и требуют
несовместимого поведения, зафиксируй противоречие. Дай две отдельные цитаты
для каждого такого дефекта и конкретную последовательность событий в пояснении.
Дополнительно проверь два типа пропусков: несовместимость расчётного правила
с конкретным ожидаемым результатом и явно заданное поведение без проверок
существенных границ или альтернативных сценариев. Для несовместимых формулировок
нужны обе дословные цитаты. Конфликт в ожидаемом результате ключевой приёмки
классифицируй critical. Для пробела покрытия ищи нормативное правило и таблицу
или раздел приёмки. Новую карточку можно дать вместо существующей, если в ней
добавлены недостающая сторона или более точная классификация. При таком замещении
индексы новых findings равны existing_count + порядковый номер в findings,
считая от 1. Укажи индекс новой карточки в keep, старой в remove.
Затем проверь existing на дубли. Если два дефекта описывают одну причину и
устраняются одним исправлением, верни пару числовых ключей keep и remove.
Предпочитай карточку с точными цитатами и ясным объяснением. Один remove нельзя
указывать дважды или использовать как keep. Разные дефекты одного требования
сохрани, даже если у них общая цитата. Используй consistent, verifiable или complete.
""" + REVIEW_RULES + "\nВердикты по критериям вычислит программа из findings. Поле checks не требуется."


def string(choices: dict | None = None) -> dict:
    return {"type": "string", "enum": list(choices)} if choices else {"type": "string"}


def array(items: dict) -> dict:
    return {"type": "array", "items": items}


def obj(**properties: dict) -> dict:
    return {"type": "object", "properties": properties, "required": list(properties)}


EVIDENCE_SCHEMA = obj(start={"type": "integer"}, end={"type": "integer"}, quote=string())
EXTRACTION_SCHEMA = obj(requirements=array(obj(
    document_id=string(), kind=string(KINDS), summary=string(), source=EVIDENCE_SCHEMA,
)))


def assessment_schema(criteria: dict, **extra: dict) -> dict:
    return obj(
        **extra,
        checks=obj(**{key: string(STATUSES) for key in criteria}),
        findings=array(obj(
            criterion=string(criteria), category=string(CATEGORIES),
            severity=string(SEVERITIES), title=string(), explanation=string(),
            fix=string(), evidence=array(EVIDENCE_SCHEMA),
        )),
    )


T = TypeVar("T")


def validated_call(
    model: Gemini, prompt: str, data: dict, schema: dict, max_tokens: int,
    validate: Callable[[dict], T], stage: str, progress: Callable[[str], None],
) -> T:
    feedback = ""
    for attempt in range(3):
        try:
            return validate(model.generate(prompt + feedback, data, schema, max_tokens))
        except (ValueError, KeyError, TypeError) as exc:
            if attempt == 2:
                raise ModelResponseError(f"{stage}: {exc}") from None
            progress(f"{stage}: перепроверяю структуру ответа и ссылки ({attempt + 1}/2).")
            feedback = (
                "\nПроверь результат заново. В предыдущей попытке была ошибка: "
                f"{exc}. Верни полную исправленную структуру."
            )
    raise AssertionError("unreachable")


def text_field(data: dict, key: str, *, optional: bool = False) -> str:
    value = data[key]
    if not isinstance(value, str) or (not optional and not compact(value)):
        raise ValueError(f"поле {key} пустое или имеет неверный тип")
    return compact(value)


def list_field(data: dict, key: str) -> list[dict]:
    values = data[key]
    if not isinstance(values, list) or any(not isinstance(item, dict) for item in values):
        raise ValueError(f"поле {key} должно быть списком объектов")
    return values


def parse_inventory(data: dict, document: Document) -> tuple[Requirement, ...]:
    unique = {}
    for row in list_field(data, "requirements"):
        evidence = document.evidence(row["source"])
        original = text_field(row, "document_id", optional=True)
        if original:
            line = document.lines[evidence.start - 1].text
            prefix = " ".join(item.text for item in document.lines[max(0, evidence.start - 4):evidence.start])
            if original not in line and original not in prefix:
                raise ValueError(f"исходный ID отсутствует рядом со строкой {evidence.start}")
        kind = text_field(row, "kind")
        if kind not in KINDS:
            raise ValueError("неизвестный тип требования")
        key = (evidence.start, evidence.end, evidence.quote)
        unique.setdefault(key, (original, kind, text_field(row, "summary"), evidence))
    return tuple(
        Requirement(f"REQ-{number:03d}", *row)
        for number, row in enumerate(
            sorted(unique.values(), key=lambda item: (item[-1].start, item[-1].end)), 1
        )
    )


def parse_assessment(
    data: dict, document: Document, criteria: dict, target: Requirement | None = None,
) -> Assessment:
    checks = data["checks"]
    if not isinstance(checks, dict) or set(checks) != set(criteria):
        raise ValueError("пропущен критерий оценки")
    if any(not isinstance(status, str) or status not in STATUSES for status in checks.values()):
        raise ValueError("неизвестный вердикт")
    findings = []
    for row in list_field(data, "findings"):
        criterion, category, severity = (
            text_field(row, field) for field in ("criterion", "category", "severity")
        )
        if checks.get(criterion) != "fail" or category not in CATEGORIES or severity not in SEVERITIES:
            raise ValueError("некорректная категория, серьёзность или критерий дефекта")
        evidence = tuple(dict.fromkeys(document.evidence(item) for item in list_field(row, "evidence")))
        if not evidence or (category == "conflict" and len(evidence) < 2):
            raise ValueError("нет цитаты для дефекта или второй стороны противоречия")
        if target and not any(
            source.start <= target.source.end and source.end >= target.source.start
            and (source.quote in target.source.quote or target.source.quote in source.quote)
            for source in evidence
        ):
            raise ValueError(
                f"{target.id}: нужна цитата из самого требования "
                f"(строки {target.source.start}–{target.source.end})"
            )
        if severity == "critical" and category not in CRITICAL_CATEGORIES:
            raise ValueError("критический уровень допустим для дефекта, блокирующего основной сценарий")
        findings.append(Finding(
            criterion, category, severity, text_field(row, "title"),
            text_field(row, "explanation"), text_field(row, "fix"), evidence,
        ))
    distinct = tuple(dict.fromkeys(findings))
    if {key for key, status in checks.items() if status == "fail"} != {issue.criterion for issue in distinct}:
        raise ValueError("есть fail без привязанного дефекта")
    return Assessment({key: checks[key] for key in criteria}, distinct)


def review_batch(
    document: Document, model: Gemini, group: tuple[Requirement, ...],
    stage: str, progress: Callable[[str], None],
) -> dict[str, Assessment]:
    pending = {item.id: item for item in group}
    accepted = {}
    schema = obj(items=array(assessment_schema(ITEM_CRITERIA, id=string())))
    numbered = document.numbered_text()
    feedback = ""

    for attempt in range(4):
        progress(f"{stage}: осталось положений {len(pending)}.")
        data = {
            "document": numbered, "criteria": ITEM_CRITERIA,
            "targets": [item.model_data() for item in pending.values()],
        }
        errors = {}
        try:
            rows = list_field(model.generate(REVIEW_ITEMS + feedback, data, schema, 18_000), "items")
        except (ValueError, KeyError, TypeError) as exc:
            rows = []
            errors["ответ"] = str(exc)
        for row in rows:
            identifier = row.get("id")
            item = pending.get(identifier) if isinstance(identifier, str) else None
            if item is None:
                continue
            try:
                accepted[item.id] = parse_assessment(row, document, ITEM_CRITERIA, item)
            except (ValueError, KeyError, TypeError) as exc:
                errors[item.id] = str(exc)
            else:
                del pending[item.id]
        if not pending:
            return accepted
        feedback = (
            "\nВерни результаты только для текущих targets. "
            "Для каждой карточки используй дословную цитату из target.quote, если она подтверждает вывод. "
            "Две разные цитаты обязательны для conflict. Ошибки предыдущего ответа: "
            f"{errors or 'пропущены обязательные результаты'}."
        )

    progress(f"{stage}: не удалось подтвердить оценки {', '.join(pending)} после четырёх попыток.")
    accepted.update({
        identifier: Assessment(dict.fromkeys(ITEM_CRITERIA, "unchecked"), ())
        for identifier in pending
    })
    return accepted


def review_document(document: Document, model: Gemini, progress: Callable[[str], None]) -> Review:
    numbered = document.numbered_text()
    progress("Выделяю требования и критерии приёмки.")
    requirements = validated_call(
        model, EXTRACT, {"document": numbered}, EXTRACTION_SCHEMA, 26_000,
        lambda data: parse_inventory(data, document), "Выделение требований", progress,
    )
    progress(f"Выделено положений: {len(requirements)}.")
    groups = [
        requirements[start:start + BATCH_SIZE]
        for start in range(0, len(requirements), BATCH_SIZE)
    ]
    completed = {}
    with ThreadPoolExecutor(max_workers=3) as workers:
        futures = [
            workers.submit(review_batch, document, model, group, f"Ревью требований {index}/{len(groups)}", progress)
            for index, group in enumerate(groups, 1)
        ]
        for future in futures:
            completed.update(future.result())
    individual = {item.id: completed[item.id] for item in requirements}

    progress("Сопоставляю требования по всему документу.")
    whole = validated_call(
        model, REVIEW_SET,
        {
            "document": numbered, "criteria": SET_CRITERIA,
            "requirements": [item.model_data() for item in requirements],
            "individual": {key: [issue.title for issue in check.findings] for key, check in individual.items() if check.findings},
        },
        assessment_schema(SET_CRITERIA), 26_000,
        lambda data: parse_assessment(data, document, SET_CRITERIA), "Ревью набора", progress,
    )
    return finalize_review(document, model, Review(requirements, individual, whole), progress)


def ordered_findings(review: Review) -> list[tuple[str, Finding]]:
    return [
        (owner, issue)
        for owner, assessment in [*review.individual.items(), ("Набор", review.whole)]
        for issue in assessment.findings
    ]


def finalize_review(document: Document, model: Gemini, review: Review, progress: Callable[[str], None]) -> Review:
    existing = ordered_findings(review)
    criteria = {key: SET_CRITERIA[key] for key in ("consistent", "verifiable", "complete")}
    schema = assessment_schema(
        criteria,
        duplicates=array(obj(keep={"type": "integer"}, remove={"type": "integer"})),
    )

    schema["properties"].pop("checks")
    schema["required"].remove("checks")

    def validate(data: dict) -> Review:
        findings = list_field(data, "findings")
        checks = dict.fromkeys(criteria, "pass")
        for finding in findings:
            criterion = text_field(finding, "criterion")
            if criterion not in checks:
                raise ValueError("недопустимый критерий финальной сверки")
            checks[criterion] = "fail"
        interactions = parse_assessment({"checks": checks, "findings": findings}, document, criteria)
        duplicates = {}
        for row in list_field(data, "duplicates"):
            keep, remove = row["keep"], row["remove"]
            if (
                type(keep) is not int or type(remove) is not int
                or not 1 <= keep <= len(existing) + len(interactions.findings)
                or not 1 <= remove <= len(existing)
                or keep == remove or remove in duplicates
            ):
                raise ValueError("некорректные номера повторяющихся дефектов")
            duplicates[remove] = keep
        if duplicates.keys() & duplicates.values():
            raise ValueError("основная карточка дефекта не должна удаляться")
        checks = review.whole.checks.copy()
        for issue in interactions.findings:
            if issue.criterion == "consistent" and len(issue.evidence) < 2:
                raise ValueError("для пересечения правил нужны две разные цитаты")
            checks[issue.criterion] = "fail"
        whole = Assessment(checks, review.whole.findings + interactions.findings)
        return Review(review.requirements, review.individual, whole, tuple(duplicates.items()))

    progress("Перепроверяю пересечения правил и повторы дефектов.")
    return validated_call(
        model, FINAL_REVIEW,
        {
            "document": document.numbered_text(),
            "existing_count": len(existing),
            "existing": [
                {
                    "key": index, "owner": owner, "category": issue.category, "title": issue.title,
                    "explanation": issue.explanation, "severity": issue.severity,
                    "evidence": [
                        {"start": source.start, "end": source.end, "quote": source.quote}
                        for source in issue.evidence
                    ],
                }
                for index, (owner, issue) in enumerate(existing, 1)
            ],
        },
        schema, 16_000, validate, "Сверка дефектов", progress,
    )
