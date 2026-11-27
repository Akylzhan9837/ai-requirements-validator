"""Модели данных. Классы *Out — схемы ответов LLM (без значений по умолчанию,
чтобы они проходили через response_schema Gemini)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Severity = Literal["critical", "major", "minor"]

ReqKind = Literal[
    "functional",      # функциональное требование
    "nonfunctional",   # нефункциональное требование
    "data",            # требование к данным
    "integration",     # требование к интеграции / API
    "interface",       # UI / UX
    "acceptance",      # критерий приёмки
    "business_rule",   # бизнес-правило, цель, ограничение
    "other",
]

ReqCode = Literal[
    "UNAMB", "VERIF", "COMPL", "SINGL", "CONS",
    "FEAS", "IMPL", "NECESS", "TRACE", "CONF",
]

SetCode = Literal[
    "S_COMPL", "S_CONS", "S_TERMS", "S_COVER",
    "S_VERIF", "S_STRUCT", "S_PRIO", "S_FEAS",
]


# ---------------------------------------------------------------- LLM-схемы
class ExtractedReq(BaseModel):
    doc_id: Optional[str] = Field(description="Идентификатор из документа (например 'Ф-1') или null")
    section: str = Field(description="Раздел документа, где находится требование")
    kind: ReqKind
    quote: str = Field(description="ДОСЛОВНАЯ цитата требования из документа")
    summary: str = Field(description="Краткая формулировка, до 15 слов")


class ExtractOut(BaseModel):
    requirements: list[ExtractedReq]


class FindingOut(BaseModel):
    criterion: ReqCode
    severity: Severity
    quote: str = Field(description="ДОСЛОВНЫЙ фрагмент текста, в котором дефект")
    problem: str = Field(description="В чём именно дефект и почему это плохо для разработки/тестирования")
    fix: str = Field(description="Что конкретно нужно уточнить или исправить")


class ReqCheckOut(BaseModel):
    rid: str
    not_applicable: list[ReqCode]
    findings: list[FindingOut]
    rewrite: Optional[str] = Field(description="Улучшенная формулировка требования или null, если замечаний нет")


class CheckBatchOut(BaseModel):
    checks: list[ReqCheckOut]


class SetFindingOut(BaseModel):
    criterion: SetCode
    severity: Severity
    involved: list[str] = Field(description="ID требований из отчёта (R-001 ...), которых касается замечание")
    quotes: list[str] = Field(description="1–3 ДОСЛОВНЫЕ цитаты из документа, на которых основано замечание")
    problem: str
    fix: str


class SetCheckOut(BaseModel):
    findings: list[SetFindingOut]


class PriorityOut(BaseModel):
    title: str = Field(description="Что исправить — коротко, императивом")
    why: str = Field(description="Почему это в первую очередь")
    refs: list[str] = Field(description="ID замечаний (F-001, N-002 ...)")


class FinalizeOut(BaseModel):
    rationale: str = Field(description="2–5 предложений: можно ли начинать разработку и тестирование и почему")
    priorities: list[PriorityOut]


# ------------------------------------------------------- внутренние модели
class Requirement(BaseModel):
    rid: str
    doc_id: Optional[str]
    id_assigned_by_agent: bool
    section: str
    kind: str
    quote: str
    summary: str
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    page: Optional[int] = None
    located: bool = True


class Finding(BaseModel):
    fid: str
    scope: Literal["requirement", "set"]
    criterion: str
    severity: Severity
    rid: Optional[str] = None            # для требования
    involved: list[str] = []             # для набора
    quotes: list[str] = []
    places: list[str] = []               # «строка N, раздел»
    problem: str
    fix: str
    quote_repaired: bool = False


class ReqResult(BaseModel):
    rid: str
    checked: bool = True
    not_applicable: list[str] = []
    findings: list[Finding] = []
    rewrite: Optional[str] = None
