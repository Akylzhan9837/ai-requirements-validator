"""Дешёвая детерминированная проверка слов-паразитов и заглушек.
Результат НЕ попадает в отчёт напрямую — это подсказки для LLM (он решает, дефект ли это)."""
from __future__ import annotations

import re

WEAK = [
    r"быстр\w*", r"удобн\w*", r"адекватн\w*", r"достаточн\w*", r"эффективн\w*", r"оптимальн\w*", r"гибк\w*",
    r"надёжн\w*", r"надежн\w*", r"простой", r"просто\w*", r"понятн\w*", r"современн\w*", r"некоторы\w+", r"различн\w+",
    r"при необходимости", r"при возможности", r"по возможности", r"как правило", r"обычно", r"примерно", r"около",
    r"и т\.\s?д\.?", r"и т\.\s?п\.?", r"и другие", r"и прочие", r"и\s*/\s*или", r"etc\.?", r"в момент", r"своевременн\w*",
    r"максимально", r"минимально", r"как можно", r"в кратчайшие", r"без задержек", r"мгновенн\w*",
]
PLACEHOLDER = [
    r"\bTBD\b", r"\bTODO\b", r"будет определ\w+", r"заполняются", r"заполнит\w*", r"не заполнен\w*", r"уточняется",
    r"\bN/?A\b", r"Да\s*/\s*Нет", r"\?\?\?",
]

_weak_re = re.compile("|".join(WEAK), re.IGNORECASE)
_ph_re = re.compile("|".join(PLACEHOLDER), re.IGNORECASE)


def lint_text(text: str) -> list[str]:
    hints: list[str] = []
    weak = sorted({m.group(0).strip() for m in _weak_re.finditer(text)}, key=str.lower)
    if weak:
        hints.append("расплывчатые/оценочные слова: " + ", ".join(f"«{w}»" for w in weak))
    ph = sorted({m.group(0).strip() for m in _ph_re.finditer(text)}, key=str.lower)
    if ph:
        hints.append("похоже на заглушку/шаблонный текст: " + ", ".join(f"«{w}»" for w in ph))
    return hints


def lint_document(lines: list[str]) -> list[str]:
    """Подсказки уровня документа: заглушки, пустые ячейки таблиц, отсутствие версии."""
    hints: list[str] = []
    joined = "\n".join(lines)
    for ln_no, ln in enumerate(lines, start=1):
        s = ln.strip()
        if s.startswith("|") and re.fullmatch(r"\|(\s*\|)+", s):
            hints.append(f"строка {ln_no}: таблица с пустыми ячейками")
        elif _ph_re.search(s):
            hints.append(f"строка {ln_no}: возможная заглушка — {s[:90]}")
    if not re.search(r"верси[яи]|version|\bv\d+(\.\d+)*\b|ревизи", joined, re.IGNORECASE):
        hints.append("в тексте не найдено указание версии документа")
    return hints[:25]
