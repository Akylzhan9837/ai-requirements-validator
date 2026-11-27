"""Заглушка LLM для проверки конвейера без сети/ключа. Намеренно отдаёт «грязные» ответы:
выдуманную цитату, неточную цитату, пропуск требования — чтобы проверить защитные механизмы агента."""
from __future__ import annotations

import json
import re

from reqvalidator.llm import LLMStats
from reqvalidator.models import (CheckBatchOut, ExtractOut, ExtractedReq, FinalizeOut, FindingOut, PriorityOut,
                                 ReqCheckOut, SetCheckOut, SetFindingOut)


class MockLLM:
    model = "mock-llm"

    def __init__(self):
        self.stats = LLMStats()
        self.skipped_once = False

    def generate(self, system, prompt, schema, tag=""):
        self.stats.calls += 1
        if schema is ExtractOut:
            doc = prompt.split("ДОКУМЕНТ:\n", 1)[1]
            reqs = []
            for m in re.finditer(r"\[L\d+\] \* \*\*(Ф-\d+)\.\*\* (.+)", doc):
                reqs.append(ExtractedReq(doc_id=m.group(1), section="Функциональные требования", kind="functional",
                                         quote=m.group(2), summary=m.group(2)[:50]))
            if "missing" not in tag:  # имитируем потерю Ф-4 → должен сработать дозапрос
                reqs = [r for r in reqs if r.doc_id != "Ф-4"] if tag == "extract-1" else reqs
            m = re.search(r"\[L\d+\] (Сервис передаёт push на отправку в течение минуты после получения события\.)", doc)
            if m and tag != "extract-missing":
                reqs.append(ExtractedReq(doc_id=None, section="НФТ", kind="nonfunctional", quote=m.group(1), summary="push за минуту"))
            m = re.search(r"\[L\d+\] \| Покупка 31 января в 23:58 на 4 000 ₸ \| (.+?) \| (.+?) \|", doc)
            if m and tag != "extract-missing":
                reqs.append(ExtractedReq(doc_id=None, section="Приёмка", kind="acceptance",
                                         quote="Покупка 31 января в 23:58 на 4 000 ₸ | " + m.group(1), summary="приёмка: граница месяца"))
            reqs.append(ExtractedReq(doc_id=None, section="X", kind="other", quote="Этой фразы нет в документе вообще никогда", summary="галлюцинация"))
            return ExtractOut(requirements=reqs)
        if schema is CheckBatchOut:
            payload = prompt.split("ТРЕБОВАНИЯ ДЛЯ ПРОВЕРКИ (JSON):\n", 1)[1].split("\n\nПОЛНЫЙ ДОКУМЕНТ", 1)[0]
            items = json.loads(payload)
            checks = []
            for it in items:
                if it["rid"] == "R-002" and not self.skipped_once:   # пропуск → повтор
                    self.skipped_once = True
                    continue
                f = []
                if it["doc_id"] == "Ф-3":
                    f.append(FindingOut(criterion="UNAMB", severity="major", quote="максимум один такой push за календарный месяц",
                                        problem="Не определено, что такое «такой push» и в какой тайм-зоне считается календарный месяц.",
                                        fix="Указать тайм-зону и уточнить, считается ли повторное достижение после возврата."))
                    f.append(FindingOut(criterion="VERIF", severity="critical", quote="Неточная цитата про push и порог месяц",
                                        problem="Нет измеримого критерия.", fix="Добавить тест-условия."))
                if it["doc_id"] == "Ф-2":
                    f.append(FindingOut(criterion="COMPL", severity="critical", quote="ЦИТАТА, КОТОРОЙ НЕТ В ДОКУМЕНТЕ",
                                        problem="Не описано поведение при частичном возврате.", fix="Описать."))
                checks.append(ReqCheckOut(rid=it["rid"], not_applicable=["IMPL"] if it["kind"] == "acceptance" else [],
                                          findings=f, rewrite="[уточнить: …]" if f else None))
            return CheckBatchOut(checks=checks)
        if schema is SetCheckOut:
            return SetCheckOut(findings=[
                SetFindingOut(criterion="S_CONS", severity="critical", involved=["R-003", "R-004", "R-999"],
                              quotes=["максимум один такой push за календарный месяц", "следующее достижение вызывает push независимо от истории предыдущей настройки"],
                              problem="Ф-3 ограничивает один push в месяц, Ф-4 разрешает push независимо от истории.", fix="Согласовать."),
                SetFindingOut(criterion="S_STRUCT", severity="minor", involved=[], quotes=["выдуманная цитата без привязки"],
                              problem="Должна быть отброшена.", fix="—")])
        if schema is FinalizeOut:
            return FinalizeOut(rationale="Есть критичные дефекты (F-002, N-001). Начинать нельзя.",
                               priorities=[PriorityOut(title="Устранить противоречие Ф-3/Ф-4", why="Блокирует тесты.", refs=["N-001", "N-777"])])
        raise AssertionError(schema)
