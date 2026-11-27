"""Regression tests for grounded retries and final defect consolidation."""

import unittest
from unittest.mock import Mock

from reporting import Passport, review_blocks, to_markdown
from review import (
    ITEM_CRITERIA, SET_CRITERIA, Assessment, Finding, Requirement, Review,
    finalize_review, review_batch,
)
from source import Document, Line


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.document = Document(
            "Уведомления", "requirements.md", "test-digest",
            (Line("Сервис отправляет одно уведомление."), Line("После изменения срока отправляются два уведомления.")),
        )
        self.references = [
            {"start": number, "end": number, "quote": line.text}
            for number, line in enumerate(self.document.lines, 1)
        ]
        self.requirements = tuple(
            Requirement(
                f"REQ-{number:03d}", "", "functional", f"Правило {number}",
                self.document.evidence(reference),
            )
            for number, reference in enumerate(self.references, 1)
        )

    def reviewed(self, index, findings=()):
        checks = dict.fromkeys(ITEM_CRITERIA, "pass")
        checks.update({finding["criterion"]: "fail" for finding in findings})
        return {"id": self.requirements[index].id, "checks": checks, "findings": list(findings)}

    def conflict(self, criterion, evidence):
        return {
            "criterion": criterion, "category": "conflict", "severity": "critical",
            "title": "Два несовместимых результата", "explanation": "Оба правила действуют при смене срока.",
            "fix": "Указать приоритет правила.", "evidence": evidence,
        }

    def test_markdown_matching_preserves_source_and_rejects_changed_amount(self):
        document = Document("ТЗ", "spec.md", "digest", (Line("* **Правило:** до 100."), Line("- **Приёмка:** 200.")))
        source = document.evidence({"start": 1, "end": 2, "quote": "Правило: до 100. Приёмка: 200."})
        self.assertEqual(source.quote, "Правило:** до 100. - **Приёмка:** 200.")
        with self.assertRaisesRegex(ValueError, "цитата не совпадает"):
            document.evidence({"start": 1, "end": 2, "quote": "Правило: до 101. Приёмка: 200."})

    def test_retry_keeps_accepted_item_and_rejects_one_sided_conflict(self):
        invalid = self.conflict("correct", self.references[1:])
        valid = self.conflict("correct", self.references)
        model = Mock()
        model.generate.side_effect = [
            {"items": [self.reviewed(0), self.reviewed(1, [invalid])]},
            {"items": [self.reviewed(1, [valid])]},
        ]
        assessments = review_batch(self.document, model, self.requirements, "Проверка", Mock())

        requests = [call.args[1]["targets"] for call in model.generate.call_args_list]
        self.assertEqual([[item["id"] for item in request] for request in requests], [
            ["REQ-001", "REQ-002"], ["REQ-002"],
        ])
        self.assertEqual(len(assessments["REQ-002"].findings[0].evidence), 2)

    def test_unresolved_review_is_visible_in_report(self):
        model = Mock()
        model.generate.return_value = {"items": []}
        group = self.requirements[:1]
        assessments = review_batch(self.document, model, group, "Проверка", Mock())
        whole = Assessment(dict.fromkeys(SET_CRITERIA, "pass"), ())
        markdown = to_markdown(review_blocks(
            self.document, Review(group, assessments, whole), Passport("date", "model", "", "commit", 4)
        ))

        self.assertEqual(model.generate.call_count, 4)
        self.assertIn("Модель не подтвердила оценку", markdown)
        self.assertIn("Готовность не установлена", markdown)

    def test_final_pass_can_replace_weaker_finding_with_grounded_conflict(self):
        early = Finding(
            "correct", "other", "major", "Разница в результате", "Вторая строка отличается.",
            "Уточнить правило.", (self.requirements[1].source,),
        )
        assessment = Assessment(dict.fromkeys(ITEM_CRITERIA, "pass") | {"correct": "fail"}, (early,))
        whole = Assessment(dict.fromkeys(SET_CRITERIA, "pass"), ())
        previous = Review(self.requirements[1:], {"REQ-002": assessment}, whole)

        model = Mock()
        model.generate.return_value = {
            "findings": [self.conflict("consistent", self.references)],
            "duplicates": [{"keep": 2, "remove": 1}],
        }
        confirmed = finalize_review(self.document, model, previous, Mock())
        markdown = to_markdown(review_blocks(
            self.document, confirmed, Passport("date", "model", "", "commit", 1)
        ))
        self.assertIn("Связанный дефект: D-001", markdown)
        self.assertIn("Всего нарушений: 1. Критичных: 1.", markdown)


if __name__ == "__main__":
    unittest.main()
