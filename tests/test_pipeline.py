"""Сквозной тест без сети: python -m tests.test_pipeline"""
from pathlib import Path

from reqvalidator.loader import load_document
from reqvalidator.pdf import markdown_to_pdf
from reqvalidator.pipeline import run_pipeline
from reqvalidator.report import build_markdown
from tests.mock_llm import MockLLM


def main():
    doc = load_document("docs/requirements.md")
    llm = MockLLM()
    res = run_pipeline(doc, llm, batch_size=3)
    md = build_markdown(res, model="mock-llm", agent_ver="test", stats=llm.stats, argv="python agent.py test")
    Path("/tmp/test_report.md").write_text(md, encoding="utf-8")
    markdown_to_pdf(md, Path("/tmp/test_report.pdf"))

    ids = [(r.rid, r.doc_id) for r in res.requirements]
    print(ids)
    assert any(d == "Ф-4" for _, d in ids), "дозапрос потерянного Ф-4 не сработал"
    assert not any("нет в документе" in r.quote for r in res.requirements), "галлюцинированное требование не отброшено"
    assert not res.unchecked, "R-002 должен быть проверен повторным запросом"
    assert res.dropped_findings == 1, res.dropped_findings
    f3 = next(r for r in res.requirements if r.doc_id == "Ф-3")
    assert any(f.quote_repaired for f in res.results[f3.rid].findings), "неточная цитата должна быть уточнена по тексту"
    f2 = next(r for r in res.requirements if r.doc_id == "Ф-2")
    assert any(f.quote_repaired for f in res.results[f2.rid].findings), "выдуманная цитата → привязка к самому требованию"
    n1 = res.set_findings[0]
    assert "R-999" not in n1.involved and n1.fid == "N-001"
    assert all(ref in {"N-001"} for p in res.priorities for ref in p["refs"]), res.priorities
    print("OK; notes:", res.notes)


if __name__ == "__main__":
    main()
