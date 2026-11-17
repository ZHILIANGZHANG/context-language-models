"""Real-data QA streams (docs/real_data_sft.md): loaders on the official formats, filters, evaluation
exclusion, stream composition, model-free checks and the necessity check. The fixtures below are tiny
hand-written records in the datasets' formats, for testing the code only."""

import json
import random
import zipfile

import pytest

from self_paging.data import qa_checks, qa_sources, qa_streams
from self_paging.data.qa_sources import Exclusion, Question, clean, guessable

HOTPOT = [{"_id": "h1", "question": "Which city hosts the museum founded by Ada Lark?", "answer": "Brindle",
           "supporting_facts": [["Ada Lark", 0], ["Lark Museum", 1]],
           "context": [["Ada Lark", ["Ada Lark founded the Lark Museum."]],
                       ["Lark Museum", ["The Lark Museum is a museum.", " It is in Brindle."]],
                       ["Other", ["Unrelated text about rivers, hills and the long northern valleys."]]]},
          {"_id": "h2", "question": "Is Brindle larger than Cobb?", "answer": "yes",
           "supporting_facts": [["Brindle", 0]], "context": [["Brindle", ["Brindle is big."]]]}]
MUSIQUE = {"id": "2hop__1", "question": "Who painted the mural in the station named after Tor Vik?",
           "answer": "Mira Holt", "answer_aliases": ["M. Holt"], "answerable": True,
           "paragraphs": [{"idx": 0, "title": "Vik Station", "paragraph_text": "Vik Station is named after Tor Vik.",
                           "is_supporting": True},
                          {"idx": 1, "title": "Mural", "paragraph_text": "The Vik Station mural was painted by Mira Holt.",
                           "is_supporting": True},
                          {"idx": 2, "title": "Noise", "paragraph_text": "A different station opened in 1990.",
                           "is_supporting": False}]}
QASPER = {"p1": {"title": "A Paper", "abstract": "We study seeds.", "figures_and_tables": [],
                 "full_text": [{"section_name": "Intro", "paragraphs": ["Seeds matter.", "The seed lexicon has 15 words."]},
                               {"section_name": "Results", "paragraphs": ["Accuracy is 81 percent."]}],
                 "qas": [{"question": "How big is the seed lexicon?", "question_id": "q1",
                          "answers": [{"answer": {"unanswerable": False, "extractive_spans": ["15 words"], "yes_no": None,
                                                  "free_form_answer": "", "evidence": ["The seed lexicon has 15 words."],
                                                  "highlighted_evidence": []}}]},
                         {"question": "Is it good?", "question_id": "q2",
                          "answers": [{"answer": {"unanswerable": False, "extractive_spans": [], "yes_no": True,
                                                  "free_form_answer": "", "evidence": ["Accuracy is 81 percent."],
                                                  "highlighted_evidence": []}}]}]}}


@pytest.fixture
def files(tmp_path):
    h = tmp_path / "hotpot.json"
    h.write_text(json.dumps(HOTPOT))
    m = tmp_path / "musique.jsonl"
    m.write_text(json.dumps(MUSIQUE) + "\n" + json.dumps(MUSIQUE)[:40])  # truncated last line (head sample)
    q = tmp_path / "qasper.json"
    q.write_text(json.dumps(QASPER))
    return {"hotpot": str(h), "musique": str(m), "qasper": str(q)}


def test_loaders_read_evidence(files, tmp_path):
    h = list(qa_sources.load_hotpot(files["hotpot"]))
    assert h[0].questions[0].evidence == [0, 1] and h[0].paragraphs[1][1] == "The Lark Museum is a museum. It is in Brindle."
    m = list(qa_sources.load_musique(files["musique"]))
    assert len(m) == 1 and m[0].questions[0].evidence == [0, 1] and "M. Holt" in m[0].questions[0].answers
    qp = list(qa_sources.load_qasper(files["qasper"]))[0]
    assert qp.natural_order and qp.paragraphs[0][0] == "Abstract"
    assert {q.qid: q.evidence for q in qp.questions} == {"qasper:q1": [2], "qasper:q2": [3]}
    # a truncated JSON array (the head of a big file) yields the complete objects only
    head = tmp_path / "head.json"
    text = json.dumps(HOTPOT)
    head.write_text(text[: text.index('{"_id": "h2"') + 30])
    assert [d.doc_id for d in qa_sources.load_hotpot(str(head))] == ["h1"]


def test_guessable_and_exclusion(files, tmp_path):
    assert guessable(Question("x", "Is it big?", ["yes"], [0])) == "yes_no"
    assert guessable(Question("x", "Which came first, Alpha or Beta?", ["Beta"], [0])) == "answer_in_question"
    assert guessable(Question("x", "Who painted it?", ["Mira Holt"], [0])) is None
    docs = list(qa_sources.load_hotpot(files["hotpot"]))
    assert clean(docs[1])[0] is None  # yes/no only
    lb = tmp_path / "data.zip"
    with zipfile.ZipFile(lb, "w") as z:
        z.writestr("data/hotpotqa.jsonl", json.dumps({"input": "Who painted the mural in the station named after Tor Vik?",
                                                      "context": "Passage 1:\nOther\nUnrelated text about rivers, hills and the long northern valleys."}) + "\n")
    ex = Exclusion()
    ex.add_longbench_zip(str(lb))
    d, why = clean(docs[0], ex)
    assert d is None and why == {"eval_paragraph_doc": 1}  # shares a paragraph with LongBench
    m = list(qa_sources.load_musique(files["musique"]))[0]
    d, why = clean(m, ex)
    assert d is None and why == {"eval_question": 1}


def _docs(n):
    out = []
    for i in range(n):
        paras = [(f"T{i}-{j}", f"Paragraph {j} of item {i} " + "word " * (10 + 7 * j)) for j in range(6)]
        out.append(qa_sources.QADoc("hotpot", f"d{i}", paras,
                                    [Question(f"hotpot:{i}", f"Question about item {i}?", [f"answer{i}"], [1, 4])]))
    return out


def test_compose_wiki_keeps_evidence_and_timing():
    cfg = qa_streams.StreamConfig(items=(3, 3), max_tokens=10 ** 6, p_no_pressure=0.0)
    count = lambda t: len(t.split())  # noqa: E731
    streams = qa_streams.build({"hotpot": _docs(30)}, 10, count, cfg, seed=1)
    assert len(streams) == 10
    for s in streams:
        assert len(s["questions"]) == 3 and s["page_tokens"] == [count(p["text"]) for p in s["pages"]]
        for q in s["questions"]:
            i = q["qid"].split(":")[1]
            for j in (1, 4):  # the evidence paragraphs are on the evidence pages
                assert any(f"Paragraph {j} of item {i} " in s["pages"][p]["text"] for p in q["evidence_pages"])
            if q["mode"] == "first":
                assert q["ask_after"] < min(q["evidence_pages"])
            else:
                assert q["ask_after"] >= max(q["evidence_pages"])
        assert s["budget"] is None or s["budget"] >= 2 * s["page_cap"]
    assert {q["mode"] for s in streams for q in s["questions"]} == {"first", "last"}
    assert len({len(s["pages"]) for s in streams}) > 1  # 1-3 paragraphs per page


def test_compose_qasper_keeps_reading_order_and_length_cap(files):
    doc = list(qa_sources.load_qasper(files["qasper"]))[0]
    d, _ = clean(doc)
    s = qa_streams.compose_qasper(random.Random(0), d, qa_streams.StreamConfig())
    joined = " ".join(p["text"] for p in s["pages"])
    assert joined.index("Seeds matter") < joined.index("seed lexicon") < joined.index("Accuracy")
    cfg = qa_streams.StreamConfig(max_tokens=6)
    assert qa_streams.finalize(random.Random(0), s, lambda t: len(t.split()), cfg) is None  # evidence cut off


def test_checks_recall_rank_and_elsewhere():
    s = {"pages": [{"text": "alpha beta"}, {"text": "the museum is in Brindle"}, {"text": "filler " * 50},
                   {"text": "Brindle again"}, {"text": "more filler " * 30}],
         "page_tokens": [10, 10, 300, 10, 200], "budget": 1300}
    q = {"question": "Where is the museum?", "answers": ["Brindle"], "evidence_pages": [1], "mode": "last", "ask_after": 4}
    k = qa_checks.check_question(s, q, system_tokens=700, structure_per_page=40)
    assert k["answer_elsewhere"] and k["answer_in_evidence"] and k["bm25_rank"] == 1
    assert k["recall_needed"]  # room 1300-700-200 = 400 keeps only the last two pages
    s["budget"] = 2000
    assert not qa_checks.check_question(s, q)["recall_needed"]
    s["budget"] = None
    assert not qa_checks.check_question(s, q)["recall_needed"]
    rep = qa_checks.report([qa_checks.annotate({**s, "budget": 1300, "kind": "wiki", "sources": ["hotpot"],
                                                "page_cap": 300, "stream_tokens": 530,
                                                "questions": [dict(q, qid="a", source="hotpot")]})])
    assert rep["share"]["recall_needed"] == 1.0 and rep["questions"] == 1


def test_necessity_scoring_and_run():
    pytest.importorskip("torch")
    from self_paging.eval import qa_necessity
    from self_paging.train.common import load_tokenizer, tiny_qwen3

    assert qa_necessity.f1("The Brindle", ["Brindle"]) == 1.0 and qa_necessity.f1("no idea", ["Brindle"]) == 0.0
    tok = load_tokenizer("toy")
    model = tiny_qwen3(tok.vocab_size).eval()
    s = {"stream_id": "s", "pages": [{"text": "the museum is in Brindle"}, {"text": "filler"}],
         "questions": [{"qid": "a", "source": "hotpot", "question": "Where?", "answers": ["Brindle"],
                        "evidence_pages": [0], "mode": "last", "ask_after": 1}]}
    v = qa_necessity.views(s, s["questions"][0])
    assert v["closed"] == [] and v["without"] == ["filler"] and len(v["with"]) == 2
    rows = qa_necessity.check(model, tok, s, max_new=3)
    assert len(rows) == 1 and set(rows[0]["f1"]) == {"closed", "without", "with"} and rows[0]["necessary"] is False
