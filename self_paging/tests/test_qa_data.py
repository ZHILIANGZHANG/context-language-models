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


def _docs(n, scale=1):
    out = []
    for i in range(n):
        paras = [(f"T{i}-{j}", f"Paragraph {j} of item {i} " + "word " * (scale * (10 + 7 * j))) for j in range(6)]
        out.append(qa_sources.QADoc("hotpot", f"d{i}", paras,
                                    [Question(f"hotpot:{i}", f"Question about item {i} paragraph?", [f"answer{i}"], [1, 4])]))
    return out


class WordTok:
    """Whitespace tokenizer with encode/decode (enough for pagination tests)."""

    def __init__(self):
        self.vocab, self.words = {}, []

    def encode(self, text):
        out = []
        for w in text.replace("\n", " \n ").split(" "):
            if w not in self.vocab:
                self.vocab[w] = len(self.words)
                self.words.append(w)
            out.append(self.vocab[w])
        return out

    def decode(self, ids):
        return " ".join(self.words[i] for i in ids).replace(" \n ", "\n")


def _stream(seed, style, scale=1, **kw):
    rng = random.Random(seed)
    kw = {"page_target": (20 * scale, 40 * scale), "chunk_sizes": (40 * scale,), **kw}  # small documents: small pages
    cfg = qa_streams.StreamConfig(p_chunk=1.0 if style == "chunk" else 0.0, p_no_pressure=0.0, **kw)
    tok = WordTok()
    flat = qa_streams.merge_wiki(rng, _docs(4, scale), cfg)
    s = qa_streams.finalize(rng, qa_streams.paginate(rng, flat, tok, cfg), lambda t: len(tok.encode(t)), cfg, 10 ** 6)
    return s, tok


@pytest.mark.parametrize("style", ["para", "chunk"])
def test_pagination_keeps_evidence_and_timing(style):
    for seed in range(5):
        s, tok = _stream(seed, style)
        assert s["style"] == style and len(s["questions"]) == 4
        for q in s["questions"]:
            i = q["qid"].split(":")[1]
            for j in (1, 4):  # the evidence paragraph's words are on the evidence pages
                marker = f"Paragraph {j} of item {i} "
                assert any(marker.split()[0] in s["pages"][p]["text"] and f"item {i}" in s["pages"][p]["text"]
                           for p in q["evidence_pages"])
            if q["mode"] == "first":
                assert q["ask_after"] < min(q["evidence_pages"]) and q["due_after"] >= max(q["evidence_pages"])
            else:
                assert q["ask_after"] == q["due_after"] >= max(q["evidence_pages"])


def test_chunk_pages_are_lossless_slices():
    s, tok = _stream(1, "chunk", chunk_sizes=(37,))
    words = " ".join(p["text"] for p in s["pages"]).split()
    assert all(len(tok.encode(p["text"])) <= 40 for p in s["pages"][:-1])
    assert words[:6] == " ".join(s["pages"][0]["text"].split()).split()[:6] and len(words) > 100


def test_compose_qasper_keeps_reading_order(files):
    doc, _ = clean(list(qa_sources.load_qasper(files["qasper"]))[0])
    flat = qa_streams.flat_qasper(doc)
    s = qa_streams.paginate(random.Random(0), flat, WordTok(), qa_streams.StreamConfig(p_chunk=0.0))
    joined = " ".join(p["text"] for p in s["pages"])
    assert joined.index("Seeds matter") < joined.index("seed lexicon") < joined.index("Accuracy")


def test_teacher_plan_and_question_targets():
    from self_paging.data.qa_policy import PolicyConfig, simulate

    found = set()
    for seed in range(8):
        s, tok = _stream(seed, "para", scale=12)
        total = sum(s["page_tokens"])
        s["budget"] = PolicyConfig().system_tokens + 24 * len(s["pages"]) + 2 * s["page_cap"] + 400
        s["stream_id"] = str(seed)
        assert s["budget"] < total
        s = qa_streams.schedule(random.Random(seed), s)
        plan = simulate(s)
        assert plan["peak"] <= s["budget"]  # the teacher never exceeds the budget
        evicted = set()
        for e in plan["events"]:
            if e["op"] == "kv_out":
                evicted |= set(e["pages"])
            if e["op"] == "kv_in":  # only evidence that was really paged out comes back
                q = next(q for q in s["questions"] if q["qid"] == e["qid"])
                assert set(e["pages"]) <= evicted and set(e["pages"]) <= set(q["evidence_pages"])
        assert sum(e["op"] == "answer" for e in plan["events"]) == len(s["questions"])
        for q in s["questions"]:
            found.add((q["target_recall"], plan["outcome"][q["qid"]]["recall"]))
    assert (True, True) in found and (False, False) in found
    s, _ = _stream(0, "para")
    s["budget"], s["stream_id"] = None, "free"
    assert simulate(s)["kv_out"] == 0  # no pressure, no page-out


def test_balanced_selection_moves_cells_to_half():
    def fake(i, recall):
        return {"stream_id": str(i), "pressure": True, "budget": 1000, "stream_tokens": 5000, "style": "para",
                "pages": [{}] * 8, "questions": [{"mode": "last", "source": "hotpot", "due_after": 6, "recall": recall}]}

    cands = [fake(i, i % 5 == 0) for i in range(500)]  # 20% recall
    chosen, stats = qa_streams.select_balanced(cands, 200, p_no_pressure=0.0, min_cell=10)
    share = sum(s["questions"][0]["recall"] for s in chosen) / len(chosen)
    assert share > 0.4 and stats["chosen"] == len(chosen)


def test_checks_rank_elsewhere_and_surface_measures():
    s = {"pages": [{"text": "alpha beta"}, {"text": "the museum is in Brindle"}, {"text": "filler " * 50},
                   {"text": "Brindle again"}, {"text": "more filler " * 30}],
         "page_tokens": [10, 10, 300, 10, 200], "budget": 1300}
    q = {"question": "Where is the museum?", "answers": ["Brindle"], "evidence_pages": [1], "mode": "last",
         "ask_after": 4, "due_after": 4}
    k = qa_checks.check_question(s, q)
    assert k["answer_elsewhere"] and k["answer_in_evidence"] and k["bm25_rank"] == 1 and k["evidence_age"] == 3
    assert qa_checks.mutual_information([0, 1, 0, 1], [0, 1, 0, 1]) == 1.0
    assert qa_checks.mutual_information([0, 0, 1, 1], [0, 1, 0, 1]) == 0.0
    streams = []
    for i in range(40):  # recall depends on the mode only: surface lookup must find it
        mode = "first" if i % 2 else "last"
        streams.append({"budget": 1000, "stream_tokens": 3000, "style": "para", "pages": [{}] * 4, "sources": ["hotpot"],
                        "questions": [{"mode": mode, "source": "hotpot", "ask_after": 1, "due_after": 2,
                                       "evidence_pages": [0], "recall": mode == "last"}]})
    sp = qa_checks.surface_predictability(streams)
    assert sp["surface_lookup_accuracy"] == 1.0 and sp["majority_rate"] == 0.5


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
