"""Real long-context QA sources with evidence annotations, for KV-management SFT data
(docs/real_data_sft.md). No synthetic data: every paragraph, question and answer comes from the
datasets' official training files.

Each source is normalized to ``QADoc``: a list of paragraphs plus questions, each question with the
indices of the paragraphs that hold its evidence.

| source | file | evidence |
|---|---|---|
| ``hotpot`` | ``hotpot_train_v1.1.json`` (JSON array; distractor context: 2 gold + 8 others) | ``supporting_facts`` titles |
| ``2wiki`` | 2WikiMultihopQA ``train.json`` (same layout as HotpotQA) | ``supporting_facts`` titles |
| ``musique`` | ``musique_ans_v1.0_train.jsonl`` / ``musique_full_v1.0_train.jsonl`` | ``is_supporting`` |
| ``qasper`` | ``qasper-train-v0.3.json`` (one paper, several questions) | ``evidence`` paragraphs |

Questions that can be guessed are dropped here (``guessable``): yes/no answers, and answers that
appear in the question itself (comparison questions "A or B?"). Exclusion against evaluation sets
(LongBench and others) is by normalized question text and by normalized paragraph text
(``Exclusion``).
"""

from __future__ import annotations

import hashlib
import json
import re
import string
from dataclasses import dataclass, field

YES_NO = {"yes", "no"}


@dataclass
class Question:
    qid: str
    question: str
    answers: list[str]
    evidence: list[int]          # paragraph indices in the owning QADoc
    answerable: bool = True


@dataclass
class QADoc:
    source: str
    doc_id: str
    paragraphs: list[tuple[str, str]]  # (title, text)
    questions: list[Question]
    natural_order: bool = False        # Qasper: a paper keeps its reading order
    meta: dict = field(default_factory=dict)


# -- normalization ------------------------------------------------------------------------------
_PUNCT = re.compile(f"[{re.escape(string.punctuation)}]")


def norm(text: str) -> str:
    return " ".join(_PUNCT.sub(" ", text.lower()).split())


def norm_hash(text: str) -> str:
    return hashlib.sha256(norm(text).encode()).hexdigest()[:24]


def guessable(q: Question) -> str | None:
    """Reason a question can be answered without the evidence, or None."""
    answers = [norm(a) for a in q.answers if a.strip()]
    if not answers:
        return "no_answer"
    if any(a in YES_NO for a in answers):
        return "yes_no"
    nq = f" {norm(q.question)} "
    if any(f" {a} " in nq for a in answers):
        return "answer_in_question"
    return None


# -- streaming JSON readers (work on full files and on truncated heads) --------------------------
def iter_json_array(path: str, chunk: int = 1 << 20):
    """Yield the objects of a top-level JSON array without loading the file; stops cleanly at a
    truncated tail (cloud tests read the first megabytes of the official files)."""
    dec = json.JSONDecoder()
    buf, pos, started = "", 0, False
    with open(path, encoding="utf-8") as f:
        eof = False
        while True:
            if not eof and len(buf) - pos < chunk:
                more = f.read(chunk)
                eof = not more
                buf = buf[pos:] + more
                pos = 0
            while pos < len(buf) and buf[pos] in " \r\n\t,":
                pos += 1
            if not started:
                if pos < len(buf) and buf[pos] == "[":
                    started, pos = True, pos + 1
                    continue
                if eof:
                    return
                continue
            if pos < len(buf) and buf[pos] == "]":
                return
            try:
                obj, end = dec.raw_decode(buf, pos)
            except json.JSONDecodeError:
                if eof:
                    return  # truncated tail
                more = f.read(chunk)
                if not more:
                    eof = True
                buf = buf[pos:] + more
                pos = 0
                continue
            yield obj
            pos = end


def iter_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                return  # truncated last line of a head sample


# -- loaders ------------------------------------------------------------------------------------
def _wiki_layout(source: str, row: dict) -> QADoc:
    """HotpotQA and 2WikiMultihopQA: context [[title, [sentences]]], supporting_facts [[title, i]]."""
    paras = [(t, "".join(s).strip()) for t, s in row["context"]]
    gold = {t for t, _ in row["supporting_facts"]}
    evidence = [i for i, (t, _) in enumerate(paras) if t in gold]
    qid = str(row.get("_id") or row.get("id"))
    q = Question(f"{source}:{qid}", row["question"].strip(), [str(row["answer"]).strip()], evidence)
    return QADoc(source, qid, paras, [q], meta={"type": row.get("type") or row.get("level")})


def load_hotpot(path: str):
    for row in iter_json_array(path):
        yield _wiki_layout("hotpot", row)


def load_2wiki(path: str):
    for row in iter_json_array(path):
        yield _wiki_layout("2wiki", row)


def load_musique(path: str):
    for row in iter_jsonl(path):
        paras = [(p["title"], p["paragraph_text"].strip()) for p in row["paragraphs"]]
        evidence = [p["idx"] for p in row["paragraphs"] if p.get("is_supporting")]
        answers = [row.get("answer", "")] + list(row.get("answer_aliases", []))
        q = Question(f"musique:{row['id']}", row["question"].strip(), [a for a in answers if a],
                     evidence, answerable=bool(row.get("answerable", True)))
        yield QADoc("musique", row["id"], paras, [q])


def _qasper_answer(a: dict) -> list[str]:
    if a.get("unanswerable"):
        return []
    if a.get("yes_no") is not None:
        return ["yes" if a["yes_no"] else "no"]
    if a.get("extractive_spans"):
        return [" ".join(a["extractive_spans"])]
    return [a["free_form_answer"]] if a.get("free_form_answer") else []


def load_qasper(path: str):
    data = json.load(open(path, encoding="utf-8"))
    for pid, paper in data.items():
        paras: list[tuple[str, str]] = [("Abstract", paper.get("abstract", "").strip())]
        for sec in paper["full_text"]:
            for p in sec["paragraphs"]:
                if p and p.strip():
                    paras.append((sec["section_name"] or "", p.strip()))
        index = {norm(t): i for i, (_, t) in enumerate(paras)}
        questions = []
        for qa in paper["qas"]:
            for ann in qa["answers"]:  # first annotator whose evidence all maps onto text paragraphs
                a = ann["answer"]
                answers = _qasper_answer(a)
                ev = a.get("evidence") or []
                if not answers or not ev or any(e.startswith("FLOAT SELECTED") for e in ev):
                    continue
                idx = [index.get(norm(e)) for e in ev]
                if any(i is None for i in idx):
                    continue
                questions.append(Question(f"qasper:{qa['question_id']}", qa["question"].strip(), answers,
                                          sorted(set(idx))))
                break
        if questions:
            yield QADoc("qasper", pid, paras, questions, natural_order=True, meta={"title": paper["title"]})


LOADERS = {"hotpot": load_hotpot, "2wiki": load_2wiki, "musique": load_musique, "qasper": load_qasper}


# -- exclusion against evaluation data -----------------------------------------------------------
class Exclusion:
    """Questions and paragraphs that must not appear in training (evaluation sets).

    ``add_longbench_zip`` reads every ``data/*.jsonl`` of the LongBench v1 archive: question
    (``input``) hashes, and paragraph hashes from contexts split into passages / blank-line blocks
    (the leading title line of a "Passage N:" block is dropped)."""

    def __init__(self):
        self.questions: set[str] = set()
        self.paragraphs: set[str] = set()

    def add_question(self, text: str) -> None:
        self.questions.add(norm_hash(text))

    def add_context(self, context: str) -> None:
        for block in re.split(r"Passage \d+:\s*\n", context):
            lines = block.strip().split("\n")
            for part in ["\n".join(lines[1:])] + re.split(r"\n\s*\n", block):
                if len(norm(part)) >= 40:
                    self.paragraphs.add(norm_hash(part))

    def add_longbench_zip(self, path: str) -> None:
        from zipfile import ZipFile

        with ZipFile(path) as z:
            for name in z.namelist():
                if name.startswith("data/") and name.endswith(".jsonl"):
                    for line in z.open(name):
                        row = json.loads(line)
                        self.add_question(row.get("input", ""))
                        self.add_context(row.get("context", ""))

    def add_jsonl(self, path: str, question_key: str = "question", context_key: str | None = None) -> None:
        for row in iter_jsonl(path):
            self.add_question(str(row.get(question_key, "")))
            if context_key and row.get(context_key):
                self.add_context(str(row[context_key]))

    def hits(self, doc: QADoc) -> dict:
        """{'questions': [...qids], 'paragraphs': [...indices]} that collide with evaluation data."""
        return {"questions": [q.qid for q in doc.questions if norm_hash(q.question) in self.questions],
                "paragraphs": [i for i, (_, t) in enumerate(doc.paragraphs) if norm_hash(t) in self.paragraphs]}


def clean(doc: QADoc, exclusion: Exclusion | None = None) -> tuple[QADoc | None, dict]:
    """Drop guessable / unanswerable / evaluation-colliding questions; drop the whole document when
    any of its paragraphs appears in evaluation data (the model must not read evaluation text in
    training, evidence or not). Returns (doc or None, why)."""
    why: dict = {}
    hits = exclusion.hits(doc) if exclusion else {"questions": [], "paragraphs": []}
    if hits["paragraphs"]:
        return None, {"eval_paragraph_doc": len(doc.questions)}
    keep = []
    for q in doc.questions:
        reason = (guessable(q) or ("unanswerable" if not q.answerable else None)
                  or ("no_evidence" if not q.evidence else None)
                  or ("eval_question" if q.qid in hits["questions"] else None))
        if reason:
            why[reason] = why.get(reason, 0) + 1
        else:
            keep.append(q)
    if not keep:
        return None, why
    return QADoc(doc.source, doc.doc_id, doc.paragraphs, keep, doc.natural_order, doc.meta), why
