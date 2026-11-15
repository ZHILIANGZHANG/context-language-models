"""T2-A: BrowseComp-Plus environment (docs/experiment_plan.md §3.1, §3.3).

Tools: ``search(query)`` returns ranked snippets, ``get_document(docid)`` returns a document
(truncated), ``answer(text)`` ends the episode. The settings follow ACM's BCP runner [R]
(snippets of 512 tokens for new results and 128 for seen ones, documents up to 8192 tokens), with
ACM's 680/150 split from ``third_party/methods/acm/data/``.

Not runnable in the cloud environment (no corpus, no index, no JDK). The searcher and the judge
are injected, so the episode logic is unit-tested with fakes; on the server use
``PyseriniSearcher`` (BM25 over the BrowseComp-Plus Lucene index) and ``OpenAIJudge`` pointed at
a local OpenAI-compatible server running the judge model (Qwen3.5-9B), with the simple-evals
grader template used by ACM's evaluator.
"""

from __future__ import annotations

import json
import re

SYSTEM = ("You are a research agent. Answer the question by searching a fixed document collection. "
          "Use search to find documents, get_document to read one, and answer when you are confident. "
          "Your memory is paged: every tool result has a header [[PAGE n ...]]; use kv_out to move pages "
          "you no longer need out of memory, kv_in to bring them back, and note to keep short summaries. "
          "The [KV] line shows what is resident.")


def _fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}}


TOOLS = [
    _fn("search", "Search the collection. Returns ranked results with docid and snippet.",
        {"query": {"type": "string"}}, ["query"]),
    _fn("get_document", "Read one document by docid.", {"docid": {"type": "string"}}, ["docid"]),
    _fn("answer", "Give the final answer and end.", {"answer": {"type": "string"}}, ["answer"]),
]

# simple-evals grader template, copied verbatim from ACM (third_party/methods/acm/src/evaluator.py,
# MIT, lixiaochuan2020/agentic-context-management @ f06f90e)
GRADER_TEMPLATE = """\
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available."""


class PyseriniSearcher:
    """BM25 over the BrowseComp-Plus Lucene index (needs pyserini + JDK 21)."""

    def __init__(self, index_path: str, k: int = 5) -> None:
        from pyserini.search.lucene import LuceneSearcher

        self.s, self.k = LuceneSearcher(index_path), k

    def search(self, query: str) -> list[tuple[str, str]]:
        hits = self.s.search(query, k=self.k)
        return [(h.docid, self._text(h.docid)) for h in hits]

    def doc(self, docid: str) -> str | None:
        return self._text(docid)

    def _text(self, docid: str) -> str | None:
        d = self.s.doc(docid)
        if d is None:
            return None
        raw = d.raw() or ""
        try:
            obj = json.loads(raw)
            return obj.get("contents") or obj.get("text") or raw
        except json.JSONDecodeError:
            return raw


class OpenAIJudge:
    """LLM judge over an OpenAI-compatible endpoint (a local SGLang/vLLM server)."""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY") -> None:
        from openai import OpenAI

        self.client, self.model = OpenAI(base_url=base_url, api_key=api_key), model

    def __call__(self, question: str, response: str, correct: str) -> bool:
        msg = GRADER_TEMPLATE.format(question=question, response=response, correct_answer=correct)
        out = self.client.chat.completions.create(model=self.model, temperature=0.0,
                                                  messages=[{"role": "user", "content": msg}])
        text = out.choices[0].message.content or ""
        m = re.search(r"correct:\s*\**\s*(yes|no)", text, re.I)
        return bool(m and m.group(1).lower() == "yes")


class BCPEnv:
    """Same protocol as ``WarehouseEnv``: ``system``, ``task``, ``tools``, ``step``, ``done``, ``score``."""

    def __init__(self, item: dict, searcher, judge, tok, snippet_new: int = 512, snippet_seen: int = 128,
                 doc_max: int = 8192, max_candidates: int = 1) -> None:
        self.item, self.searcher, self.judge, self.tok = item, searcher, judge, tok
        self.snippet_new, self.snippet_seen, self.doc_max = snippet_new, snippet_seen, doc_max
        self.max_candidates = max_candidates
        self.system, self.tools = SYSTEM, TOOLS
        self.task = f"Question: {item['question']}\nStart by calling search."
        self.seen: set[str] = set()
        self.final: str | None = None
        self.done = False
        self.n_pages = 0

    def _cut(self, text: str, n: int) -> str:
        ids = self.tok.encode(text)
        return text if len(ids) <= n else self.tok.decode(ids[:n]) + " …"

    def step(self, name: str, args: dict) -> tuple[str, dict, bool]:
        if name == "search":
            q = str(args.get("query", "")).strip()
            if not q:
                return "ERROR: empty query.", {}, False
            lines = []
            for rank, (docid, text) in enumerate(self.searcher.search(q), 1):
                n = self.snippet_seen if docid in self.seen else self.snippet_new
                self.seen.add(docid)
                lines.append(f"[{rank}] docid={docid}\n{self._cut(text or '', n)}")
            self.n_pages += 1
            return "\n\n".join(lines) or "No results.", {"env_page": self.n_pages - 1}, False
        if name == "get_document":
            docid = str(args.get("docid", ""))
            text = self.searcher.doc(docid)
            if text is None:
                return f"ERROR: no document {docid}.", {}, False
            self.seen.add(docid)
            self.n_pages += 1
            return f"docid={docid}\n{self._cut(text, self.doc_max)}", {"env_page": self.n_pages - 1}, False
        if name == "answer":
            self.final = str(args.get("answer", ""))
            self.done = True
            return "Answer recorded.", {}, True
        return f"ERROR: unknown tool {name}.", {}, False

    def score(self) -> float:
        if self.final is None:
            return 0.0
        # one candidate only: lists of guesses are judged wrong (docs/data_and_training.md §3.2)
        if self.max_candidates == 1 and len(re.split(r"\s+or\s+|;|\n", self.final.strip())) > 1:
            return 0.0
        return float(self.judge(self.item["question"], self.final, self.item["answer"]))


def load_split(path: str) -> list[dict]:
    """ACM's split files: a JSON list of {id, question, answer}."""
    with open(path) as f:
        return json.load(f)
