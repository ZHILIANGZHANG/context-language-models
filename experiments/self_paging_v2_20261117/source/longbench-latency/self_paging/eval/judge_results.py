"""Judge archived BCP/LongMemEval answers and report paired pass@k.

Uses a local Qwen3.5-9B OpenAI-compatible service. All prompts and raw outputs
are retained; malformed output is an error, never an incorrect model answer.
The calibration report lists disagreements for separate human inspection.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

from .long_eval import atomic_json, digest
from ..envs.longmemeval_judge import get_anscheck_prompt
from ..envs.t2_bcp import GRADER_TEMPLATE, parse_judgement


def make_prompt(row):
    if row['dataset'] == 'bcp':
        return GRADER_TEMPLATE.format(question=row['question'], response=row['answer'],
                                      correct_answer=row['reference'])
    return get_anscheck_prompt(row['question_type'], row['question'], row['reference'], row['answer'],
                              abstention='_abs' in row['question_id'])


def parse_answer(dataset, raw):
    if dataset == 'bcp':
        return parse_judgement(raw)
    clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.S).strip().lower()
    match = re.fullmatch(r'\**(yes|no)\**[.!]?', clean)
    if not match:
        raise ValueError('LongMemEval judge did not give a single yes/no')
    return match[1] == 'yes'


def exact_match(a, b):
    # A deliberately transparent calibration comparator, not a semantic scorer.
    return ' '.join(str(a).casefold().split()) == ' '.join(str(b).casefold().split())


def pass_at_k(n, c, k):
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError('Invalid pass@k inputs')
    return 1.0 if n-c < k else 1-math.comb(n-c, k)/math.comb(n, k)


def summarize(rows):
    by = defaultdict(lambda: defaultdict(list))
    seen = set()
    for r in rows:
        if r['reward'] is None:
            raise ValueError('Cannot summarize pending judgements')
        key = (r['dataset'], r['method'], r['budget'])
        unique = key + (r['question_id'], r['sample'])
        if unique in seen:
            raise ValueError('Duplicate episode')
        seen.add(unique)
        by[key][r['question_id']].append(r)
    out = []
    for key, questions in sorted(by.items(), key=lambda x: str(x[0])):
        counts = [len(rs) for rs in questions.values()]
        if len(set(counts)) != 1:
            raise ValueError('Unequal samples per question; incomplete run')
        samples = [tuple(sorted(r['sample'] for r in rs)) for rs in questions.values()]
        if len(set(samples)) != 1:
            raise ValueError('Sample IDs differ between questions')
        n = counts[0]
        correct = [sum(int(r['reward']) for r in rs) for rs in questions.values()]
        out.append(dict(dataset=key[0], method=key[1], budget=key[2], questions=len(questions), samples=n,
                        accuracy=sum(correct)/(len(questions)*n),
                        pass_at_k={str(k): sum(pass_at_k(n,c,k) for c in correct)/len(correct)
                                   for k in sorted({1,n} | ({8} if n>=8 else set()))},
                        scope='observed question set only; completeness checked by campaign manifest'))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', nargs='+', required=True, help='Inference directories')
    ap.add_argument('--out', required=True)
    ap.add_argument('--base-url', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--calibration', type=int, default=200)
    a = ap.parse_args()
    if urlparse(a.base_url).hostname not in ('127.0.0.1', 'localhost', '::1'):
        ap.error('Use a local endpoint (SSH tunnel if necessary), not an external judge')
    from openai import OpenAI
    client = OpenAI(base_url=a.base_url, api_key='EMPTY', timeout=180, max_retries=0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {'model':a.model, 'base_url':a.base_url, 'temperature':0, 'max_tokens':2048,
                'enable_thinking':False, 'source_prompt_sha256':digest(GRADER_TEMPLATE),
                'longmemeval_prompt_file_sha256':digest(Path(__file__).parents[1].joinpath('envs/longmemeval_judge.py').read_text())}
    mp = out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text()) != manifest:
        raise ValueError('Judge settings changed')
    atomic_json(mp, manifest)
    inputs=[]
    for directory in a.input:
        for path in sorted(Path(directory).glob('*.json')):
            row=json.loads(path.read_text())
            if 'judge_status' in row:
                inputs.append((path,row))
    if not inputs:
        raise ValueError('No predictions found')
    judged, calibration = [], []
    for path,row in inputs:
        result=out/(digest([row['settings_sha256'],row['question_id'],row['method'],row['budget'],row['sample']])+'.json')
        input_sha=digest(row)
        if result.exists():
            r=json.loads(result.read_text())
            if r['input_sha256']!=input_sha:raise ValueError('Input prediction changed')
        elif row['judge_status']=='pending':
            prompt=make_prompt(row)
            answer=client.chat.completions.create(model=a.model, temperature=0, max_tokens=2048,
                messages=[{'role':'user','content':prompt}],
                extra_body={'chat_template_kwargs':{'enable_thinking':False}})
            raw=answer.choices[0].message.content or ''
            evidence=dict(input_sha256=input_sha, prompt=prompt, raw_output=raw,
                          finish_reason=answer.choices[0].finish_reason, settings=manifest)
            # Preserve unparseable responses before raising; never silently retry.
            atomic_json(result.with_suffix('.judge-evidence.json'),evidence)
            if answer.choices[0].finish_reason != 'stop':
                raise ValueError('Judge did not terminate normally')
            r=row | {'reward':float(parse_answer(row['dataset'],raw)), 'judge_status':'judged',
                     'input_sha256':input_sha, 'evidence_sha256':digest(evidence)}
            atomic_json(result,r)
        else:
            r=row | {'input_sha256':input_sha}
            atomic_json(result,r)
        judged.append(r)
        if row['judge_status']=='pending':
            calibration.append(dict(question_id=row['question_id'], dataset=row['dataset'], method=row['method'],
                                    sample=row['sample'], response=row['answer'], reference=row['reference'],
                                    exact_match=exact_match(row['answer'],row['reference']),judge=bool(r['reward'])))
    calibration=sorted(calibration,key=lambda x:digest(x))[:a.calibration]
    disagreements=[r for r in calibration if r['exact_match']!=r['judge']]
    atomic_json(out/'calibration.json',dict(n=len(calibration),requested=a.calibration,
        agreement=sum(r['exact_match']==r['judge'] for r in calibration)/len(calibration) if calibration else None,
        records=calibration, disagreements_for_human_review=disagreements[:50],
        human_review_complete=False, note='Exact-match disagreement is not automatically a judge error.'))
    atomic_json(out/'summary.json',summarize(judged))


if __name__=='__main__':main()
