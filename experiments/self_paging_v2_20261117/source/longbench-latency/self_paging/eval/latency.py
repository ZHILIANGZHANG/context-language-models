"""Per-example timings for LongBench. All time fields use milliseconds."""

from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter_ns, time

import torch

from ..hf.logstore import PagedKV


class Latency:
    def __init__(self, device):
        self.device = torch.device(device)
        self.phases = defaultdict(list)
        self.turns = []
        self.submitted_at = None
        self.finished_at = None
        self.sync()
        self.started_at = perf_counter_ns()
        self.started_at_epoch = time()

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def elapsed(self):
        return (perf_counter_ns() - self.started_at) / 1e6

    @contextmanager
    def phase(self, name, *, device_events=False):
        events = None
        if device_events and self.device.type == 'cuda':
            events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            events[0].record(torch.cuda.current_stream(self.device))
        started = perf_counter_ns()
        try:
            yield
        finally:
            wall = (perf_counter_ns() - started) / 1e6
            if events:
                events[1].record(torch.cuda.current_stream(self.device))
            self.phases[name].append((wall, events))

    def feed(self, cache, pid, ids):
        kind = cache.table.pages[pid].kind
        name = 'document_read' if kind == 'tool' else 'other_prefill'
        with self.phase(name, device_events=kind == 'tool'):
            return cache.feed(pid, ids)

    def begin_decode(self):
        self.turns.append(dict(start_ms=self.elapsed(), first_token_ms=None, end_ms=None,
                               tokens=0, submitted=False))

    def first_token(self):
        turn = self.turns[-1]
        if turn['first_token_ms'] is None:
            turn['first_token_ms'] = self.elapsed()

    def end_decode(self, tokens):
        self.turns[-1].update(end_ms=self.elapsed(), tokens=tokens)

    def submit(self):
        if self.submitted_at is None:
            self.submitted_at = self.elapsed()
            self.turns[-1]['submitted'] = True

    def finish(self):
        self.sync()
        self.finished_at = self.elapsed()

    def record(self):
        if self.finished_at is None:
            raise RuntimeError('Evaluation timer was not stopped')
        phases = {}
        for name, values in self.phases.items():
            event_times = [events[0].elapsed_time(events[1]) for _, events in values if events]
            phases[name] = dict(calls=len(values), host_ms=sum(wall for wall, _ in values),
                                cuda_stream_ms=sum(event_times) if event_times else None)
        turns = [t for t in self.turns if t['end_ms'] is not None]
        submitted = next((t for t in turns if t['submitted']), None)
        tokens = sum(t['tokens'] for t in turns)
        generation = sum(t['end_ms'] - t['start_ms'] for t in turns)
        post_first_ms = sum(t['end_ms'] - t['first_token_ms'] for t in turns if t['tokens'] > 1)
        post_first_tokens = sum(max(0, t['tokens'] - 1) for t in turns)
        answer_first_to_end = (submitted['end_ms'] - submitted['first_token_ms']
                               if submitted and submitted['tokens'] > 1 else None)
        return dict(total_ms=self.finished_at, submit_ms=self.submitted_at,
                    first_generated_token_ms=next((t['first_token_ms'] for t in turns if t['tokens']), None),
                    answer_turn_first_token_ms=submitted['first_token_ms'] if submitted else None,
                    answer_turn_generation_ms=(submitted['end_ms'] - submitted['start_ms']) if submitted else None,
                    answer_turn_ms_per_token=(answer_first_to_end / (submitted['tokens'] - 1)
                                              if answer_first_to_end is not None else None),
                    generation_wall_ms=generation, decode_after_first_ms=post_first_ms,
                    control_turn_generation_ms=sum(t['end_ms'] - t['start_ms'] for t in turns
                                                   if not t['submitted']),
                    generated_tokens=tokens, control_turn_tokens=sum(t['tokens'] for t in turns
                                                                                     if not t['submitted']),
                    decode_ms_per_token=post_first_ms / post_first_tokens if post_first_tokens else None,
                    phases=phases, turns=turns)


class TimedPagedKV(PagedKV):
    def __init__(self, *args, latency, **kwargs):
        super().__init__(*args, **kwargs)
        self.latency = latency

    def page_out(self, pids, physical=False):
        with self.latency.phase('page_out_request'):
            return super().page_out(pids, physical=physical)

    def page_in(self, pids):
        with self.latency.phase('page_in_request'):
            return super().page_in(pids)

    def _gather(self, pid):
        with self.latency.phase('kv_cache_rebuild', device_events=True):
            return super()._gather(pid)
