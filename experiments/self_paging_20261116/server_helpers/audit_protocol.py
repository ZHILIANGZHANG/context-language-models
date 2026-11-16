"""Post-hoc T1 environment-call audit; does not modify predictions or trajectories."""
from collections import Counter
from self_paging.core.protocol import parse_tool_calls


def audit(record, tok):
    decisions = {d['page']: d for d in record['decisions']}
    predictions = {p: None for p in decisions}
    cursor, pending = 0, None
    counts = Counter()
    errors = []
    for page in record['pages']:
        if page['kind'] != 'assistant':
            continue
        text = tok.decode(record['tokens'][page['start']:page['start'] + page['length']])
        calls, _, bad = parse_tool_calls(text)
        counts['malformed_calls'] += bad
        counts['assistant_turns'] += 1
        counts['tool_calls'] += len(calls)
        for call in calls:
            name, args = call['name'], call['arguments']
            if name in ('kv_in','kv_out','kv_drop','note'):
                counts['memory_calls'] += 1
                continue
            if name not in ('next_events','store'):
                counts['unknown_calls'] += 1
                continue
            counts['environment_calls'] += 1
            error = None
            if name == 'next_events':
                if pending is not None:
                    error = 'read_while_decision_pending'
                elif cursor < len(record['environment_pages']):
                    pending = cursor if cursor in decisions else None
                    cursor += 1
            elif pending is None:
                error = 'store_without_pending_decision'
            else:
                try:
                    shelf = int(args.get('shelf'))
                except (TypeError,ValueError):
                    error = 'noninteger_shelf'
                if error is None:
                    predictions[pending] = shelf
                    pending = None
            if error:
                counts['environment_errors'] += 1
                counts[error] += 1
                errors.append(dict(assistant_page=page['pid'],name=name,arguments=args,error=error))
    assert predictions == {p:d['predicted'] for p,d in decisions.items()}, 'Replay disagrees with archived accepted answers'
    return dict(counts=counts, environment_error_rate=counts['environment_errors']/max(1,counts['environment_calls']),
                errors=errors, replayed_predictions=predictions,
                note='Legacy meta.invalid counts malformed/unknown/no-call turns, not environment state errors.')
