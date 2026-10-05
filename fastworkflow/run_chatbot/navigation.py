"""Read-only navigation hierarchy; evidence always retains its source scope."""
from datetime import datetime, timezone
import json


def _key(*parts):
    return json.dumps(parts, separators=(',', ':'))


def _node(kind, label, *identity, **fields):
    return dict(key=_key(kind, *identity), kind=kind, label=label, children=[], **fields)


def _pages(method, **kwargs):
    offset = 0
    while True:
        rows = method(limit=500, offset=offset, **kwargs)
        yield from rows
        if len(rows) < 500:
            return
        offset += len(rows)


def _date(value):
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(timezone.utc).date().isoformat()
    except (ValueError, TypeError, AttributeError):
        return 'Unknown date'


def _experiment_sort_key(row):
    return row.get('created_at') or '', row.get('experiment_id') or ''


def newest_experiments_first(rows):
    """Give every experiment surface the same newest-first chronology."""
    return sorted(rows, key=_experiment_sort_key, reverse=True)


# Turn nodes are selected via /api/turn/<key>; the rail and the conversation
# turn list only read status and started_at from turn.info. Other kinds keep
# full info because showHierarchyInfo dumps every key for the selected node.
_TURN_INFO_KEYS = ('status', 'started_at')


def _slim_turn_info(turn):
    if not turn:
        return {}
    return {key: turn[key] for key in _TURN_INFO_KEYS if key in turn and turn[key] is not None}


def read_source(store, source):
    return {
        'source': source,
        'experiments': list(_pages(store.list_experiments)),
        'conversations': list(_pages(store.list_conversations)),
        'turns': list(_pages(store.list_turns)),
    }


def build_navigation(benchmarks, registrations, sources, warnings=()):
    root = _node('root', 'Benchmarks', info={'warnings': list(warnings)})
    branches = {}
    for row in benchmarks:
        bid = row['benchmark_id']
        node = _node('benchmark', row.get('title') or bid, bid,
                     benchmark_id=bid, info=row)
        branches[bid] = node
        root['children'].append(node)
    adhoc = _node('adhoc', 'ad-hoc conversations', info={'dates': 'UTC'})
    experiments, dates, conversations = {}, {}, {}

    def experiment(row, source, registered=False):
        eid = row['experiment_id']
        key = _key(source, eid)
        if key in experiments:
            return experiments[key]
        bid = row.get('benchmark_id')
        if bid not in branches:
            branches[bid] = _node('benchmark', bid or 'Experiments without a benchmark', bid,
                                  benchmark_id=row.get('benchmark_id'), info={})
            root['children'].append(branches[bid])
        # The node's text is the author's optional description, carried for the
        # detail pane; the rail and the crumbs name an experiment by its id.
        node = _node('experiment', row.get('description') or '',
                     source, eid, experiment_id=eid, source=source, registered=registered,
                     recorded=row.get('status') != 'registered', info=row)
        branches[bid]['children'].append(node)
        experiments[key] = node
        return node

    for row in registrations:
        experiment(dict(row, status='registered'), None, True)

    for spec in sources:
        if 'store' in spec:
            spec = read_source(spec['store'], spec['source'])
        source = spec['source']
        for row in spec['experiments']:
            node = experiment(row, source)
            node['recorded'] = True
            node['info'] = row
            node['label'] = row.get('description') or ''
        conv_rows = spec['conversations']
        conv_index = {(c['channel_id'], c['conversation_id']): c for c in conv_rows}
        turns = spec['turns']
        seen = set()

        def parent_for(row):
            eid = row.get('experiment_id')
            if eid:
                return experiment(row if 'benchmark_id' in row else {'experiment_id': eid}, source)
            date = _date(row.get('started_at'))
            if date not in dates:
                dates[date] = _node('date', date, date, info={'date': date, 'timezone': 'UTC'})
                adhoc['children'].append(dates[date])
            return dates[date]

        def conversation(row, parent):
            cid, channel = row.get('conversation_id'), row.get('channel_id')
            key = _key(parent['key'], source, channel, cid)
            if key not in conversations:
                metadata = conv_index.get((channel, cid), {})
                label = metadata.get('topic') or row.get('task_id') or ('Conversation #' + str(cid) if cid is not None else 'Conversation-less turns')
                conversations[key] = _node('conversation', label, key, source=source,
                                           info=dict(metadata, channel_id=channel, conversation_id=cid,
                                                     task_id=metadata.get('task_id') or row.get('task_id'),
                                                     attempt=metadata.get('attempt') or row.get('attempt')))
                parent['children'].append(conversations[key])
            return conversations[key]

        for turn in reversed(turns):
            conv = conversation(turn, parent_for(turn))
            seen.add((turn.get('channel_id'), turn.get('conversation_id')))
            conv['children'].append(_node('turn', ((turn.get('user_message') or '(no message)')[:100] + ('…' if len(turn.get('user_message') or '') > 100 else '')), source, turn['turn_key'],
                turn_key=turn['turn_key'], source=source, info=_slim_turn_info(turn)))
        for row in conv_rows:
            if (row['channel_id'], row['conversation_id']) in seen:
                continue
            conversation(row, parent_for(row))
    adhoc['children'].sort(key=lambda n: n['label'], reverse=True)
    for branch in branches.values():
        branch['children'] = sorted(
            branch['children'],
            key=lambda node: _experiment_sort_key(
                dict(node.get('info') or {}, experiment_id=node.get('experiment_id'))
            ),
            reverse=True,
        )
    root['children'].append(adhoc)
    return root
