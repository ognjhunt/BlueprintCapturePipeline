"""Bound the inference view while retaining the complete SDK history on disk."""
import json


def compact_authoring_history(items):
    if not isinstance(items, list):
        return items
    rows = [v.model_dump(mode='json') if hasattr(v, 'model_dump') else v for v in items]
    latest_calls = {}
    for index, row in enumerate(rows):
        if isinstance(row, dict) and row.get('type') == 'function_call':
            latest_calls[row.get('name')] = index
    kept_calls = {rows[i]['call_id'] for i in latest_calls.values()}
    user_rows = [i for i, row in enumerate(rows) if isinstance(row, dict) and row.get('role') == 'user']
    # Original evidence plus the latest independent feedback. The latest exact
    # observation, CAD program, Blender program, renders and their results remain.
    keep_users = set(user_rows[:1] + user_rows[-2:])
    assistant_rows = [i for i, row in enumerate(rows) if isinstance(row, dict)
                      and row.get('role') == 'assistant' and row.get('type') != 'function_call']
    result = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            result.append(row)
            continue
        kind = row.get('type')
        if kind == 'reasoning':
            continue
        if kind == 'function_call' and index not in latest_calls.values():
            continue
        if kind == 'function_call_output' and row.get('call_id') not in kept_calls:
            continue
        if row.get('role') == 'user' and index not in keep_users:
            continue
        if index in assistant_rows and index not in assistant_rows[-2:]:
            continue
        result.append(row)
    if len(result) < len(rows):
        result.insert(1, {'role': 'user', 'content': json.dumps({
            'context_note': 'Older authoring turns omitted from this inference view only. Full history and artifacts remain retained. '
                'Latest tool programs, outputs, original evidence and independent feedback follow. '
                'Omitted history is not approval; use tools and satisfy independent validation.'})})
    return result
