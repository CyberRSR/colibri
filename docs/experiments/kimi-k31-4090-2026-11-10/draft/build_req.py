"""build_req.py <wire.jsonl> <out.json> <token budget> [skip turns]: an OpenAI chat request that
replays a Kimi Code session as it stood (system prompt, the tools sent to the model, the turns
that were not undone, each answer with its thinking and tool calls, each tool result), cut at
the last user or tool message inside the budget. `skip turns` drops that many first turns."""
import json, sys
from tokenizers import Tokenizer

wire, out, budget = sys.argv[1], sys.argv[2], int(sys.argv[3])
skip = int(sys.argv[4]) if len(sys.argv) > 4 else 0
tok = Tokenizer.from_file('C:/AI/Kimi-K3.1-Pro/tokenizer.json')
ntok = lambda s: len(tok.encode(s, add_special_tokens=False).ids) if s else 0

system, tools, msgs, turns = '', [], [], []
for line in open(wire, encoding='utf-8'):
    try: o = json.loads(line)
    except Exception: continue
    t = o.get('type')
    if 'systemPrompt' in o and o.get('systemPrompt'): system = o['systemPrompt']
    elif t == 'llm.tools_snapshot': tools = o.get('tools') or tools
    elif t == 'turn.prompt': turns.append(len(msgs))
    elif t == 'context.undone':
        for _ in range(int(o.get('turns') or 0)):
            if turns: del msgs[turns.pop():]
    elif t == 'context.append_message':
        m = o.get('message') or {}
        if m.get('role') == 'user' and 'id' not in m:   # what the client put in beside the user's own words
            text = '\n'.join(c.get('text') or '' for c in m.get('content') or [] if isinstance(c, dict))
            if text: msgs.append({'role': 'user', 'content': text})
    elif t == 'agent.message.appended':
        m = (o.get('message') or {}).get('message') or {}
        role = m.get('role')
        parts = [c for c in m.get('content') or [] if isinstance(c, dict)]
        text = '\n'.join(c.get('text') or '' for c in parts if 'text' in c)
        think = '\n'.join(c.get('think') or '' for c in parts if 'think' in c)
        if role == 'assistant':
            a = {'role': 'assistant', 'content': text}
            if think: a['reasoning_content'] = think
            calls = [{'id': c.get('id'), 'type': 'function', 'function': {'name': c.get('name'), 'arguments': c.get('arguments') or '{}'}}
                     for c in m.get('toolCalls') or []]
            if calls: a['tool_calls'] = calls
            if text or think or calls: msgs.append(a)
        elif role == 'tool':
            msgs.append({'role': 'tool', 'tool_call_id': m.get('toolCallId'), 'content': text})
        elif role == 'user' and text:
            msgs.append({'role': 'user', 'content': text})

if skip and skip < len(turns): msgs = msgs[turns[skip]:]

# every tool call answered, no answer without its call, the user's lines of a row as one message
clean, open_calls = [], {}
for m in msgs:
    if m['role'] == 'tool':
        if m.get('tool_call_id') in open_calls:
            del open_calls[m['tool_call_id']]
            clean.append(m)
        continue
    if open_calls:
        for cid in list(open_calls): clean.append({'role': 'tool', 'tool_call_id': cid, 'content': '[interrupted]'})
        open_calls = {}
    if m['role'] == 'user' and clean and clean[-1]['role'] == 'user':
        clean[-1] = {'role': 'user', 'content': clean[-1]['content'] + '\n\n' + m['content']}
        continue
    clean.append(m)
    for c in m.get('tool_calls') or []: open_calls[c['id']] = 1

tool_defs = [{'type': 'function', 'function': {'name': t.get('name'), 'description': t.get('description') or '',
                                                'parameters': t.get('parameters') or t.get('inputSchema') or {'type': 'object'}}} for t in tools]
used = ntok(system) + ntok(json.dumps(tool_defs, ensure_ascii=False))
base, cut, answer = used, 0, 0
ans_at_cut = 0
for i, m in enumerate(clean):
    n = ntok(m.get('content') or '') + ntok(m.get('reasoning_content') or '') + sum(ntok(c['function']['name'] + c['function']['arguments']) for c in m.get('tool_calls') or []) + 8
    if used + n > budget: break
    used += n
    if m['role'] == 'assistant': answer += n
    nxt = clean[i + 1] if i + 1 < len(clean) else None
    if m['role'] in ('user', 'tool') and not (m['role'] == 'tool' and nxt and nxt['role'] == 'tool'):
        cut, ans_at_cut, used_at_cut = i + 1, answer, used
req = {'model': 'kimi-k3.1-pro', 'messages': [{'role': 'system', 'content': system}] + clean[:cut], 'tools': tool_defs,
       'max_tokens': 1, 'temperature': 0, 'enable_thinking': True, 'stream': False}
json.dump(req, open(out, 'w', encoding='utf-8'), ensure_ascii=False)
print(f'{out}: system+tools ~{base} tokens, {cut} of {len(clean)} messages, ~{used_at_cut if cut else base} tokens in all, ~{ans_at_cut} of them the answers '
      f'({sum(1 for m in clean[:cut] if m["role"] == "assistant")} assistant messages, {len(turns)} turns in the record)')
