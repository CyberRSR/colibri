"""api_root.py PORT OUT.json SYSTEM_FILE N -- N new conversations that share one long system message:
each is [system, one user question], sent as a fresh conversation. Time to the first token, total, text."""
import json, sys, time, urllib.request

port, out, sysfile, n = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
base = f'http://127.0.0.1:{port}/v1'
QUESTIONS = ['Which model does the text say the engine was first written for? One short sentence.',
             'Name two GPU backends the text mentions. One short sentence.',
             'What does the text say streams from disk? One short sentence.',
             'In one short sentence: what is the project called?']


def chat(messages, max_tokens=32):
    body = json.dumps({'model': 'kimi-k3.1-pro', 'messages': messages, 'max_tokens': max_tokens, 'temperature': 0,
                       'stream': True, 'stream_options': {'include_usage': True}}).encode()
    req = urllib.request.Request(base + '/chat/completions', data=body, headers={'Content-Type': 'application/json'})
    t0 = time.time(); first = None; text = []; usage = None
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode('utf-8', 'replace').strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                break
            ev = json.loads(data)
            if ev.get('usage'):
                usage = ev['usage']
            for ch in ev.get('choices', []):
                d = ch.get('delta', {})
                piece = d.get('content') or ''
                if (piece or d.get('reasoning_content') or d.get('reasoning')) and first is None:
                    first = time.time() - t0
                text.append(piece)
    return {'ttft_s': round(first or -1, 2), 'total_s': round(time.time() - t0, 2), 'usage': usage, 'text': ''.join(text)}


t0 = time.time()
while True:
    try:
        urllib.request.urlopen(base + '/models', timeout=5).read(); break
    except Exception:
        if time.time() - t0 > 900: raise SystemExit('the server did not come up')
        time.sleep(1)
system = open(sysfile, encoding='utf-8').read()
res = {'ready_after_s': round(time.time() - t0, 1), 'turns': []}
for i in range(n):
    r = chat([{'role': 'system', 'content': system}, {'role': 'user', 'content': QUESTIONS[i % len(QUESTIONS)]}])
    res['turns'].append(r)
    print(f"conversation {i + 1}: first token {r['ttft_s']} s, total {r['total_s']} s, prompt {r['usage']['prompt_tokens']} tokens: {r['text'][:110]!r}", flush=True)
json.dump(res, open(out, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print('done')
