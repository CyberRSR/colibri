"""api_sess.py PORT OUT.json SYS_A SYS_B PHASE -- two conversations through the API, one between the other's turns.
phase 1: A1 (SYS_A + a question), B1 (SYS_B + a question: another conversation), A2 (A1's history + a question).
phase 2: A3 (A2's history from OUT.json + a question), after the server's restart.
Time to the first token, total, prompt tokens and text of each; the histories kept in OUT.json."""
import json, sys, time, urllib.request

port, out, sysa, sysb, phase = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
base = f'http://127.0.0.1:{port}/v1'
QA = ['Which model does the text say the engine was first written for? One short sentence.',
      'Name two GPU backends the text mentions. One short sentence.',
      'What does the text say streams from disk? One short sentence.']
QB = ['In one short sentence: what is the project called?']


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
A = [{'role': 'system', 'content': open(sysa, encoding='utf-8').read()}]
B = [{'role': 'system', 'content': open(sysb, encoding='utf-8').read()}]
res = {'ready_after_s': round(time.time() - t0, 1), 'turns': []}
if phase == '2':
    prev = json.load(open(out, encoding='utf-8'))
    A, res['turns'] = prev['A'], prev['turns']


def turn(name, conv, q):
    conv.append({'role': 'user', 'content': q})
    r = chat(conv)
    conv.append({'role': 'assistant', 'content': r['text']})
    r['name'] = name
    res['turns'].append(r)
    print(f"{name}: first token {r['ttft_s']} s, total {r['total_s']} s, prompt {r['usage']['prompt_tokens']} tokens: "
          f"{r['text'][:100]!r}", flush=True)


if phase == '1':
    turn('A1', A, QA[0])
    turn('B1', B, QB[0])
    turn('A2', A, QA[1])
else:
    turn('A3', A, QA[2])
res['A'] = A
json.dump(res, open(out, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print('done')
