"""api_session.py PORT OUT.json [--long FILE] -- a served Kimi K3.1 Pro session through the OpenAI API:
  1. a short question
  2. a follow-up, the history sent back WITHOUT the reply's reasoning (as agents do)
  3. a long user message (FILE) appended to the same conversation
  4. a short follow-up on top of it (the context past index_topk: decode's picks)
  5. the same request as 4 again (the whole prompt held: nothing to compute but the last tokens)
Each: time to the first token, total time, the tokens' usage, the text. Streamed (SSE)."""
import json, sys, time, urllib.request

port, out = int(sys.argv[1]), sys.argv[2]
long_file = sys.argv[sys.argv.index('--long') + 1] if '--long' in sys.argv else None
max_tokens = int(sys.argv[sys.argv.index('--max-tokens') + 1]) if '--max-tokens' in sys.argv else 48
base = f'http://127.0.0.1:{port}/v1'


def wait_ready(limit=900):
    t0 = time.time()
    while time.time() - t0 < limit:
        try:
            with urllib.request.urlopen(base + '/models', timeout=5) as r:
                return json.load(r), time.time() - t0
        except Exception:
            time.sleep(1)
    raise SystemExit('the server did not come up')


def chat(messages, max_tokens=max_tokens):
    body = json.dumps({'model': 'kimi-k3.1-pro', 'messages': messages, 'max_tokens': max_tokens, 'temperature': 0,
                       'stream': True, 'stream_options': {'include_usage': True}}).encode()
    req = urllib.request.Request(base + '/chat/completions', data=body, headers={'Content-Type': 'application/json'})
    t0 = time.time(); first = None; text = []; reasoning = []; usage = None
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
                rpiece = d.get('reasoning_content') or d.get('reasoning') or ''
                if (piece or rpiece) and first is None:
                    first = time.time() - t0
                text.append(piece); reasoning.append(rpiece)
    return {'ttft_s': round(first or -1, 2), 'total_s': round(time.time() - t0, 2), 'usage': usage,
            'text': ''.join(text), 'reasoning': ''.join(reasoning)}


models, waited = wait_ready()
res = {'ready_after_s': round(waited, 1), 'turns': []}
msgs = [{'role': 'user', 'content': 'What is the capital of France? Answer in one short sentence.'}]
r = chat(msgs); res['turns'].append({'name': 'short question', **r}); print(json.dumps(res['turns'][-1])[:400], flush=True)
msgs += [{'role': 'assistant', 'content': r['text']}, {'role': 'user', 'content': 'And of Italy? One short sentence.'}]
r = chat(msgs); res['turns'].append({'name': 'follow-up, history without reasoning', **r}); print(json.dumps(res['turns'][-1])[:400], flush=True)
if long_file:
    doc = open(long_file, encoding='utf-8').read()
    msgs += [{'role': 'assistant', 'content': r['text']}, {'role': 'user', 'content': doc}]
    r = chat(msgs); res['turns'].append({'name': 'a long message appended', **r}); print(json.dumps(res['turns'][-1])[:400], flush=True)
    msgs += [{'role': 'assistant', 'content': r['text']},
             {'role': 'user', 'content': 'Which model does the document say the engine was first written for? One short sentence.'}]
    r = chat(msgs); res['turns'].append({'name': 'follow-up past index_topk', **r}); print(json.dumps(res['turns'][-1])[:400], flush=True)
    r = chat(msgs); res['turns'].append({'name': 'the same request again', **r}); print(json.dumps(res['turns'][-1])[:400], flush=True)
json.dump(res, open(out, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print('session done')
