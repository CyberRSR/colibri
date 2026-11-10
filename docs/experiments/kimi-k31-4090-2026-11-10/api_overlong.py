"""api_overlong.py PORT FILE -- a conversation longer than the served context: the API answers 400
(context_length_exceeded), and the engine keeps serving (a short request right after is answered)."""
import json, sys, time, urllib.request, urllib.error

port, long_file = int(sys.argv[1]), sys.argv[2]
base = f'http://127.0.0.1:{port}/v1'


def post(messages, max_tokens=24):
    body = json.dumps({'model': 'kimi-k3.1-pro', 'messages': messages, 'max_tokens': max_tokens, 'temperature': 0}).encode()
    req = urllib.request.Request(base + '/chat/completions', data=body, headers={'Content-Type': 'application/json'})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            return r.status, json.load(r), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode('utf-8', 'replace') or '{}'), time.time() - t0


t0 = time.time()
while True:
    try:
        urllib.request.urlopen(base + '/models', timeout=5).read(); break
    except Exception:
        if time.time() - t0 > 600: raise SystemExit('the server did not come up')
        time.sleep(1)
print(f'ready after {time.time() - t0:.1f} s')
doc = open(long_file, encoding='utf-8').read()
for name, msgs in (('the over-long conversation', [{'role': 'user', 'content': doc}]),
                   ('the same again', [{'role': 'user', 'content': doc}]),
                   ('a short question after it', [{'role': 'user', 'content': 'What is the capital of France? Answer in one short sentence.'}])):
    code, body, dt = post(msgs)
    text = body.get('choices', [{}])[0].get('message', {}).get('content') if code == 200 else json.dumps(body.get('error', body))[:300]
    print(f'{name}: HTTP {code} in {dt:.1f} s: {text!r}', flush=True)
print('done')
