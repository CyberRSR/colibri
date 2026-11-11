"""api_resume.py PORT OUT.json SYSTEM_FILE GIVEUP_S -- a request whose client gives up after GIVEUP_S seconds
(the socket closed mid-prompt: the gateway cancels it), then the same request again to its end."""
import json, socket, sys, time, urllib.request

port, out, sysfile, giveup = int(sys.argv[1]), sys.argv[2], sys.argv[3], float(sys.argv[4])
base = f'http://127.0.0.1:{port}/v1'
msgs = [{'role': 'system', 'content': open(sysfile, encoding='utf-8').read()},
        {'role': 'user', 'content': 'Which model does the text say the engine was first written for? One short sentence.'}]
body = json.dumps({'model': 'kimi-k3.1-pro', 'messages': msgs, 'max_tokens': 32, 'temperature': 0,
                   'stream': True, 'stream_options': {'include_usage': True}}).encode()


def req():
    return urllib.request.Request(base + '/chat/completions', data=body, headers={'Content-Type': 'application/json'})


t0 = time.time()
while True:
    try:
        urllib.request.urlopen(base + '/models', timeout=5).read(); break
    except Exception:
        if time.time() - t0 > 900: raise SystemExit('the server did not come up')
        time.sleep(1)
res = {}
t0 = time.time()
res['first'] = 'sent by curl, which gave up after GIVEUP_S'
print('request 1:', res['first'], flush=True)
time.sleep(3)
t0 = time.time(); first = None; text = []; usage = None
with urllib.request.urlopen(req(), timeout=7200) as r:
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
            if (piece or d.get('reasoning_content')) and first is None:
                first = time.time() - t0
            text.append(piece)
res['second'] = {'ttft_s': round(first or -1, 2), 'total_s': round(time.time() - t0, 2), 'usage': usage, 'text': ''.join(text)}
print('request 2:', res['second'], flush=True)
json.dump(res, open(out, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
