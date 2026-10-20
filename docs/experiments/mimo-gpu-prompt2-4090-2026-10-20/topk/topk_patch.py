"""mimo.c with greedy sample() printing each step's top 3 (id, logit) to stderr: topk_patch.py OUT [IN, default c/mimo.c]"""
import sys
s = open(sys.argv[2] if len(sys.argv) > 2 else 'C:/colibri/dev/c/mimo.c', encoding='utf-8', newline='').read()
a = '''static int sample(const float *logits, int vocab, float temperature, float top_p) {
    if (temperature <= 0.0f) return argmax(logits, vocab);'''
assert s.count(a) == 1
s = s.replace(a, '''static int sample(const float *logits, int vocab, float temperature, float top_p) {
    if (temperature <= 0.0f) {
        int t[3] = {-1, -1, -1};
        for (int i = 0; i < vocab; i++) {
            if (t[0] < 0 || logits[i] > logits[t[0]]) { t[2] = t[1]; t[1] = t[0]; t[0] = i; }
            else if (t[1] < 0 || logits[i] > logits[t[1]]) { t[2] = t[1]; t[1] = i; }
            else if (t[2] < 0 || logits[i] > logits[t[2]]) t[2] = i;
        }
        fprintf(stderr, "[topk] %d %.6f %d %.6f %d %.6f\\n", t[0], logits[t[0]], t[1], logits[t[1]], t[2], logits[t[2]]);
        return argmax(logits, vocab);
    }''')
open(sys.argv[1], 'w', encoding='utf-8', newline='').write(s)
print('ok')
