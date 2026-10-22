/* tok_encode against HF's ids on byte strings that may hold NUL: reads "HEX\tID,ID,..."
 * lines, encodes the bytes by their count (as the serve path does), compares, and checks
 * that decoding gives the bytes back. */
#define _GNU_SOURCE
#include "tok.h"

static int hexval(int c) { return c <= '9' ? c - '0' : (c | 32) - 'a' + 10; }

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s tokenizer.json cases.txt\n", argv[0]); return 1; }
    Tok T;
    tok_load(&T, argv[1]);
    FILE *f = fopen(argv[2], "rb");
    if (!f) { perror(argv[2]); return 1; }
    static char line[1 << 20];
    int total = 0, pass = 0, rt = 0;
    while (fgets(line, sizeof line, f)) {
        char *tab = strchr(line, '\t');
        if (!tab) continue;
        *tab = 0;
        int nb = (int)(strlen(line) / 2);
        unsigned char *text = malloc(nb + 1);
        for (int i = 0; i < nb; i++) text[i] = (unsigned char)(hexval(line[2 * i]) * 16 + hexval(line[2 * i + 1]));
        text[nb] = 0;
        static int want[1 << 16], got[1 << 16];
        int nw = 0;
        for (char *q = tab + 1; *q && *q != '\n' && *q != '\r';) {
            want[nw++] = atoi(q);
            while (*q && *q != ',' && *q != '\n') q++;
            if (*q == ',') q++;
        }
        int ng = tok_encode(&T, (const char *)text, nb, got, 1 << 16);
        int ok = ng == nw;
        for (int i = 0; ok && i < ng; i++) ok = got[i] == want[i];
        static char dec[1 << 20];
        int nd = tok_decode(&T, got, ng, dec, sizeof dec - 1);
        int back = nd == nb && !memcmp(dec, text, nb);
        total++; pass += ok; rt += back;
        if (!ok) {
            fprintf(stderr, "MISMATCH case %d (%d bytes)\n  hf (%d):", total, nb, nw);
            for (int i = 0; i < nw && i < 40; i++) fprintf(stderr, " %d", want[i]);
            fprintf(stderr, "\n  c  (%d):", ng);
            for (int i = 0; i < ng && i < 40; i++) fprintf(stderr, " %d", got[i]);
            fprintf(stderr, "\n");
        }
        free(text);
    }
    printf("ids equal to HF: %d/%d, decode gives the bytes back: %d/%d\n", pass, total, rt, total);
    return pass == total && rt == total ? 0 : 2;
}
