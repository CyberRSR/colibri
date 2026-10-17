/* Per-logical-processor busy % and effective performance % (PDH, English counter names):
 * cpumon SECONDS INTERVAL_MS OUT. OUT: "t inst busy perf" lines. */
#include <windows.h>
#include <pdh.h>
#include <pdhmsg.h>
#include <stdio.h>
#include <stdlib.h>

static void dump(FILE *f, double t, PDH_HCOUNTER c, const char *tag) {
    DWORD sz = 0, n = 0;
    PdhGetFormattedCounterArrayW(c, PDH_FMT_DOUBLE, &sz, &n, NULL);
    PDH_FMT_COUNTERVALUE_ITEM_W *it = malloc(sz);
    if (PdhGetFormattedCounterArrayW(c, PDH_FMT_DOUBLE, &sz, &n, it) == ERROR_SUCCESS)
        for (DWORD i = 0; i < n; i++) fprintf(f, "%.2f %s %ls %.2f\n", t, tag, it[i].szName, it[i].FmtValue.doubleValue);
    free(it);
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "cpumon SECONDS INTERVAL_MS OUT\n"); return 1; }
    double secs = atof(argv[1]); int ms = atoi(argv[2]);
    FILE *f = fopen(argv[3], "w");
    PDH_HQUERY q; PDH_HCOUNTER busy, perf, util;
    if (PdhOpenQueryW(NULL, 0, &q) != ERROR_SUCCESS) return 1;
    PDH_STATUS s1 = PdhAddEnglishCounterW(q, L"\x5cProcessor Information(*)\x5c% Processor Time", 0, &busy);
    PDH_STATUS s2 = PdhAddEnglishCounterW(q, L"\x5cProcessor Information(*)\x5c% Processor Performance", 0, &perf);
    PDH_STATUS s3 = PdhAddEnglishCounterW(q, L"\x5cProcessor Information(*)\x5c% Processor Utility", 0, &util);
    if (s1 || s2 || s3) { fprintf(stderr, "add counter %lx %lx %lx\n", s1, s2, s3); }
    PdhCollectQueryData(q);
    LARGE_INTEGER fr, t0, t; QueryPerformanceFrequency(&fr); QueryPerformanceCounter(&t0);
    for (;;) {
        Sleep(ms);
        PdhCollectQueryData(q);
        QueryPerformanceCounter(&t);
        double el = (double)(t.QuadPart - t0.QuadPart) / fr.QuadPart;
        dump(f, el, busy, "busy"); dump(f, el, perf, "perf"); dump(f, el, util, "util");
        fflush(f);
        if (el > secs) break;
    }
    fclose(f);
    return 0;
}
