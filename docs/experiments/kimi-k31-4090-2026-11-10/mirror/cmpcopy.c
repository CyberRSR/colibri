/* cmpcopy.c <dirA> <dirB> [must-contain] [must-not-contain]: every *.safetensors of dirA (whose name
 * contains / lacks the given text) against the file of the same name in dirB, byte for byte.
 * Unbuffered reads of both files at once (overlapped), 4 files in parallel. Exit 0 = all equal. */
#include <windows.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#define CH (32u << 20)
#define NW 4
static char names[1024][MAX_PATH]; static int nn, next_i;
static const char *dirA, *dirB;
static CRITICAL_SECTION cs;
static volatile LONG64 total_bytes; static volatile LONG bad, missing;

static DWORD WINAPI worker(LPVOID arg) {
    (void)arg;
    uint8_t *a = VirtualAlloc(NULL, CH, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    uint8_t *b = VirtualAlloc(NULL, CH, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    OVERLAPPED oa, ob; HANDLE ea = CreateEventA(NULL, TRUE, FALSE, NULL), eb = CreateEventA(NULL, TRUE, FALSE, NULL);
    for (;;) {
        EnterCriticalSection(&cs); int i = next_i++; LeaveCriticalSection(&cs);
        if (i >= nn) break;
        char pa[2 * MAX_PATH], pb[2 * MAX_PATH];
        snprintf(pa, sizeof pa, "%s\\%s", dirA, names[i]); snprintf(pb, sizeof pb, "%s\\%s", dirB, names[i]);
        HANDLE ha = CreateFileA(pa, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, NULL);
        HANDLE hb = CreateFileA(pb, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, NULL);
        if (ha == INVALID_HANDLE_VALUE || hb == INVALID_HANDLE_VALUE) {
            printf("MISSING %s\n", names[i]); InterlockedIncrement(&missing);
            if (ha != INVALID_HANDLE_VALUE) CloseHandle(ha);
            if (hb != INVALID_HANDLE_VALUE) CloseHandle(hb);
            continue;
        }
        LARGE_INTEGER sa, sb; GetFileSizeEx(ha, &sa); GetFileSizeEx(hb, &sb);
        int ok = sa.QuadPart == sb.QuadPart; long long where = -1;
        for (long long off = 0; ok && off < sa.QuadPart; off += CH) {
            ZeroMemory(&oa, sizeof oa); ZeroMemory(&ob, sizeof ob);
            oa.hEvent = ea; ob.hEvent = eb; ResetEvent(ea); ResetEvent(eb);
            oa.Offset = ob.Offset = (DWORD)off; oa.OffsetHigh = ob.OffsetHigh = (DWORD)(off >> 32);
            DWORD ga = 0, gb = 0;
            if (!ReadFile(ha, a, CH, NULL, &oa) && GetLastError() != ERROR_IO_PENDING) { ok = 0; where = off; break; }
            if (!ReadFile(hb, b, CH, NULL, &ob) && GetLastError() != ERROR_IO_PENDING) { ok = 0; where = off; break; }
            if (!GetOverlappedResult(ha, &oa, &ga, TRUE) || !GetOverlappedResult(hb, &ob, &gb, TRUE) || ga != gb) { ok = 0; where = off; break; }
            if (memcmp(a, b, ga)) {
                for (DWORD k = 0; k < ga; k++) if (a[k] != b[k]) { where = off + k; break; }
                ok = 0; break;
            }
            InterlockedAdd64(&total_bytes, ga);
        }
        CloseHandle(ha); CloseHandle(hb);
        if (!ok) { InterlockedIncrement(&bad); printf("DIFF %s (sizes %lld / %lld, first difference at %lld)\n", names[i], sa.QuadPart, sb.QuadPart, where); }
        else printf("same %s (%lld bytes)\n", names[i], sa.QuadPart);
        fflush(stdout);
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: cmpcopy dirA dirB [must-contain] [must-not-contain]\n"); return 2; }
    dirA = argv[1]; dirB = argv[2];
    const char *inc = argc > 3 && *argv[3] ? argv[3] : NULL, *exc = argc > 4 && *argv[4] ? argv[4] : NULL;
    char pat[MAX_PATH]; snprintf(pat, sizeof pat, "%s\\*.safetensors", dirA);
    WIN32_FIND_DATAA fd; HANDLE f = FindFirstFileA(pat, &fd);
    if (f == INVALID_HANDLE_VALUE) { fprintf(stderr, "nothing in %s\n", pat); return 2; }
    do {
        if (inc && !strstr(fd.cFileName, inc)) continue;
        if (exc && strstr(fd.cFileName, exc)) continue;
        if (nn < 1024) snprintf(names[nn++], MAX_PATH, "%s", fd.cFileName);
    } while (FindNextFileA(f, &fd));
    FindClose(f);
    InitializeCriticalSection(&cs);
    LARGE_INTEGER q0, q1, fr; QueryPerformanceFrequency(&fr); QueryPerformanceCounter(&q0);
    HANDLE th[NW];
    for (int i = 0; i < NW; i++) th[i] = CreateThread(NULL, 0, worker, NULL, 0, NULL);
    WaitForMultipleObjects(NW, th, TRUE, INFINITE);
    QueryPerformanceCounter(&q1);
    double s = (double)(q1.QuadPart - q0.QuadPart) / fr.QuadPart;
    printf("%d files, %d differ, %d missing; %.1f GB compared in %.0f s (%.1f GB/s of each drive)\n",
           nn, (int)bad, (int)missing, total_bytes / 1e9, s, total_bytes / 1e9 / s);
    return bad || missing ? 1 : 0;
}
