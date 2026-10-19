/* Sample a running process's threads' instruction pointers (external, no change to the program):
 * winsample3 PID SECONDS INTERVAL_US OUT [MAXTHREADS] [SKIP_EVERY]
 * Threads are sorted by creation time (thread 0 = the main thread). Each tick thread 0 is sampled;
 * the others every SKIP_EVERY-th tick (default 1). OUT: "mod base size path" lines, "thr i tid ideal
 * start prio" lines, then "i rip" samples, then "end i cycles kernel100ns user100ns" lines
 * (deltas over the window). */
#include <windows.h>
#include <tlhelp32.h>
#include <psapi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define STACKQ 512
#define NRA 8
static ULONG64 text_lo, text_hi;
typedef LONG (WINAPI *NtQIT)(HANDLE, ULONG, PVOID, ULONG, PULONG);
typedef struct { DWORD tid; HANDLE h; ULONGLONG created; ULONG64 cyc0; ULONGLONG k0, u0; } Th;

static int cmp(const void *a, const void *b) {
    const Th *x = a, *y = b;
    return x->created < y->created ? -1 : x->created > y->created;
}
static ULONGLONG ft(FILETIME f) { return ((ULONGLONG)f.dwHighDateTime << 32) | f.dwLowDateTime; }

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "winsample3 PID SECONDS INTERVAL_US OUT [MAXTHREADS] [SKIP_EVERY]\n"); return 1; }
    DWORD pid = (DWORD)atoi(argv[1]);
    double secs = atof(argv[2]);
    int us = atoi(argv[3]), maxth = argc > 5 ? atoi(argv[5]) : 1, skip = argc > 6 ? atoi(argv[6]) : 1;
    if (skip < 1) skip = 1;
    NtQIT ntqit = (NtQIT)GetProcAddress(GetModuleHandleA("ntdll.dll"), "NtQueryInformationThread");
    HANDLE proc = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, pid);
    if (!proc) { fprintf(stderr, "OpenProcess %lu\n", GetLastError()); return 1; }
    FILE *f = fopen(argv[4], "w");
    if (!f) { fprintf(stderr, "cannot write %s\n", argv[4]); return 1; }
    HMODULE mods[1024]; DWORD need;
    if (!EnumProcessModulesEx(proc, mods, sizeof mods, &need, LIST_MODULES_ALL)) { fprintf(stderr, "EnumProcessModules %lu\n", GetLastError()); return 1; }
    for (DWORD i = 0; i < need / sizeof(HMODULE) && i < 1024; i++) {
        MODULEINFO mi; char name[MAX_PATH] = "?";
        GetModuleInformation(proc, mods[i], &mi, sizeof mi);
        GetModuleFileNameExA(proc, mods[i], name, sizeof name);
        fprintf(f, "mod %llx %lx %s\n", (unsigned long long)(uintptr_t)mi.lpBaseOfDll, (unsigned long)mi.SizeOfImage, name);
        if (i == 0) { text_lo = (ULONG64)(uintptr_t)mi.lpBaseOfDll + 0x1000; text_hi = text_lo + mi.SizeOfImage; }
    }
    static Th th[1024]; int n = 0;
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    THREADENTRY32 te; te.dwSize = sizeof te;
    for (BOOL ok = Thread32First(snap, &te); ok && n < 1024; ok = Thread32Next(snap, &te)) {
        if (te.th32OwnerProcessID != pid) continue;
        HANDLE h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, te.th32ThreadID);
        if (!h) continue;
        FILETIME c, e, k, u;
        GetThreadTimes(h, &c, &e, &k, &u);
        th[n].tid = te.th32ThreadID; th[n].h = h; th[n].created = ft(c);
        n++;
    }
    CloseHandle(snap);
    qsort(th, n, sizeof *th, cmp);
    if (maxth > n) maxth = n;
    for (int i = 0; i < n; i++) {
        PROCESSOR_NUMBER pn = {0};
        GetThreadIdealProcessorEx(th[i].h, &pn);
        void *start = NULL;
        if (ntqit) ntqit(th[i].h, 9 /* ThreadQuerySetWin32StartAddress */, &start, sizeof start, NULL);
        fprintf(f, "thr %d %lu %u %llx %d\n", i, th[i].tid, (unsigned)pn.Number, (unsigned long long)(uintptr_t)start, GetThreadPriority(th[i].h));
        FILETIME c, e, k, u;
        GetThreadTimes(th[i].h, &c, &e, &k, &u);
        QueryThreadCycleTime(th[i].h, &th[i].cyc0);
        th[i].k0 = ft(k); th[i].u0 = ft(u);
    }
    HANDLE timer = CreateWaitableTimerExW(NULL, NULL, 0x00000002 /* CREATE_WAITABLE_TIMER_HIGH_RESOLUTION */, TIMER_ALL_ACCESS);
    if (!timer) { fprintf(stderr, "timer %lu\n", GetLastError()); return 1; }
    LARGE_INTEGER fr, t0, t; QueryPerformanceFrequency(&fr); QueryPerformanceCounter(&t0);
    long samples = 0, tick = 0;
    double el = 0;
    for (;; tick++) {
        QueryPerformanceCounter(&t);
        el = (double)(t.QuadPart - t0.QuadPart) / fr.QuadPart;
        if (el > secs) break;
        for (int i = 0; i < maxth; i++) {
            if (i && tick % skip) continue;
            if (SuspendThread(th[i].h) == (DWORD)-1) continue;
            CONTEXT ctx; ctx.ContextFlags = CONTEXT_CONTROL;
            if (GetThreadContext(th[i].h, &ctx)) {
                /* RSP and the first exe-text values above it (a heuristic backtrace) */
                ULONG64 st[STACKQ]; SIZE_T got = 0; int nr = 0; ULONG64 ra[NRA];
                if (i == 0 && ReadProcessMemory(proc, (LPCVOID)(uintptr_t)ctx.Rsp, st, sizeof st, &got))
                    for (size_t q = 0; q < got / 8 && nr < NRA; q++)
                        if (st[q] >= text_lo && st[q] < text_hi) ra[nr++] = st[q];
                fprintf(f, "%d %llx %llx", i, (unsigned long long)ctx.Rip, (unsigned long long)ctx.Rsp);
                for (int r = 0; r < nr; r++) fprintf(f, " %llx", (unsigned long long)ra[r]);
                fputc('\n', f); samples++;
            }
            ResumeThread(th[i].h);
        }
        LARGE_INTEGER due; due.QuadPart = -(LONGLONG)us * 10;
        SetWaitableTimer(timer, &due, 0, NULL, NULL, FALSE);
        WaitForSingleObject(timer, INFINITE);
    }
    for (int i = 0; i < n; i++) {
        FILETIME c, e, k, u; ULONG64 cyc = 0;
        GetThreadTimes(th[i].h, &c, &e, &k, &u);
        QueryThreadCycleTime(th[i].h, &cyc);
        fprintf(f, "end %d %llu %llu %llu\n", i, (unsigned long long)(cyc - th[i].cyc0),
                (unsigned long long)(ft(k) - th[i].k0), (unsigned long long)(ft(u) - th[i].u0));
    }
    fprintf(f, "window %.3f ticks %ld\n", el, tick);
    fclose(f);
    fprintf(stderr, "%ld samples of %d thread(s), %ld ticks in %.2fs\n", samples, maxth, tick, el);
    return 0;
}
