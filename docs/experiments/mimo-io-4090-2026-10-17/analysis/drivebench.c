/* Unbuffered 20 MB reads at random 4 KiB-aligned offsets of the given files, from N threads
 * for S seconds: the GB/s one device gives at that queue depth. Each thread opens its own
 * handles (as the engine's readers do). drivebench SECONDS THREADS FILE... */
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static int nfiles, secs;
static char **files;
static LONG64 total;
static volatile LONG stop;
#define RB (20LL << 20)

static DWORD WINAPI worker(LPVOID arg) {
    unsigned seed = (unsigned)(uintptr_t)arg * 2654435761u + 12345;
    HANDLE *h = calloc(nfiles, sizeof(HANDLE));
    LONG64 *sz = calloc(nfiles, sizeof(LONG64));
    for (int i = 0; i < nfiles; i++) {
        h[i] = CreateFileA(files[i], GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, NULL);
        LARGE_INTEGER s; GetFileSizeEx(h[i], &s); sz[i] = s.QuadPart;
    }
    void *buf = VirtualAlloc(NULL, RB, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    while (!stop) {
        seed = seed * 1103515245u + 12345u;
        int f = (seed >> 8) % nfiles;
        seed = seed * 1103515245u + 12345u;
        LONG64 blocks = (sz[f] - RB) / 4096;
        LONG64 off = ((LONG64)(((uint64_t)seed << 20) ^ (seed >> 3)) % blocks) * 4096;
        OVERLAPPED ov = {0};
        ov.Offset = (DWORD)off; ov.OffsetHigh = (DWORD)(off >> 32);
        DWORD got = 0;
        if (!ReadFile(h[f], buf, (DWORD)RB, &got, &ov)) { fprintf(stderr, "read error %lu\n", GetLastError()); break; }
        InterlockedAdd64(&total, got);
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "drivebench SECONDS THREADS FILE...\n"); return 1; }
    secs = atoi(argv[1]); int n = atoi(argv[2]);
    files = argv + 3; nfiles = argc - 3;
    HANDLE th[64];
    LARGE_INTEGER f, t0, t1; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t0);
    for (int i = 0; i < n; i++) th[i] = CreateThread(NULL, 0, worker, (LPVOID)(uintptr_t)(i + 1), 0, NULL);
    Sleep(secs * 1000);
    stop = 1;
    WaitForMultipleObjects(n, th, TRUE, INFINITE);
    QueryPerformanceCounter(&t1);
    double dt = (double)(t1.QuadPart - t0.QuadPart) / f.QuadPart;
    printf("%d threads: %.2f GB/s (%.1f GB in %.1f s)\n", n, total / dt / 1e9, total / 1e9, dt);
    return 0;
}
