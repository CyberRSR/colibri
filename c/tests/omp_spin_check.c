/* omp_spin_check.c -- tests/test_omp_spin.py's driver: the pragmas an engine linked with
 * omp_spin.c uses, each iteration counted, against what OpenMP promises. Prints one line
 * per check and "ok" at the end; exits 1 at the first wrong count. With --two, the checks
 * again on two threads at once, one of them on libgomp's team (the file's fallback). With
 * --time, the cost of an empty region, of a barrier and of a region met inside a region. */
#include <omp.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int failed;
#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL %s:%d ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); failed = 1; } } while (0)

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

/* a kernel's own region, met inside a task (moe_wave's matmul_mxfp4) */
static void inner(int *hits, int n, int *threads_seen, int *in_par) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        hits[i]++;
        if (i == 0) { *threads_seen = omp_get_num_threads(); *in_par = omp_in_parallel(); }
    }
}

static void checks(int rounds) {
    enum { N = 10007 };
    int *hits = calloc(N, sizeof(int));
    int max = omp_get_max_threads();
    for (int r = 0; r < rounds; r++) {
        /* static: every iteration once */
        memset(hits, 0, N * sizeof(int));
        int team = 0, inpar = 0;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            hits[i]++;
            if (i == 0) { team = omp_get_num_threads(); inpar = omp_in_parallel(); }
        }
        for (int i = 0; i < N; i++) CHECK(hits[i] == 1, "static %d: %d", i, hits[i]);
        CHECK(team == max, "static team %d, max %d", team, max);
        CHECK(inpar == (max > 1), "static in_parallel %d", inpar);

        /* dynamic, chunk 1 and 7, int (long entry points) */
        for (int chunk = 1; chunk <= 7; chunk += 6) {
            memset(hits, 0, N * sizeof(int));
            if (chunk == 1) {
                #pragma omp parallel for schedule(dynamic, 1)
                for (int i = 0; i < N; i++) __atomic_fetch_add(&hits[i], 1, __ATOMIC_RELAXED);
            } else {
                #pragma omp parallel for schedule(dynamic, 7)
                for (int i = 3; i < N; i += 2) __atomic_fetch_add(&hits[i], 1, __ATOMIC_RELAXED);
            }
            for (int i = 0; i < N; i++) {
                int want = chunk == 1 ? 1 : (i >= 3 && (i - 3) % 2 == 0);
                CHECK(hits[i] == want, "dynamic chunk %d %d: %d", chunk, i, hits[i]);
            }
        }
        /* dynamic down a step of -3 */
        memset(hits, 0, N * sizeof(int));
        #pragma omp parallel for schedule(dynamic, 5)
        for (int i = N - 1; i >= 10; i -= 3) __atomic_fetch_add(&hits[i], 1, __ATOMIC_RELAXED);
        for (int i = 0; i < N; i++) CHECK(hits[i] == (i >= 10 && (N - 1 - i) % 3 == 0), "down %d: %d", i, hits[i]);
        /* dynamic over int64_t (the unsigned long long entry points), up and down */
        memset(hits, 0, N * sizeof(int));
        #pragma omp parallel for schedule(dynamic, 512)
        for (int64_t i = 0; i < (int64_t)N; i++) __atomic_fetch_add(&hits[i], 1, __ATOMIC_RELAXED);
        for (int i = 0; i < N; i++) CHECK(hits[i] == 1, "int64 %d: %d", i, hits[i]);
        memset(hits, 0, N * sizeof(int));
        #pragma omp parallel for schedule(dynamic, 3)
        for (int64_t i = (int64_t)N - 1; i >= -5; i--) if (i >= 0) __atomic_fetch_add(&hits[i], 1, __ATOMIC_RELAXED);
        for (int i = 0; i < N; i++) CHECK(hits[i] == 1, "int64 down %d: %d", i, hits[i]);
        /* loops without iterations */
        int none = 0;
        #pragma omp parallel for schedule(dynamic, 1)
        for (int i = 5; i < 5; i++) __atomic_fetch_add(&none, 1, __ATOMIC_RELAXED);
        #pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 9; i < 2; i++) __atomic_fetch_add(&none, 1, __ATOMIC_RELAXED);
        CHECK(none == 0, "empty loops ran %d", none);

        /* a region with two loops and barriers between phases (attention's shape) */
        int *a = calloc(N, sizeof(int)), *b = calloc(N, sizeof(int));
        int bad = 0;
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int i = 0; i < N; i++) a[i] = i;
            #pragma omp for schedule(static)
            for (int i = 0; i < N; i++) b[i] = a[N - 1 - i] + 1;
            #pragma omp barrier
            int t = omp_get_thread_num(), n = omp_get_num_threads();
            for (int i = t; i < N; i += n) if (b[i] != N - i) __atomic_fetch_add(&bad, 1, __ATOMIC_RELAXED);
            #pragma omp for schedule(dynamic, 3) nowait
            for (int i = 0; i < N; i++) a[i] = -1;
        }
        for (int i = 0; i < N; i++) CHECK(a[i] == -1, "nowait loop %d", i);
        CHECK(bad == 0, "barrier: %d wrong", bad);
        free(a); free(b);

        /* regions inside tasks: a team of one each, every iteration once */
        enum { T = 64, M = 300 };
        int *tasks = calloc((size_t)T * M, sizeof(int));
        int seen[T], par[T];
        #pragma omp parallel for schedule(dynamic, 1)
        for (int t = 0; t < T; t++) inner(tasks + (size_t)t * M, M, &seen[t], &par[t]);
        for (int t = 0; t < T; t++) {
            CHECK(seen[t] == 1, "nested team %d", seen[t]);
            CHECK(par[t] == (max > 1), "nested in_parallel %d", par[t]);
            for (int i = 0; i < M; i++) CHECK(tasks[(size_t)t * M + i] == 1, "nested %d %d", t, i);
        }
        free(tasks);
        int outside = omp_in_parallel(), tnum = omp_get_thread_num(), tn = omp_get_num_threads();
        CHECK(!outside && tnum == 0 && tn == 1, "outside a region: in_parallel %d num %d threads %d", outside, tnum, tn);
        if (failed) break;
    }
    free(hits);
}

/* a second thread's regions while the main one's run (libgomp's team for one of them) */
static void *other_master(void *arg) {
    (void)arg;
    checks(200);
    return NULL;
}

static void timing(void) {
    static volatile int sink[64];
    (void)sink;
    for (int rep = 0; rep < 3; rep++) {
        int n = 2000;
        double t0 = now_s();
        for (int k = 0; k < n; k++) {
            #pragma omp parallel
            { sink[omp_get_thread_num() & 63] = k; }
        }
        double t1 = now_s();
        for (int k = 0; k < n; k++) {
            #pragma omp parallel
            {
                #pragma omp barrier
                #pragma omp barrier
            }
        }
        double t2 = now_s();
        int *hits = calloc((size_t)16 * n * 34, sizeof(int));
        #pragma omp parallel for schedule(dynamic, 1)
        for (int t = 0; t < 16 * n; t++) inner(hits + (size_t)t * 34, 32, hits + (size_t)t * 34 + 32, hits + (size_t)t * 34 + 33);
        double t3 = now_s();
        free(hits);
        printf("time: %d threads: region %.2f us, two barriers %.2f us more, nested region %.3f us a task\n",
               omp_get_max_threads(), 1e6 * (t1 - t0) / n, 1e6 * ((t2 - t1) - (t1 - t0)) / n, 1e6 * (t3 - t2) / (16.0 * n));
    }
}

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "--time")) { timing(); return 0; }
    checks(300);
    printf("checks: %s\n", failed ? "FAILED" : "pass");
    if (argc > 1 && !strcmp(argv[1], "--two")) {
        pthread_t th;
        pthread_create(&th, NULL, other_master, NULL);
        checks(200);
        pthread_join(th, NULL);
        printf("two masters: %s\n", failed ? "FAILED" : "pass");
    }
    printf("%s\n", failed ? "FAILED" : "ok");
    return failed;
}
