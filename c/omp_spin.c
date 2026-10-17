/* omp_spin.c -- an engine's OpenMP regions on a team that spins (MinGW builds).
 *
 * MinGW-w64's libgomp parks its team on POSIX semaphores (config/posix/bar.c, which never
 * spins: OMP_WAIT_POLICY and GOMP_SPINCOUNT change nothing). At 16 threads a region's fork
 * and join cost ~80 us, a barrier ~60 us, and a region met inside another one -- a team of
 * one -- ~7 us of semaphores and allocations. A MiMo-V2.6-Pro decode step on the RTX 4090
 * runs ~690 regions and ~68000 nested ones (the expert kernel inside moe_wave's tasks):
 * ~90 ms of a ~790 ms step (docs/experiments/mimo-omp-spin-4090-2026-10-17).
 *
 * Linked with --wrap (the Makefile's OMP_SPIN_WRAP), the calls GCC emits for the pragmas
 * the engine uses come here: GOMP_parallel, GOMP_barrier, a dynamic schedule's start, next
 * and end (long and unsigned long long), and the omp_get_thread_num, omp_get_num_threads
 * and omp_in_parallel the code asks. The team is the caller and threads of this file that
 * spin between regions for COLI_OMP_SPIN_US (default 2000) before they sleep. The
 * outlined functions are GCC's own: a static loop takes from omp_get_num_threads() and
 * omp_get_thread_num() the share libgomp would give it, a dynamic one takes chunks as
 * libgomp hands them out, and every iteration runs the same code on the same data -- the
 * numbers do not move. A region met inside a region runs at once on the thread that met
 * it: a team of one, as under libgomp, without building one.
 *
 * libgomp keeps the rest: COLI_OMP_SPIN=0, and a region opened while another thread's runs
 * here (a region inside one of those runs at once too). Not served: single, sections,
 * ordered, tasks, guided and runtime schedules. Inside a region here they would find no
 * libgomp team and run on every thread, so an engine linked with this file uses none of
 * them (tests/test_omp_spin.py checks mimo.c's calls). */
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define SPIN_PAUSE() _mm_pause()
#else
#define SPIN_PAUSE() ((void)0)
#endif

typedef unsigned long long ull;

/* libgomp's own, for everything this file leaves to it */
void __real_GOMP_parallel(void (*fn)(void *), void *data, unsigned num_threads, unsigned flags);
void __real_GOMP_barrier(void);
bool __real_GOMP_loop_nonmonotonic_dynamic_start(long start, long end, long incr, long chunk, long *istart, long *iend);
bool __real_GOMP_loop_nonmonotonic_dynamic_next(long *istart, long *iend);
bool __real_GOMP_loop_ull_nonmonotonic_dynamic_start(bool up, ull start, ull end, ull incr, ull chunk, ull *istart, ull *iend);
bool __real_GOMP_loop_ull_nonmonotonic_dynamic_next(ull *istart, ull *iend);
void __real_GOMP_loop_end_nowait(void);
int __real_omp_get_thread_num(void);
int __real_omp_get_num_threads(void);
int __real_omp_in_parallel(void);

#define SPIN_LINE 64
#define SPIN_WS 8          /* work-sharing loops one region may meet */

/* A dynamic loop's iterations, as libgomp's work share keeps them: the next one, the end
 * (a loop without iterations starts at its end), the chunk times the step. A line each. */
typedef struct {
    _Atomic int claim, ready;
    int up;
    _Atomic long long next;
    long long end, chunk;
    _Atomic ull next_u;
    ull end_u, chunk_u;
} SpinWs;
_Static_assert(sizeof(SpinWs) == SPIN_LINE, "a work share is one cache line");

typedef struct {
    void (*fn)(void *);
    void *data;
    int size;                                 /* threads in the region, the caller included */
    int threads;                              /* workers started */
    _Alignas(SPIN_LINE) _Atomic unsigned gen; /* one more for every region */
    _Alignas(SPIN_LINE) _Atomic int done;     /* workers through the region */
    _Alignas(SPIN_LINE) _Atomic int bar_count;
    _Atomic unsigned bar_gen;
    _Alignas(SPIN_LINE) _Atomic int sleepers;
    pthread_mutex_t mx;
    pthread_cond_t cv;
    _Alignas(SPIN_LINE) SpinWs ws[SPIN_WS];
} SpinTeam;

/* What a thread is running: the team (NULL: a team of one), its number and the team's
 * size, how many regions of this file are around it (0: none) and whether one of them
 * has more than one thread, the loops it met, the one it takes chunks from. */
typedef struct {
    SpinTeam *team;
    int num, size, level, active, ws_k;
    SpinWs *ws;
    SpinWs own;
} SpinThread;

static SpinTeam g_team = { .mx = PTHREAD_MUTEX_INITIALIZER, .cv = PTHREAD_COND_INITIALIZER };
static atomic_flag g_busy = ATOMIC_FLAG_INIT;
static int g_mode = -1;                       /* -1 not read yet, 0 libgomp's, 1 this file's */
static long long g_spin_ns;
static _Thread_local SpinThread t_spin;

static long long spin_now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (long long)t.tv_sec * 1000000000LL + t.tv_nsec;
}

static void spin_die(const char *what) {
    fprintf(stderr, "[OMP] spinning team: %s -- not served (COLI_OMP_SPIN=0: libgomp's team)\n", what);
    abort();
}

/* Until *word leaves `old`: a barrier (the threads are working, the wait is short). */
static void spin_while_eq(_Atomic unsigned *word, unsigned old) {
    for (unsigned i = 0; atomic_load_explicit(word, memory_order_acquire) == old; i++) {
        SPIN_PAUSE();
        if ((i & 0xFFFF) == 0xFFFF) sched_yield();
    }
}

/* A worker between regions: spins COLI_OMP_SPIN_US for the next one, then sleeps. The
 * caller bumps gen, then wakes the sleepers it sees; a worker counts itself a sleeper, then
 * looks at gen again, under the mutex (sequentially consistent: one sees the other). */
static unsigned spin_wait_region(SpinTeam *T, unsigned seen) {
    long long until = 0;
    unsigned g;
    for (unsigned i = 0;; i++) {
        g = atomic_load_explicit(&T->gen, memory_order_acquire);
        if (g != seen) return g;
        SPIN_PAUSE();
        if ((i & 63) == 63) {
            long long now = spin_now_ns();
            if (!until) until = now + g_spin_ns;
            else if (now >= until) break;
        }
    }
    pthread_mutex_lock(&T->mx);
    atomic_fetch_add(&T->sleepers, 1);
    while ((g = atomic_load(&T->gen)) == seen) pthread_cond_wait(&T->cv, &T->mx);
    atomic_fetch_sub(&T->sleepers, 1);
    pthread_mutex_unlock(&T->mx);
    return g;
}

typedef struct { int id; unsigned seen; } SpinStart;

static void *spin_worker(void *arg) {
    SpinStart start = *(SpinStart *)arg;
    free(arg);
    SpinTeam *T = &g_team;
    for (unsigned seen = start.seen;;) {
        seen = spin_wait_region(T, seen);
        if (start.id < T->size) {
            t_spin = (SpinThread){ .team = T, .num = start.id, .size = T->size, .level = 1, .active = 1 };
            T->fn(T->data);
            t_spin = (SpinThread){0};
        }
        /* every worker, in the region or not: the caller changes nothing before all have */
        atomic_fetch_add_explicit(&T->done, 1, memory_order_release);
    }
    return NULL;
}

/* Workers for a team of `size` (between regions: the ones running wait for the next). */
static int spin_grow(SpinTeam *T, int size) {
    while (T->threads < size - 1) {
        SpinStart *s = malloc(sizeof *s);
        pthread_t th;
        if (!s) break;
        s->id = T->threads + 1;
        s->seen = atomic_load_explicit(&T->gen, memory_order_relaxed);
        if (pthread_create(&th, NULL, spin_worker, s)) { free(s); break; }
        pthread_detach(th);
        T->threads++;
    }
    return T->threads + 1 < size ? T->threads + 1 : size;
}

static int spin_mode(void) {
    if (g_mode < 0) {
        const char *v = getenv("COLI_OMP_SPIN"), *us = getenv("COLI_OMP_SPIN_US");
        g_mode = !(v && *v && atoi(v) == 0);
        g_spin_ns = (us && *us ? atoll(us) : 2000) * 1000;
        if (g_spin_ns < 0) g_spin_ns = 0;
        if (g_mode)
            fprintf(stderr, "[OMP] spinning team: %d threads spin %lld us between regions "
                            "(COLI_OMP_SPIN_US; COLI_OMP_SPIN=0: libgomp's team)\n",
                    omp_get_max_threads(), g_spin_ns / 1000);
    }
    return g_mode;
}

void __wrap_GOMP_parallel(void (*fn)(void *), void *data, unsigned num_threads, unsigned flags) {
    SpinThread *me = &t_spin;
    int mode = spin_mode();
    if (me->level > 0 || (mode && omp_get_level() > 0)) {   /* inside a region: a team of one, at once */
        SpinThread outer = *me;
        int active = outer.level > 0 ? outer.active : __real_omp_in_parallel();
        *me = (SpinThread){ .team = NULL, .num = 0, .size = 1, .level = outer.level + 1, .active = active };
        fn(data);
        *me = outer;
        return;
    }
    if (!mode || atomic_flag_test_and_set_explicit(&g_busy, memory_order_acquire)) {
        __real_GOMP_parallel(fn, data, num_threads, flags);
        return;
    }
    SpinTeam *T = &g_team;
    int size = omp_get_max_threads();
    if (num_threads > 0 && (int)num_threads < size) size = (int)num_threads;
    if (size < 1) size = 1;
    size = spin_grow(T, size);
    T->fn = fn; T->data = data; T->size = size;
    for (int k = 0; k < SPIN_WS; k++) {
        atomic_store_explicit(&T->ws[k].claim, 0, memory_order_relaxed);
        atomic_store_explicit(&T->ws[k].ready, 0, memory_order_relaxed);
    }
    atomic_store_explicit(&T->done, 0, memory_order_relaxed);
    atomic_store_explicit(&T->bar_count, 0, memory_order_relaxed);
    atomic_fetch_add(&T->gen, 1);
    if (atomic_load(&T->sleepers) > 0) {
        pthread_mutex_lock(&T->mx);
        pthread_cond_broadcast(&T->cv);
        pthread_mutex_unlock(&T->mx);
    }
    *me = (SpinThread){ .team = T, .num = 0, .size = size, .level = 1, .active = size > 1 };
    fn(data);
    *me = (SpinThread){0};
    for (unsigned i = 0; atomic_load_explicit(&T->done, memory_order_acquire) < T->threads; i++) {
        SPIN_PAUSE();
        if ((i & 0xFFFF) == 0xFFFF) sched_yield();
    }
    atomic_flag_clear_explicit(&g_busy, memory_order_release);
}

void __wrap_GOMP_barrier(void) {
    SpinThread *me = &t_spin;
    if (!me->level) { __real_GOMP_barrier(); return; }
    if (!me->team || me->size < 2) return;
    SpinTeam *T = me->team;
    unsigned g = atomic_load_explicit(&T->bar_gen, memory_order_acquire);
    if (atomic_fetch_add_explicit(&T->bar_count, 1, memory_order_acq_rel) == me->size - 1) {
        atomic_store_explicit(&T->bar_count, 0, memory_order_relaxed);
        atomic_store_explicit(&T->bar_gen, g + 1, memory_order_release);
    } else spin_while_eq(&T->bar_gen, g);
}

/* The next loop of the region: the team's (whoever comes first sets it up, the others wait
 * for it), or a team of one's own. 1 when the caller sets it up. */
static SpinWs *ws_enter(SpinThread *me, int *init) {
    if (!me->team) { me->ws = &me->own; *init = 1; return me->ws; }
    if (me->ws_k >= SPIN_WS) spin_die("more than 8 work-sharing loops in one region");
    SpinWs *w = &me->team->ws[me->ws_k++];
    int expect = 0;
    me->ws = w;
    *init = atomic_compare_exchange_strong(&w->claim, &expect, 1);
    if (!*init)
        for (unsigned i = 0; !atomic_load_explicit(&w->ready, memory_order_acquire); i++) {
            SPIN_PAUSE();
            if ((i & 0xFFFF) == 0xFFFF) sched_yield();
        }
    return w;
}

static bool ws_next(SpinWs *w, long *istart, long *iend) {
    long long end = w->end, start = atomic_load_explicit(&w->next, memory_order_relaxed);
    for (;;) {
        if (start == end) return false;
        long long left = end - start, c = w->chunk, nend;
        if (c < 0 ? c < left : c > left) c = left;
        nend = start + c;
        if (atomic_compare_exchange_weak_explicit(&w->next, &start, nend, memory_order_relaxed, memory_order_relaxed)) {
            *istart = (long)start; *iend = (long)nend;
            return true;
        }
    }
}

static bool ws_next_ull(SpinWs *w, ull *istart, ull *iend) {
    ull end = w->end_u, start = atomic_load_explicit(&w->next_u, memory_order_relaxed);
    for (;;) {
        if (start == end) return false;
        ull left = end - start, c = w->chunk_u, nend;
        if (w->up ? c > left : c < left) c = left;
        nend = start + c;
        if (atomic_compare_exchange_weak_explicit(&w->next_u, &start, nend, memory_order_relaxed, memory_order_relaxed)) {
            *istart = start; *iend = nend;
            return true;
        }
    }
}

bool __wrap_GOMP_loop_nonmonotonic_dynamic_start(long start, long end, long incr, long chunk, long *istart, long *iend) {
    SpinThread *me = &t_spin;
    if (!me->level) return __real_GOMP_loop_nonmonotonic_dynamic_start(start, end, incr, chunk, istart, iend);
    int init;
    SpinWs *w = ws_enter(me, &init);
    if (init) {
        w->end = (incr > 0 && start > end) || (incr < 0 && start < end) ? start : end;
        w->chunk = (long long)chunk * incr;
        atomic_store_explicit(&w->next, start, memory_order_relaxed);
        atomic_store_explicit(&w->ready, 1, memory_order_release);
    }
    return ws_next(w, istart, iend);
}

bool __wrap_GOMP_loop_nonmonotonic_dynamic_next(long *istart, long *iend) {
    SpinThread *me = &t_spin;
    if (!me->level) return __real_GOMP_loop_nonmonotonic_dynamic_next(istart, iend);
    return ws_next(me->ws, istart, iend);
}

bool __wrap_GOMP_loop_ull_nonmonotonic_dynamic_start(bool up, ull start, ull end, ull incr, ull chunk, ull *istart, ull *iend) {
    SpinThread *me = &t_spin;
    if (!me->level) return __real_GOMP_loop_ull_nonmonotonic_dynamic_start(up, start, end, incr, chunk, istart, iend);
    int init;
    SpinWs *w = ws_enter(me, &init);
    if (init) {
        w->up = up;
        w->end_u = (up && start > end) || (!up && start < end) ? start : end;
        w->chunk_u = chunk * incr;
        atomic_store_explicit(&w->next_u, start, memory_order_relaxed);
        atomic_store_explicit(&w->ready, 1, memory_order_release);
    }
    return ws_next_ull(w, istart, iend);
}

bool __wrap_GOMP_loop_ull_nonmonotonic_dynamic_next(ull *istart, ull *iend) {
    SpinThread *me = &t_spin;
    if (!me->level) return __real_GOMP_loop_ull_nonmonotonic_dynamic_next(istart, iend);
    return ws_next_ull(me->ws, istart, iend);
}

void __wrap_GOMP_loop_end_nowait(void) {
    SpinThread *me = &t_spin;
    if (!me->level) { __real_GOMP_loop_end_nowait(); return; }
    me->ws = NULL;
}

int __wrap_omp_get_thread_num(void) {
    SpinThread *me = &t_spin;
    return me->level ? me->num : __real_omp_get_thread_num();
}

int __wrap_omp_get_num_threads(void) {
    SpinThread *me = &t_spin;
    return me->level ? me->size : __real_omp_get_num_threads();
}

int __wrap_omp_in_parallel(void) {
    SpinThread *me = &t_spin;
    return me->level ? me->active : __real_omp_in_parallel();
}
