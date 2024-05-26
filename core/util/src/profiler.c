#include <stdio.h>
#include <time.h>

#include <util/profiler.h>

#undef return // profile.h
#undef printf // debug_memory.h

static time_t prev_time;
static int count_marks = 0;

typedef struct ProfilerInfo {
    time_t cur;
    time_t delta;
} profiler_info_t;

profiler_info_t mark_time() {
    time_t time_since = 0;
    time_t now = clock();
    if (count_marks != 0) {
        time_since = now - prev_time;
    }
    count_marks++;

    prev_time = now;
    profiler_info_t info;
    info.cur = now;
    info.delta = time_since;
    return info;
}


void mark_func_time(const char* file, const char* func, int line) {
    profiler_info_t info = mark_time();
    printf("%s:%d [%s] in %llu ms (at %llu s)\n", file, line, func, info.delta, info.cur / 1000);
}

void mark_func_entry_time(const char* file, const char* func, int line, const char* entry) {
    profiler_info_t info = mark_time();
    printf("%s:%d [%s] in %llu ms (at %llu s): %s\n", file, line, func, info.delta, info.cur / 1000, entry);
}

void mark_func_exit(const char* file, const char* func, int line) {
    mark_func_time(file, func, line);
    printf("EXITING %s\n", func);
}