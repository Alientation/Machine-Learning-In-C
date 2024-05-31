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
    return (profiler_info_t) {.cur = now, .delta = time_since};
}


void mark_func_time(const char* file, const char* func, int line) {
    profiler_info_t info = mark_time();
    printf("%s:%d [%s] in %llu ms (at %.4f s)\n", file, line, func, info.delta, info.cur / 1000.0);
    mark_time(); // dont count the time of printing
}

void mark_func_entry_time(const char* file, const char* func, int line, const char* entry) {
    profiler_info_t info = mark_time();
    printf("%s:%d [%s] in %llu ms (at %.4f s): %s\n", file, line, func, info.delta, info.cur / 1000.0, entry);
    mark_time(); // dont count the time of printing
}

void mark_func_exit(const char* file, const char* func, int line) {
    profiler_info_t info = mark_time();
    printf("%s:%d [%s] in %llu ms (at %.4f s)\n", file, line, func, info.delta, info.cur / 1000.0);
    printf("EXITING %s\n", func);
    mark_time(); // dont count the time of printing
}