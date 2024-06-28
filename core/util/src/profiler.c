/**
 * \file                profiler.c
 * \brief               Profile execution time
 * \todo                See if a macro can be used to mark the function entry time
 *                      Log information to file
 */

#include <util/profiler.h>

#include <stdio.h>
#include <time.h>

static time_t prev_time; /* track last marked time */
static int count_marks = 0; /* number of times mark_time was called */

/**
 * \brief               Timer info
 * \note                Used to track current and elapsed time
 */
typedef struct ProfilerInfo {
    time_t cur;                                 /*!< Current time in milliseconds */
    time_t delta;                               /*!< Elapsed time in milliseconds */
} profiler_info_t;

/**
 * \brief               Mark current time and track elapsed time since last mark
 * \note                Time is stored in milliseconds
 * \return              Current and elapsed time
 */
static profiler_info_t
mark_time(void) {
    time_t time_since = 0; /* initialize to 0 in case this is the first call */
    time_t now = clock(); 
    if (count_marks != 0) { /* called before so find elapsed time */
        time_since = now - prev_time;
    }
    count_marks++;

    prev_time = now;
    return (profiler_info_t) {.cur = now, .delta = time_since};
}

/**
 * \brief               Mark time
 */
void 
mark_func_time(const char* file, const char* func, int line) {
    profiler_info_t info = mark_time();
    printf("%s:%d [%s] in %llu ms (at %.4f s)\n", file, line, func, info.delta, info.cur / 1000.0); /* log time */
    mark_time(); /* don't count the time taken to print */
}

/**
 * \brief               Mark time with a name
 */
void mark_func_entry_time(const char* file, const char* func, int line, const char* entry) {
    profiler_info_t info = mark_time();
    printf("%s:%d [%s] in %llu ms (at %.4f s): %s\n", file, line, func, info.delta, info.cur / 1000.0, entry); /* log time and entry name*/
    mark_time(); /* don't count the time taken to print */
}

/**
 * \brief               Mark function exit time
 */
void mark_func_exit(const char* file, const char* func, int line) {
    profiler_info_t info = mark_time();
    printf("%s:%d [%s] in %llu ms (at %.4f s)\n", file, line, func, info.delta, info.cur / 1000.0); /* log time */
    printf("EXITING %s\n", func);
    mark_time(); /* don't count the time taken to print */
}