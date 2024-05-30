#pragma once
#ifndef PROFILER_H
#define PROFILER_H

void mark_func_time(const char* file, const char* func, int line);
void mark_func_exit(const char* file, const char* func, int line);


#define CLOCK_MARK mark_func_time(__FILE__, __func__, __LINE__);
#define CLOCK_MARK_ENTRY(entry) mark_func_entry_time(__FILE__, __func__, __LINE__, entry);
#ifndef PROFILER_DISABLE_FUNCTION_RETURN
#define return mark_func_exit(__FILE__, __func__, __LINE__); \
return
#endif

#ifdef TIME
    #undef CLOCK_MARK
    #undef CLOCK_MARK_ENTRY
    #undef return
#endif
#endif // PROFILER_H
