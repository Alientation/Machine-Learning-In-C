/**
 * \file                profiler.h
 * \brief               profiler include file
 * \note                '#TIME' will enable stopwatch behavior through the macros '#CLOCK_MARK' and '#CLOCK_MARK_ENTRY'
 *                          Setting '#PROFILER_DISABLE_FUNCTION_RETURN' will prevent replacing all function 'return' statements with a
 *                          mark time call
 */

#pragma once
#ifndef PROFILER_H
#define PROFILER_H

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * \defgroup            Profiler
 * \brief               Timer control 
 * \{
 */

void mark_func_time(const char* file, const char* func, int line);
void mark_func_exit(const char* file, const char* func, int line);
void mark_func_entry_time(const char* file, const char* func, int line, const char* entry);


#ifdef TIME
/**
 * \brief               Mark current and elapsed time with information about source location
 * \hideinitializer
 */
#define CLOCK_MARK mark_func_time(__FILE__, __func__, __LINE__);

/**
 * \brief               Mark and label current and elapsed time with information about source location
 * \hideinitializer
 */
#define CLOCK_MARK_ENTRY(entry) mark_func_entry_time(__FILE__, __func__, __LINE__, entry);

#ifndef PROFILER_DISABLE_FUNCTION_RETURN
/**
 * \brief               Marks on function exit
 */
/* WARNING in certain circumstances like bracketless control statements */
#define return  mark_func_exit(__FILE__, __func__, __LINE__); \
                return
#endif /* PROFILER_DISABLE_FUNCTION_RETURN */
#endif /* TIME */

/**
 * \}
 */

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* PROFILER_H */
