/*
 * engine/logging.h
 *
 * 【platform 层】轻量调试日志（stdout/stderr）。
 * 用于 ModelCoordinator 切换、懒加载、信号合并等关键路径的可观测性。
 */
#pragma once

#include <cstdio>
#include <cstdarg>

inline void LogInfo(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    std::fprintf(stdout, "[INFO] ");
    std::vfprintf(stdout, fmt, args);
    std::fprintf(stdout, "\n");
    std::fflush(stdout);
    va_end(args);
}

inline void LogWarn(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    std::fprintf(stderr, "[WARN] ");
    std::vfprintf(stderr, fmt, args);
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
    va_end(args);
}

inline void LogError(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    std::fprintf(stderr, "[ERROR] ");
    std::vfprintf(stderr, fmt, args);
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
    va_end(args);
}

inline void LogFatal(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    std::fprintf(stderr, "[FATAL] ");
    std::vfprintf(stderr, fmt, args);
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
    va_end(args);
}