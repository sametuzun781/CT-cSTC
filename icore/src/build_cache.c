/* Metadata checks for the Python build cache. No filesystem facts are cached.
 * Paths and expected metadata belong to the caller; all descriptors are local
 * to this call. Keep the numerical solver independent of this optional helper. */
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE 1
#endif
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include <stddef.h>
#include <stdint.h>
#if defined(__linux__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

struct icore_file_group {
    const char *parent;
    size_t count;
    const char *const *paths;
    const char *const *names;
    const uint64_t *expected;
};

#ifdef __cplusplus
extern "C" {
#endif
/* Batch PATH probing without caching filesystem facts. Match Python's
 * stat-following-symlinks, non-directory and access(X_OK) checks exactly. */
ptrdiff_t icore_executable_candidate(size_t count, const char *const *paths)
{
#if defined(__linux__) || defined(__APPLE__)
    for (size_t i = 0; i < count; ++i) {
        struct stat s;
        if (stat(paths[i], &s) == 0 && !S_ISDIR(s.st_mode) && access(paths[i], X_OK) == 0)
            return (ptrdiff_t)i;
    }
    return -1;
#else
    (void)count;
    (void)paths;
    return -2;
#endif
}

int icore_files_unchanged(size_t groups, const struct icore_file_group *group)
{
#if defined(__linux__) || defined(__APPLE__)
    size_t g, i;
    for (g = 0; g < groups; ++g) {
        const struct icore_file_group *p = group + g;
        int descriptor = p->count >= 8 ? open(p->parent, O_RDONLY | O_DIRECTORY | O_CLOEXEC) : -1;
        for (i = 0; i < p->count; ++i) {
            struct stat s;
            const uint64_t *v = p->expected + 8 * i;
            int status =
                descriptor < 0 ? stat(p->paths[i], &s) : fstatat(descriptor, p->names[i], &s, 0);
#if defined(__APPLE__)
#define MODIFIED st_mtimespec
#define CHANGED st_ctimespec
#else
#define MODIFIED st_mtim
#define CHANGED st_ctim
#endif
            if (status || v[0] != (uint64_t)s.st_dev || v[1] != (uint64_t)s.st_ino ||
                v[2] != (uint64_t)s.st_mode || v[3] != (uint64_t)s.st_size ||
                v[4] != (uint64_t)s.MODIFIED.tv_sec || v[5] != (uint64_t)s.MODIFIED.tv_nsec ||
                v[6] != (uint64_t)s.CHANGED.tv_sec || v[7] != (uint64_t)s.CHANGED.tv_nsec) {
                if (descriptor >= 0)
                    close(descriptor);
                return 0;
            }
#undef MODIFIED
#undef CHANGED
        }
        if (descriptor >= 0)
            close(descriptor);
    }
    return 1;
#else
    (void)groups;
    (void)group;
    return -1;
#endif
}
#ifdef __cplusplus
}
#endif
