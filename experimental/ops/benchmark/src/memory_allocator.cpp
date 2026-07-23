//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef __APPLE__
#include <malloc.h>
#endif

//#undef VICIOUS_PAGE_SIZE

#include <stdio.h>
#include <stdlib.h>

#ifdef VICIOUS_PAGE_SIZE
#include <sys/mman.h>
#endif

#include "utils.hpp"

void *allocate_aligned_memory(size_t alignment, size_t size) {
#ifdef __APPLE__
    size_t real_size = roundup(size, alignment);

    void *r=aligned_alloc(alignment, real_size);
#else
#ifdef VICIOUS_PAGE_SIZE
    // "Vicious" allocaton policy - round the provided size up to the next page, and provide a pointer offset such
    // that overreading or overwriting will stray into invalid memory.

    // Some kernels deliberately overread by up to 7 bytes.
    size += 7;

    size_t aligned_size = ((size + VICIOUS_PAGE_SIZE-1) / VICIOUS_PAGE_SIZE) * VICIOUS_PAGE_SIZE;
    // Track allocations - we need to provide our own addresses to make sure there are gaps.
    static size_t base_offset=0;

    void *aligned_ptr = mmap((void *)(0xf00000000000 + base_offset), aligned_size, PROT_READ | PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    void *r;

    base_offset += (aligned_size + VICIOUS_PAGE_SIZE);

    size_t offset = size % VICIOUS_PAGE_SIZE;

    if (offset) {
        r = aligned_ptr + (VICIOUS_PAGE_SIZE - offset);
        printf("Allocating %zu bytes, offset=%zu, pointer=%p\n", size, offset, r);
    } else {
        r = aligned_ptr;
        printf("Size already aligned: %zu bytes, pointer=%p\n", size, r);
    }
#else
    void *r=memalign(alignment, size);
#endif // VICIOUS_PAGE_SIZE
#endif // ! __APPLE__

    if (r == nullptr) {
        printf("Memory allocation error.\n");
        exit(1);
    }

    return r;
}

void free_aligned_memory(void *ptr) {
#ifdef VICIOUS_PAGE_SIZE
    // Leak for now - would need a table to do this properly.
#else
    free(ptr);
#endif
}
