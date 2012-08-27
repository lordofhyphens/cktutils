#ifndef MERGESTATE_H
#define MERGESTATE_H

#include <stdint.h>
#include <cstddef>
#include <cstdio>
#include <vector>

void merge_write(FILE* file, const uint32_t* array, const size_t length);
void merge_read(FILE* file, uint32_t** array);
void merge_header(FILE* file, const size_t count);

#endif // MERGESTATE_H
