#ifndef MERGESTATE_H
#define MERGESTATE_H

#include <stdint.h>
#include <cstddef>
#include <cstdio>
#include <vector>
#include "subckt.h"

void merge_write(FILE* file, const uint32_t* array, const size_t length);
void merge_read(FILE* file, uint32_t** array);
void merge_header(FILE* file, const size_t count);
void merge_save(const char* filename, const std::vector<uint32_t*>& src, const std::vector<SubCkt>& sckts);
void merge_save(const char* filename, const uint32_t* src, const size_t width);

#endif // MERGESTATE_H
