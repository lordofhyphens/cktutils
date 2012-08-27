#ifndef MERGESTATE_H
#define MERGESTATE_H

#include <stdint.h>
#include <cstddef>
#include <cstdio>
#include <vector>
#include "subckt.h"

void merge_write(FILE* file, const int32_t* array, const size_t length);
void merge_header(FILE* file, const size_t count);
void merge_save(const char* filename, const std::vector<int32_t*>& src, const std::vector<SubCkt>& sckts);
void merge_save(const char* filename, const int32_t* src, const size_t width);

void merge_read(FILE* file, int32_t** array);
void merge_load(const char* filename, std::vector<int32_t*>& dst);
int32_t* merge_load(const char* filename);
#endif // MERGESTATE_H
