#include "mergestate.h"

// Writes the introductory header
inline void merge_header(FILE* file, const size_t count) {
	fwrite(&count, sizeof(size_t), 1, file);
}
// Saves a single array to a file.
void merge_write(FILE* file, const uint32_t* array, const size_t length) {
	fwrite(&length, sizeof(size_t), 1, file);
	fwrite(array, sizeof(uint32_t), length, file);
}
// Reads an array from a file, allocating it. 
void merge_read(FILE* file, uint32_t** array) {
	uint64_t length;
	fread(&length, sizeof(size_t), 1, file); // get the count
	*array = new uint32_t[length];
	fread(array, sizeof(uint32_t), length, file);
}

// save a single 
void merge_save(const char* filename, const uint32_t* src, const size_t width) {
	FILE* out = fopen(filename, "wb");
	merge_header(out, 1);
	merge_write(out, src, width);
	fclose(out);
}
void merge_load() {
}
