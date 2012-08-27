#include "mergestate.h"
using namespace std;
// Writes the introductory header
inline void merge_header(FILE* file, const size_t count) {
	fwrite(&count, sizeof(size_t), 1, file);
}
// Saves a single array to a file.
void merge_write(FILE* file, const int32_t* array, const size_t length) {
	fwrite(&length, sizeof(size_t), 1, file);
	fwrite(array, sizeof(int32_t), length, file);
}
// Reads an array from a file, allocating it. 
void merge_read(FILE* file, int32_t** array) {
	size_t length;
	fread(&length, sizeof(size_t), 1, file); // get the count
	*array = new int32_t[length];
	fread(array, sizeof(int32_t), length, file);
}

// save a single 
void merge_save(const char* filename, const int32_t* src, const size_t width) {
	FILE* out = fopen(filename, "wb");
	merge_header(out, 1);
	merge_write(out, src, width);
	fclose(out);
}
void merge_save(const char* filename, const std::vector<int32_t*>& src, const vector<SubCkt>& sckts) {
	FILE* out = fopen(filename, "wb");
	size_t subid = 0;
	merge_header(out, src.size());
	for (vector<SubCkt>::const_iterator it = sckts.begin(); it < sckts.end(); it++)
		merge_write(out, src.at(subid++), it->size());
	fclose(out);
}
void merge_load(const char* filename, std::vector<int32_t*>& dst) {
	int32_t* temp_array;
	FILE* in = fopen(filename,"rb");
	size_t length;
	fread(&length, sizeof(size_t), 1, in);
	for (size_t i = 0; i < length; i++) {
		temp_array = NULL;
		merge_read(in, &temp_array);
		dst.push_back(temp_array);
	}
}

int32_t* merge_load(const char* filename) {
	int32_t* temp_array;
	FILE* in = fopen(filename,"rb");
	size_t length;
	fread(&length, sizeof(size_t), 1, in);
	for (size_t i = 0; i < length; i++) {
		merge_read(in, &temp_array);
	}
	return temp_array;
}
