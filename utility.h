#ifndef UTILITY_H
#define UTILITY_H
#include <ctime>
#include <cmath>
#include "array2d.h"
#include "subckt.h"
#include <cstdlib>
#include <algorithm>
#include "subckt.h"
	timespec diff(timespec start, timespec end);
	float floattime(timespec time);
    ARRAY2D<int> gpuAllocateBlockResults(size_t height);
	void selectGPU();
	void loadSubCkts(const Circuit& ckt, std::vector<SubCkt>& subckt, const std::string filename);
	int gpuCalculateSimulPatterns(int lines, int patterns);
	void checkCudaError(const char* file, int line);
	int largest_size(std::vector<SubCkt>::iterator& start, std::vector<SubCkt>::iterator& end, int level);
	int** gpuLoadSubCkts(std::vector<SubCkt>::iterator start, std::vector<SubCkt>::iterator end);
	void gpuCleanup(void*);
	void loadSubCkts(const Circuit& ckt,std::vector<SubCkt>& subckt, std::string filename);

#endif // UTILITY_H
