#ifndef UTILITY_H
#define UTILITY_H
#include <ctime>
#include <cmath>
#include "array2d.h"
#include "subckt.h"
#include <cstdlib>
#include <algorithm>
	timespec diff(timespec start, timespec end);
	float floattime(timespec time);
    ARRAY2D<int> gpuAllocateBlockResults(size_t height);
	void selectGPU();
	void loadSubCkts(const Circuit& ckt, std::vector<SubCkt>& subckt, const std::string filename);
	int gpuCalculateSimulPatterns(int lines, int patterns);
	inline bool deleteAll(int * theElement ) { if (theElement != NULL) delete [] theElement; return true; }
#endif // UTILITY_H
