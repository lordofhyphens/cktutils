#ifndef UTILITY_H
#define UTILITY_H
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include "array2d.h"
#include "subckt.h"
	timespec diff(timespec start, timespec end);
	float floattime(timespec time);
    ARRAY2D<int> gpuAllocateBlockResults(size_t height);
	uint8_t selectGPU();
	void loadSubCkts(const Circuit& ckt, std::vector<SubCkt>& subckt, const std::string filename);
	int gpuCalculateSimulPatterns(unsigned int lines, unsigned int patterns, uint8_t deviceID);
	void checkCudaError(const char* file, int line);
	int largest_size(std::vector<SubCkt>::iterator& start, std::vector<SubCkt>::iterator& end, int level);
	GPU_SCKT_BATCH gpuLoadSubCkts(std::vector<SubCkt>::iterator start, std::vector<SubCkt>::iterator end);
	void gpuCleanup(void*);
	void loadSubCkts(const Circuit& ckt,std::vector<SubCkt>& subckt, std::string filename);
	float elapsed(const timespec start);
	void loadSubCkts(const Circuit& ckt, std::vector<SubCkt>& subckt);
	inline bool deleteAll(int * theElement ) { if (theElement != NULL) delete [] theElement; return true; }
	size_t gpuCheckMemory();
	void resetGPU();
	
#ifdef __CUDACC__
#include <stdio.h>
__device__ inline int find(int* ckt, int tgt) {
	int i = 0;
	while (ckt[i] != -1 && ckt[i] != tgt)
		i++;
	return i;
}

#define KEY_NOT_FOUND -1
__device__ inline int midpoint(int min, int max) {
	return (min + ((max-min)/2));
}
// Performs a reverse-lookup using binary search.
// src is the position in the *subcircuit* the current gate lies.
// limit is either 0 or the size of the subcircuit, indicates 
// whether or not looking for a PI or a PO reference.
// uses a delayed detection of equality to try to keep 
// all threads together
// ckt is the subcircuit
__device__ inline int bin_find(const int ckt[],const int src,const int tgt,const size_t limit) {
	int imin, imax;
	imin = (limit >= src)*(src) + (src >= limit)*(limit);
	imax = (limit >= src)*(limit) + (src >= limit)*(src);
	while (imin < imax) {
		int imid = midpoint(imin, imax);
		imax = (tgt < ckt[imid])*(imid-1) + (tgt > ckt[imid])*(imax) + (tgt == ckt[imid])*imid;
		imin = (tgt < ckt[imid])*(imin) + (tgt > ckt[imid])*(imid+1) + (tgt == ckt[imid])*imid;
	}
	// check for equality
	//printf("%s:%d finish: imin == imax:%d, ckt[imin] == key: %d, return %d\n",__FILE__,__LINE__,imin == imax, ckt[imin] == tgt, KEY_NOT_FOUND*(ckt[imin] != tgt) + (ckt[imin] == tgt)*imin);
	return KEY_NOT_FOUND*(ckt[imin] != tgt) + (ckt[imin] == tgt)*(imin);

}
#endif // CPU include guard

#endif // UTILITY_H
