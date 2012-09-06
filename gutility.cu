#include "utility.h"
#include "defines.h"
#include <cuda.h>
#include <stdint.h>
ARRAY2D<int32_t> gpuAllocateBlockResults(size_t height) {
	int32_t* tgt = NULL;
	cudaMalloc(&tgt, sizeof(int)*(height));
	cudaMemset(tgt, -1, sizeof(int)*height);
	return ARRAY2D<int32_t>(tgt, 1, height, sizeof(int32_t)*height);
}
void checkCudaError(const char* file, int line) {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) { DPRINT("Error %s : before %s:%d\n", cudaGetErrorString(err),file,line);}
}
void selectGPU() {
	int num_devices, device;
	cudaGetDeviceCount(&num_devices);
	cudaDeviceProp properties;
	if (num_devices > 1) {
		unsigned int max_multiprocessors = 0, max_device = 0;
		for (device = 0; device < num_devices; device++) {
			cudaGetDeviceProperties(&properties, device);
			if (max_multiprocessors < properties.totalGlobalMem) {
				max_multiprocessors = properties.totalGlobalMem;
				max_device = device;
			}
		}
		cudaSetDevice(max_device);
		cudaGetDeviceProperties(&properties, max_device);
		DPRINT("Selected %s as GPU.\n", properties.name);
	}
}

int gpuCalculateSimulPatterns(int lines, int patterns) {
	// get free memory
	size_t free_mem, total_mem;
	int allowed_patterns;
	cudaMemGetInfo(&free_mem, &total_mem);
	// added a buffer 	
	allowed_patterns = (free_mem + (lines*sizeof(int))) / (lines*(sizeof(uint32_t)*2.5) + sizeof(uint8_t)*1.5);
	return min(patterns, allowed_patterns -(allowed_patterns % 32));
}
std::string gpuMemCheck(){
	size_t free_mem, total_mem;
	std::stringstream temp;
	cudaMemGetInfo(&free_mem, &total_mem);
	temp << free_mem;
	return temp.str();
}

int** gpuLoadSubCkts(std::vector<SubCkt>::iterator start, std::vector<SubCkt>::iterator end) {
	int dist = std::distance(start,end);
	int **h_sckt_path = (int**)malloc(sizeof(int*)*dist);
	int **sckt_path;
	checkCudaError(__FILE__,__LINE__);
	for (int i = 0; (start+i) < end; i++) {
		h_sckt_path[i] = (start+i)->gpu();
	}
	cudaMalloc(&sckt_path, sizeof(int*)*dist);
	checkCudaError(__FILE__,__LINE__);
	cudaMemcpy(sckt_path, h_sckt_path, sizeof(int*)*dist,cudaMemcpyHostToDevice);

	free(h_sckt_path);
	return sckt_path;
}
