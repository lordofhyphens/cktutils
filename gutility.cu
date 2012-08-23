#include "utility.h"
#include "defines.h"
#include <cuda.h>
ARRAY2D<int> gpuAllocateBlockResults(size_t height) {
	int* tgt = NULL;
	cudaMalloc(&tgt, sizeof(int)*(height));
	cudaMemset(tgt, -1, sizeof(int)*height);
	return ARRAY2D<int>(tgt, 1, height, sizeof(int)*height);
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

int gpuCalculateSim(int lines, int patterns) {
	// get free memory
	size_t free_mem, total_mem;
	int allowed_patterns;
	cudaMemGetInfo(&free_mem, &total_mem);
	// added a buffer 	
	allowed_patterns = (free_mem / (lines*(sizeof(char)*2.5)));
	allowed_patterns = min(patterns, allowed_patterns -(allowed_patterns % 32));
	// attempt to allocate/free the required amount of GPU memory, using initial guess as a baseline.
	// We assume that the circuit is already in GPU memory.
	return min(patterns, allowed_patterns);
}

int gpuCalculateSimulPatterns(int lines, int patterns) {
	// get free memory
	size_t free_mem, total_mem;
	int allowed_patterns;
	cudaMemGetInfo(&free_mem, &total_mem);
	// added a buffer 	
	allowed_patterns = (free_mem + (lines*sizeof(char))) / (lines*(sizeof(int)*2.5) + sizeof(char)*lines*2.5);
	int simpatterns = free_mem / (lines * sizeof(char));
	allowed_patterns = min(patterns, allowed_patterns -(allowed_patterns % 32));
	allowed_patterns = min(simpatterns, allowed_patterns);
	// attempt to allocate/free the required amount of GPU memory, using initial guess as a baseline.
	// We assume that the circuit is already in GPU memory.
	return min(patterns, allowed_patterns);
}


