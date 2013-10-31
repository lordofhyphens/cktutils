#include "utility.h"
#include "subckt.h"
#include "defines.h"
#include <cuda.h>
#include <stdint.h>
extern int verbose_flag;
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
uint8_t selectGPU() {
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
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

		DPRINT("Selected %s as GPU.\n", properties.name);
		return max_device;
	}
	return 0;
}
size_t gpuCheckMemory() {
	size_t freeMem = 0;
	size_t totalMem = 0;
	cudaMemGetInfo(&freeMem, &totalMem);  
	DPRINT("Memory avaliable: Free: %lu, Total: %lu\n",freeMem, totalMem); 
	return freeMem;

}
#define DIV_UP(x, y) ( (y) * ( ((x)+(y)-1) / (y) ) )
int gpuCalculateSimulPatterns(unsigned int lines, unsigned int patterns, uint8_t deviceID) {
	// get free memory
	cudaDeviceProp gprop; 
	cudaGetDeviceProperties(&gprop, deviceID);
	DPRINT("Calculating for %u lines, %u patterns\n", lines, patterns);
	size_t free_mem;
	free_mem = gpuCheckMemory();
	int allowed_patterns;
	// added a buffer
	allowed_patterns = (free_mem - sizeof(int2)*lines ) / (lines*(sizeof(int32_t)+sizeof(uint8_t)));
	allowed_patterns *=.8;
	DPRINT("Allowed patterns: %d, ", allowed_patterns);
	//allowed_patterns = (free_mem + (lines*sizeof(uint32_t))) / (lines*(sizeof(uint32_t)*4) + sizeof(uint8_t)*1.5);
	while (DIV_UP(allowed_patterns*sizeof(int32_t), gprop.textureAlignment) > allowed_patterns*sizeof(int32_t)) {
		allowed_patterns-=1;
	}
	allowed_patterns /= 32;
	allowed_patterns *= 32;
	DPRINT("%d corrected for 32\n", allowed_patterns);
	return min(patterns, allowed_patterns);
}
std::string gpuMemCheck(){
	size_t free_mem, total_mem;
	std::stringstream temp;
	cudaMemGetInfo(&free_mem, &total_mem);
	temp << free_mem;
	return temp.str();
}
void resetGPU() { cudaDeviceReset();}
