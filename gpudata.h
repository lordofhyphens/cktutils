#ifndef GPUDATA_H
#define GPUDATA_H

#include <iostream> // included for debugging

#include <utility>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include "errors.h"
#include "defines.h"
#include "array2d.h"
#include "cpudata.h"

#ifdef __CUDACC__ 
	#define HOST_DEVICE __device__ __host__
#else 
	#define HOST_DEVICE
#endif 

#define CHARPAIR (std::pair<uint8_t*,uint8_t*>())
typedef std::vector<ARRAY2D<uint8_t> >::iterator dataiter;
template <class T>
struct GPU_DATA_type {
	size_t pitch, width, height;
	T* data;
};
typedef GPU_DATA_type<uint8_t> g_GPU_DATA;

union coalesce_t { 
	uint32_t packed;
	uint8_t rows[4];
	HOST_DEVICE operator uint32_t() { return packed; }
	HOST_DEVICE coalesce_t() : packed((unsigned)0) {} 
	HOST_DEVICE coalesce_t(const coalesce_t& other) : packed(other.packed) {}
	HOST_DEVICE coalesce_t(uint32_t a) : packed(a) {} 
	HOST_DEVICE coalesce_t( uint8_t p0, uint8_t p1, uint8_t p2, uint8_t p3 ) { rows[0] = p0; rows[1] = p1; rows[2] = p2; rows[3] = p3;} 
};

HOST_DEVICE coalesce_t vectAND(coalesce_t a, uint32_t b); 
// Specialized REF2D
#ifdef __CUDACC__
template <class T> 
HOST_DEVICE inline T& REF2D(const GPU_DATA_type<T>& POD, int PID, int GID) { return ((T*)((char*)POD.data + GID*POD.pitch))[PID]; }

HOST_DEVICE inline coalesce_t& REF2D(const GPU_DATA_type<coalesce_t>& POD, int PID, int GID) { assert(GID >= 0); assert(PID >= 0); return *((coalesce_t*)((char*)POD.data + GID*POD.pitch)+PID); }

#else
template <class T> 
inline T& REF2D(const GPU_DATA_type<T>& POD, int PID, int GID) { return (T*)((char*)(POD.data) + GID*POD.pitch)[PID]; }
#endif


class GPU_Data : public CPU_Data {
	private:
		size_t _block_size;
		ARRAY2D<uint8_t>* _gpu; // fixed size GPU memory space.
		uint32_t copy(uint32_t, bool coherent = false); // copy CPU to GPU. If coherent=true, performs a GPU<->CPU swap
		size_t rows, columns, bl_width;
	public: 
		void clear();
		void unload();
		inline g_GPU_DATA gpu_pack(int ref) { g_GPU_DATA z; ARRAY2D<uint8_t> t = gpu(ref); z.pitch = t.pitch; z.width = t.width; z.height = t.height; z.data = t.data; return z; }
		inline g_GPU_DATA gpu_pack() { g_GPU_DATA z; ARRAY2D<uint8_t> t = gpu(); z.pitch = t.pitch; z.width = t.width; z.height = t.height; z.data = t.data; return z; }
		ARRAY2D<uint8_t> gpu(uint32_t ref, bool coherent = false); // this will throw an out_of_range exception if ref > size; Also changes current.
		ARRAY2D<uint8_t> gpu() { return gpu(this->_current);}
		ARRAY2D<uint8_t> gpu() const { return *_gpu;} // special function for read-only version
		uint32_t refresh(); // ensures that the GPU memory space is equivalent to cpu-current.
		size_t block_width() { return this->_block_size;}
		uint32_t initialize(size_t, size_t, uint32_t);
		GPU_Data();
		GPU_Data(size_t rows, size_t columns);
		GPU_Data(size_t rows, size_t columns, uint32_t blockwidth);
		~GPU_Data();
		std::string debug();
		ARRAY2D<uint8_t> ar2d() const { return *(this->_gpu); }
};

inline const g_GPU_DATA toPod(GPU_Data& data) {
	g_GPU_DATA tmp;
	tmp.data = data.gpu().data;
	tmp.height = data.gpu().height;
	tmp.pitch = data.gpu().pitch;
	tmp.width = data.gpu().width;
	return tmp;
}

template <class T>
inline const GPU_DATA_type<T> toPod(GPU_Data& data) {
	GPU_DATA_type<T> tmp;
	tmp.data = (T*)data.gpu().data;
	tmp.height = data.gpu().height;
	tmp.pitch = data.gpu().pitch;
	tmp.width = data.gpu().width;
	return tmp;
}
template <class T>
inline const GPU_DATA_type<T> toPod(GPU_Data& data, size_t chunk) {
	GPU_DATA_type<T> tmp;
	tmp.data = (T*)data.gpu(chunk).data;
	tmp.height = data.gpu(chunk).height;
	tmp.pitch = data.gpu(chunk).pitch;
	tmp.width = data.gpu(chunk).width;
	return tmp;
}

void gpu_shift(GPU_Data& pack);
void debugDataOutput(ARRAY2D<uint8_t> results, std::string outfile);
void debugDataOutput(GPU_Data& results, std::string outfile);
#endif //GPUDATA_H
