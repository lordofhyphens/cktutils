#ifndef DEFINES_CUH
#define DEFINES_CUH

#ifdef __CUDACC__
template <class T> 
__device__ __host__ inline T& REF2D(T* ARRAY,size_t PITCH, int X, int Y) { return ((T*)((char*)ARRAY + Y*PITCH))[X]; }
#else
template <class T> 
inline T& REF2D(T* ARRAY,size_t PITCH, int X, int Y) { return ((T*)((char*)ARRAY + Y*PITCH))[X]; }
#endif
#endif 
