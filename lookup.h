#ifndef LOOKUP_H
#define LOOKUP_H

#define hashsize(n) ((uint32_t)1<<(n))
#define hashmask(n) (hashsize(n)-1)

#if defined(__CUDACC__)
 __host__ __device__ uint32_t hashlittle( const void *key, size_t length, uint32_t initval);
#else 
uint32_t hashlittle( const void *key, size_t length, uint32_t initval);
#endif // CUDA-specific stuff.

#endif
