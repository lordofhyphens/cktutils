/* Templated segment data structures.
 * Requires CUDA.
 *
 *
 */
#include <cuda.h>
#include <stdint.h>

// Used to store GID/PID. Since membership in the cache is sufficient, do not
// need to store anything else.

typedef union hashable_int2_t {
	int2 key;
	unsigned char h[sizeof(int2)];
} hashable_int2;

typedef struct cache_b_t { 
	uint32_t mutex;
	hashable_int2 key;
} cache_b;


/* Generic segment structure. Uses int2 for the pid, -/+ */
template <int N>
struct segment_t { 
	uint32_t mutex;
	int key[N];
	int2 pid;
};

// Specialized templates to take advantage of inbuilt CUDA vector types.
template<>
struct segment_t<4> {
	uint32_t mutex;
	int4 nums;
	__device__ __host__ inline int get(uint8_t id) { switch (id) { case 0: return nums.x; case 1: return nums.y; case 2: return nums.z; case 3: return nums.w;} return 0;};
	__device__ __host__ inline void set(uint8_t id, const int& val) { switch (id) { case 0: nums.x = val; break; case 1: nums.y = val; break; case 2: nums.z = val; break; case 3: nums.w = val;} };
};
template<>
struct segment_t<2> {
	int2 nums;
	uint32_t mutex;
	__device__ __host__ inline int get(uint8_t id) { switch (id) { case 0: return nums.x; case 1: return nums.y;} return 0;};
	__device__ __host__ inline void set(uint8_t id, const int& val) { switch (id) { case 0: nums.x = val; break; case 1: nums.y = val;} };
};

