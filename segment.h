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

template <int N>
union hashkey {
	unsigned int k[N];
	unsigned char h[sizeof(unsigned int)*N];
};
/* Generic segment structure. Uses int2 for the pid, -/+ */
template <int N>
struct segment_t { 
	uint32_t mutex;
	hashkey<N> key;
	int2 pid;
};
