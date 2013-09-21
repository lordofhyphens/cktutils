#if defined(__CUDACC__)
#include "lookup.h"
#include "segment.cuh"
// Prototype code to test effects of multiple hash functions on our expected key sets.

// Starting at src, get a segment of up to length l by following 
// from this node to following nodes. Return some structure containing 
// a L-tuple. The tuple length is defined at compile-time.

 __host__ __device__ uint32_t hashlittle( const void *key, size_t length, uint32_t initval);

// Utility function and structure to pack data about out hash functions. 
typedef struct hashfuncs_t { uint32_t *hashlist, max, slots; } hashfuncs;
__device__ __host__ hashfuncs make_hashfuncs(uint32_t* hashlist, max, slots) { hashfuncs a; a.hashlist = hashlist, a.max = max, a.slots = slots; return a;}

// This function needs to be called with different key values per-thread or
// else it will thrash the hashmap.  Usage: Allocate enough memory on the
// device for the hashmap. We assume/hope/pray that that value is small enough
// so we don't have to allocate (avg_nfo)^(segment_length) * 1.5 maximum slots,
// but we probably will.

template <int N, class T> 
__device__ void insertIntoHash(segment_t<N,T>* storage, const hashfuncs hashfunc, const uint32_t slots, __keytype<N> key, T data) {
	uint8_t hashes = 0;
	segment_t<N,T> evict; // hopefully we won't need this.
	evict.key.num[0] = 1;
	// repeat the insertion process while the thread still has an item.
	// Treat failures to acquire a lock as a collision w/o .
	// Performance will be lower on non-Kepler architectures.
	// due to less efficient memory atomics.
	// If there is a collision, evict and try again with the next hash.
	int pred = 1;
	while (evict.key.num[0] != 0) {
		hash = getHash<N>(key.block,I,hashlist[hashes],hashfunc.slots);
		while (pred && hashes < hashfunc.max) {
			pred = atomicCAS(&(st[hash].lock), 0, 1) != 0;
			if (pred) { // divergence possibility here.
				hashes++; // try another hash function.
			}
			// Loop until every thread gets a lock on something,
			// or we run out of functions.
		}
		if (pred) { 
			// Something is seriously wrong here. It may be possible that the locking 
			// thread will give it up and we'll boot it out eventually.
			printf("Somehow, we managed to have %d hash collisions with concurrent executions and ran out of hash functions. Trying again.\n", N);
			hashes = 0;
			continue;
		}
		// Value of hash should be different in different threads. 
		// We're allowed to write to the structure now.
		evict = st[hash]; evict.lock = 0;
		st[hash].key = key;
		st[hash].pattern = data;
		atomicExch(&(st[hash].lock),0); // reset the lock on this entry in memory.
		// check to see whether or not the eviction booted out a legal value.
		// if it has, need to boot it out.
		key = evict.key; data = evict.pattern;
		hashes++;
		if (hashes >= hashfunc.max) {
		 	// We ran out of hashes, reset? 
			// Probably a terrible idea.
			hashes = 0;
		}
	}
}
template <int N>
__device__ __host__	bool isEmpty(const segment_t<N> &a) {
	bool tmp = true;
	for (uint8_t i = 0; i < N; i++) {
		tmp = tmp && a.key.num[i] == 0;
	}
	return tmp;
}
typedef segment_t<2> seg_2;

#define hashsize(n) ((uint32_t)1<<(n))
#define hashmask(n) (hashsize(n)-1)
#define HASH(I,HV,LEN,KEYLEN) (hashlittle(I.key.block,sizeof(uint32_t)*KEYLEN, HV) & hashmask(LEN))
#define HASH_2(I,HV,LEN) (getHash<2>(I,HV,LEN))


template <int N>
__device__ __host__ inline uint32_t getHash(const segment_t<N>& I, const uint32_t seed, uint8_t bits) {
	return hashlittle(I.key.block,sizeof(uint32_t)*N, seed) & hashmask(bits);
}
template <int N>
__device__ __host__ inline uint32_t getHash(const __keytype<N>& I, const uint32_t seed, uint8_t bits) {
	return hashlittle(I.block,sizeof(uint32_t)*N, seed) & hashmask(bits);
}

// if this key is in the array, return the hashvalue used to find it, -1 if not found. 
__host__ __device__ int32_t hasKey(const uint32_t *hashvals, seg_2 *array, __keytype<2> key, const uint32_t max) {
	int32_t used = 0;
	uint32_t hash = getHash<2>(key,used,21);
	while (array[hash].key != key && used < max) {
		printf("Checking hash id %d: %u\n", used, hashvals[used]);
		hash = getHash<2>(key,hashvals[used],21);
		assert(hash < hashsize(21));
		printf("Hash key found: %u %u\n", array[hash].key.num[0], array[hash].key.num[1]);
		used++;	
	}
	return (used < max ? hashvals[used] : -1);
}

#endif
