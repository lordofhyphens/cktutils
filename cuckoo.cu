#include <cuda.h>
#include <cstdio>
#include "util/utility.h"
#include "util/ckt.h"
#include "util/gpuckt.h"
#include "util/ckt.h"
#include "util/gpudata.h"
#include "util/vectors.h"
#include "util/subckt.h"
#include <stdint.h>
#include <set>
#include "lookup.h"
// Prototype code to test effects of multiple hash functions on our expected key sets.

// Starting at src, get a segment of up to length l by following 
// from this node to following nodes. Return some structure containing 
// a L-tuple. The tuple length is defined at compile-time.

 __host__ __device__ uint32_t hashlittle( const void *key, size_t length, uint32_t initval);
template <int N> 
union __keytype { uint32_t num[N]; uint8_t block[N*sizeof(uint32_t)]; } ;	
typedef __keytype<2> key_2;

template <int N>
__device__ __host__ inline bool operator==(const __keytype<N>& lhs, const __keytype<N>&rhs) {
	bool tmp = true;
	#pragma unroll 2
	for (int i = 0; i < N; i++)
		tmp = tmp && (lhs.num[i] == rhs.num[i]);
	return tmp;
}
template <int N>
__device__ __host__ inline bool operator!=(const __keytype<N>& lhs, const __keytype<N>&rhs) {
	return !(lhs == rhs);
}
template <int N, class T>
struct segment_t {
	uint32_t lock;
	__keytype<N> key;
	T pattern;
};

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
	printf("Looking for (%d, %d):\n", key.num[0], key.num[1]);
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

__global__ void hashcheck(const uint32_t *hashvals, seg_2 *array, uint32_t g, uint32_t gnext) {
	key_2 tmpkey =  { { g, gnext }}; 
	int32_t hashval = hasKey(hashvals, array, tmpkey, 25);
	if (hashval >= 0) {
		seg_2 result = array[HASH_2(tmpkey,hashval,21)];
		printf("\n%u, %u: %d\n", result.key.num[0], result.key.num[1], result.pattern );
	} else {
		printf("\nHash not found\n");
	}
}
seg_2 array[hashsize(21)];

__host__ int main(int argc, const char* argv[]) {
	uint8_t device = selectGPU();
	GPU_Circuit ckt;
	srand(time(0));
	std::cerr << "Reading benchmark file " << argv[1] << "....";
	std::string infile(argv[1]);
	if (infile.find("bench") != std::string::npos) {
		ckt.read_bench(infile.c_str());
	} else {
			std::clog << "presorted benchmark " << infile << " ";
		ckt.load(infile.c_str());
	}
//	ckt.print();
	// Build a list of all segments.
	uint32_t segments = 2;
	std::set<uint32_t> counts;
	uint32_t collisions = 0;
	uint32_t segcount = 0;
	uint32_t storagecount=0;
	uint32_t maxused = 0;
	uint32_t *dc_h;
	uint32_t maxhashes=25;
	uint32_t h[25];
	srand(time(0));
	for (int i = 0;i < 25; i++) {
		h[i] = rand();
	}
	cudaError_t err = cudaSuccess;
	err = cudaMalloc(&dc_h, sizeof(uint32_t)*maxhashes);
	if (err != cudaSuccess) { printf("Error at %d", __LINE__);}
	err = cudaMemcpy(dc_h, h, sizeof(uint32_t)*maxhashes,cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { printf("Error at %d", __LINE__);}
	for (int g = 0; g < ckt.size(); g++) {
		const NODEC& gate = ckt[g];
		// iterate over all fanins/fanouts of this node
		// For segment length of 2, only the POs have no fan-outs.
		// Can use C++ tuples for compile-time indexing.
		// Segment length of 1:
		switch(segments) {
			case 1: 
				break;
			case 2:
				for (int fot = 0; fot < gate.fot.size(); fot++) {
					// true is T1
					// false is T0
					// Segment length of 2
					seg_2 k;
					k.key.num[0] = g;
					k.key.num[1] = gate.fot.at(fot).second;
					k.pattern = -1;
					segcount++;
					
					uint32_t usedhashes = 0;
					uint32_t hash;
					hash = HASH_2(k,h[usedhashes],21);
					seg_2 evict;
					// Otherwise, swap it with this on
					evict = array[hash]; // store the old value into evict
					array[hash] = k; // store our value.
						std::cout << "Storing " << k.key.num[0] << "," << k.key.num[1] << " with hashvalue " << hash << ", id " << usedhashes << "\n";
					k = evict;
					while (!isEmpty(evict) && usedhashes < maxhashes) {
						usedhashes++;
						hash = (HASH_2(k,h[usedhashes],21));
						evict = array[hash]; // store the old value into evict
						array[hash] = k; // store our value.
						std::cout << "Storing " << k.key.num[0] << "," << k.key.num[1] << " with hashvalue " << hash << ", id " << usedhashes << "\n";
						k = evict;
					}
					if (!isEmpty(evict)) {
						collisions++;
					} else {
						storagecount++;
						if (usedhashes > maxused) {
							maxused = usedhashes;
						}
					}

				}
				break;
		}
	}
	seg_2 *dc_segs;
	key_2 test_key = { { 10868, 15952 }};
	if ( hasKey(h, array, test_key, 25) >=0) {
		std::cout << "in array." << "\n";
	} else { 
		std::cout << "not in array." << "\n";
	}
	err = cudaMalloc(&dc_segs, sizeof(seg_2)*hashsize(21));
	if (err != cudaSuccess) { printf("Error at %d", __LINE__);}
	err =cudaMemcpy(dc_segs, array, sizeof(seg_2)*hashsize(21),cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { printf("\nError at %d: %s", __LINE__, cudaGetErrorString(err));}
	hashcheck<<<1,1>>>(dc_h,dc_segs,10868,15952);
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess) { printf("\nError at %d: %s", __LINE__, cudaGetErrorString(err));}
	std::cout << "Collisions: " << collisions << " for " << segcount << " segments, stored " << storagecount << " in " << hashsize(21) << " slots, " << (double)storagecount / hashsize(21) << " load, " << (double)collisions /segcount << "% collisions\n"; 
	std::cout << "Max chain used:" << maxused << "\n";
	cudaDeviceReset();
	return 0;
}
