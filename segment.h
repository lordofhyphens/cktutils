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

