#ifndef SEGMENT_CUH

#define SEGMENT_CUH
#ifdef __CUDACC__
#else
	#define __host__ 
	#define __device__ 
#endif
#include "defines.h"

template <int N> 
union __keytype { int32_t num[N]; uint8_t block[N*sizeof(uint32_t)]; } ;	

#ifdef __CUDACC__
	typedef int2 h_int2;
	typedef uint2 h_uint2;
#else 
typedef struct __h_int2 { int x; int y;  } h_int2;
typedef struct __h_uint2 { unsigned int x; unsigned int y;  } h_uint2;
#endif 

template <int N>
__host__ __device__ __device__ __host__ inline bool operator==(const __keytype<N>& lhs, const __keytype<N>&rhs) {
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
struct segment {
	__keytype<N> key;
	T pattern;
};

template<int N, class T>
inline void descendSegment(const Circuit& ckt, const NODEC& g, const int& level, const int& fin, segment<N,T> v, std::vector<segment<N,T> >& segs) {
	// every call represents another level
	if (level == N-1) {
/*		std::cerr << "Adding (level " << N << "): sid (";
		#pragma unroll
		for (int j = 0; j < N; j++) {
			std::cerr << v.key.num[j] << ":" << ckt.at(v.key.num[j]).name;
			if (j != N-1) 
				std::cerr << ",";
		}
		std::cerr << ")\n";
*/
		segs.push_back(v);
	} else if (g.po != true) {
		// recurse to next level
		for (unsigned int i = 0; i < g.nfo; i++) {
			v.key.num[level+1] = g.fot.at(i).second;
//			std::cerr << "Descending to level " << level+1 << " gate " << g.fot.at(i).second << "\n";
			descendSegment(ckt, ckt.at(g.fot.at(i).second), level+1, g.fot.at(i).second, v, segs);
		}
	} else if (ckt.at(v.key.num[0]).typ == INPT) { 
		#pragma unroll
		for (unsigned int j = level+1; j < N; j++) { v.key.num[j] = -1; }
		segs.push_back(v);
	}
}

// Rules for segment generation:
template <int N, class T>
void generateSegmentList(segment<N,T>** seglist, const Circuit& ckt) {
	std::vector<segment<N,T> > segs;
	// Start at all nodes.
	for (unsigned int gid = 0; gid < ckt.size(); gid++) {
		const NODEC& gate = ckt.at(gid);
		segment<N, T> tmp;
		tmp.key.num[0] = gid;
		if (N > 1) {
			for (unsigned int j = 0; j < gate.fot.size(); j++) {
				tmp.key.num[1] = gate.fot.at(j).second;
				descendSegment(ckt, ckt.at(gate.fot.at(j).second), 1, gate.fot.at(j).second, tmp, segs);
			}
		} else {
			segs.push_back(tmp);
		}
	}
	segment<N, T> tmp;
	tmp.pattern.x = -1;
	tmp.pattern.y = -1;
	#pragma unroll
	for (int i = 0; i < N; i++) {
		tmp.key.num[i] = ckt.size();
	}
	segs.push_back(tmp);
	if (*seglist == NULL) {
		*seglist = (segment<N,T>*)malloc(sizeof(segment<N,T>)*segs.size());
	} else {
		*seglist = (segment<N,T>*)realloc(*seglist, sizeof(segment<N,T>)*segs.size());
	}
	for (unsigned int i = 0; i < segs.size(); i++) {
		(*seglist)[i] = segs[i];
	}
}


template<int N, class T>
void displaySegmentList(segment<N,T>* seglist, const Circuit& ckt) {
	int i = 0;
	while (seglist[i].key.num[0] < ckt.size()) {
		std::cout << "Sid " << i << ": (";
		#pragma unroll
		for (int j = 0; j < N; j++) {
			std::cerr << seglist[i].key.num[j] << ":" << ckt.at(seglist[i].key.num[j]).name;
			if (j != N-1) 
				std::cout << ",";
		}
		std::cout << ")\n";
		i++;
	}
}
#endif
