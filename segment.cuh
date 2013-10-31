#ifndef SEGMENT_CUH
#define SEGMENT_CUH
#ifdef __CUDACC__
#include <cuda.h>
#include <algorithm>
#include "gpudata.h"
inline void HandleCUDAError( cudaError_t err, const char *file, uint32_t line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
 } 
#else
	#define __host__ 
	#define __device__ 
#endif
#include "defines.h"

template <int N> 
union __keytype { 
	int32_t num[N]; uint8_t block[N*sizeof(uint32_t)]; 
	HOST_DEVICE  __keytype& operator=( const int32_t& rhs ) { for (int i = 0; i < N; i++) num[i] = rhs; return *this;}
} ;	

#ifdef __CUDACC__
	typedef int2 h_int2;
	typedef uint2 h_uint2;
#else 
typedef struct __h_int2 { int x; int y;  } h_int2;
typedef struct __h_uint2 { unsigned int x; unsigned int y;  } h_uint2;
#endif 

template <int N>
HOST_DEVICE bool operator==(const __keytype<N>& lhs, const __keytype<N>&rhs) {
	bool tmp = true;
	for (int i = 0; i < N; i++)
		tmp = tmp && (lhs.num[i] == rhs.num[i]);
	return tmp;
}

template <int N>
HOST_DEVICE bool operator<(const __keytype<N>& lhs, const __keytype<N>&rhs) {
	bool tmp = true;
	bool prev = false;
	for (int i = 0; i < N; i++) {
		tmp = tmp && (lhs.num[i] <= rhs.num[i] || prev);
		prev = tmp; 
	}
	return tmp && (lhs != rhs);
}
template <int N>
HOST_DEVICE bool operator!=(const __keytype<N>& lhs, const __keytype<N>&rhs) {
	return !(lhs == rhs);
}

// DC_SEGMENT
// Device-context structure of arrays
// Characteristics: 
// Keys height: height
// width N (entries), pitch set in pitch.
// REF2D(key, pitch, S, sid)
template <int N, class T>
struct dc_segment {
	int32_t* key;
	T* pattern;
	size_t height;
	size_t pitch;
};

template <int N, class T>
struct segment {
	__keytype<N> key;
	T pattern;
	HOST_DEVICE segment(const segment<N,T>& s) { key = s.key; pattern = s.pattern; };
	HOST_DEVICE segment() { key = 0; };
};

template <int N, class T>
HOST_DEVICE bool operator==(const segment<N,T>& lhs, const segment<N,T>&rhs) {
	return lhs.key == rhs.key;
}
template <int N, class T>
HOST_DEVICE bool operator<(const segment<N,T>& lhs, const segment<N,T>&rhs) {
	return lhs.key < rhs.key;
}

template<int N, class T>
inline void descendSegment(const Circuit& ckt, const SubCkt& sckt, const NODEC& g, const int& level, const int& fin, segment<N,T> v, std::vector<segment<N,T> >* segs) {
	// every call represents another level
	if (level == N-1) {
//		std::cout << "Adding (level " << N << "): sid " << segs->size() << " (";
//		for (int j = 0; j < N; j++) {
//			std::cout << v.key.num[j] << ": " << ckt.at(v.key.num[j]).name;
//			if (j != N-1) 
//			std::cout << ",";
//		}
//		std::cout << ")\n";
		segs->push_back(v);
	} else if (g.po != true) {
		// recurse to next level
		for (unsigned int i = 0; i < g.nfo; i++) {
			if (sckt.in(g.fot.at(i).second) < 0) continue;
			v.key.num[level+1] = sckt.reverse_ref(g.fot.at(i).second);
			std::cerr << "Descending to level " << level+1 << " gate " << g.fot.at(i).second << "\n";
			descendSegment(ckt, sckt, ckt.at(g.fot.at(i).second), level+1, g.fot.at(i).second, v, segs);
		}
	} else if (ckt.at(v.key.num[0]).typ == INPT) { 
		for (unsigned int j = level+1; j < N; j++) { v.key.num[j] = -1; }
		segs->push_back(v);
	}
}

template<int N, class T>
inline void descendSegment(const Circuit& ckt, const NODEC& g, const int& level, segment<N,T> v, std::vector<segment<N,T> >* segs) {
	// every call represents another level
	if (level == N-1) {
		segs->push_back(v);
	} else if (g.po != true) {
		// recurse to next level
		for (unsigned int i = 0; i < g.nfo; i++) {
			v.key.num[level+1] = g.fot.at(i).second;
			descendSegment(ckt, ckt.at(g.fot.at(i).second), level+1, v, segs);
		}
	} else if (ckt.at(v.key.num[0]).typ == INPT) { 
		for (unsigned int j = level+1; j < N; j++) { v.key.num[j] = -1; }
		segs->push_back(v);
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
				descendSegment(ckt, ckt.at(gate.fot.at(j).second), 1, tmp, &segs);
			}
		} else {
			segs.push_back(tmp);
		}
	}
	// done copying 
	segment<N, T> tmp;
	tmp.pattern.x = -1;
	tmp.pattern.y = -1;
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

// Extra rule for segment, has to be in the same segment in order to get added.
template <int N, class T>
void generateSegmentList(segment<N,T>** seglist, const Circuit& ckt, const SubCkt& sckt) {
	std::vector<segment<N,T> > segs;
	// Start at all nodes.
	for (int gid = 0; gid < sckt.size(); gid++) {
		const NODEC& gate = sckt[gid];
		segment<N, T> tmp;
		tmp.key.num[0] = gid;
		if (N > 1) {
			for (unsigned int j = 0; j < gate.fot.size(); j++) {
				if (sckt.in(gate.fot.at(j).second) < 0) continue;
				tmp.key.num[1] = sckt.reverse_ref(gate.fot.at(j).second);
				descendSegment(ckt, sckt, ckt.at(gate.fot.at(j).second), 1, gate.fot.at(j).second, tmp, &segs);
			}
		} else {
			segs.push_back(tmp);
		}
	}
	// done copying 
	segment<N, T> tmp;
	tmp.pattern.x = -1;
	tmp.pattern.y = -1;
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
#ifdef __CUDACC__
template <int N, class T>
 void generateDcSegmentList(dc_segment<N,T>& seglist, const Circuit& ckt, const T ival) {
	std::vector<segment<N,T> > segs;
	const size_t keypitch = N*sizeof(int32_t);
	// Start at all nodes.
	for (unsigned int gid = 0; gid < ckt.size(); gid++) {
		const NODEC& gate = ckt.at(gid);
		segment<N, T> tmp;
		for (int i = 0; i < N; i++) {
			tmp.key.num[i] = ckt.size();
		}
		tmp.key.num[0] = gid;
		if (N > 1) {
			for (unsigned int j = 0; j < gate.fot.size(); j++) {
				tmp.key.num[1] = gate.fot.at(j).second;
				descendSegment(ckt, ckt.at(gate.fot.at(j).second), 1, gate.fot.at(j).second, tmp, &segs);
			}
		} else {
			segs.push_back(tmp);
		}
	}
	/*
	for (typename std::vector<segment<N,T> >::iterator it = segs.begin(); it < segs.end(); ++it) {

		std::cout << "Segment Added: sid " << std::distance(segs.begin(),it) << " (";
		for (int j = 0; j < N; j++) {
			std::cout << it->key.num[j] << ": " << ckt.at(it->key.num[j]).name;
			if (j != N-1) 
				std::cout << ",";
		}
		std::cout << ")\n";
	}
	*/
	// done generaing segment list, now copy to GPU
	seglist.height = segs.size();
	std::cerr << "Copying " << seglist.height << " segments to GPU.\n";

	cudaMallocPitch(&(seglist.key), &(seglist.pitch), keypitch, segs.size());
	cudaMalloc(&(seglist.pattern), sizeof(T)*segs.size());

	int32_t* key = new int32_t[segs.size()*N];
	int32_t* key2 = new int32_t[segs.size()*N];
	for (unsigned int i = 0; i < segs.size(); i++) {
		for (unsigned int j = 0; j < N; j++) {
			REF2D(key, keypitch, i, j) = -1;
		}
	}
	T* pattern = new T[segs.size()];

	for (unsigned int i = 0; i < segs.size(); i++) {
		for (int j = 0; j < N; j++) { 
			REF2D(key, keypitch, i, j) = segs[i].key.num[j];
		}
		pattern[i] = ival;
	}
	cudaMemcpy(seglist.pattern, pattern, seglist.height*sizeof(T), cudaMemcpyHostToDevice);
	HandleCUDAError(cudaGetLastError(),__FILE__,__LINE__);
	cudaMemcpy2D(seglist.key, seglist.pitch, key, keypitch, keypitch, segs.size(), cudaMemcpyHostToDevice);

	cudaMemcpy2D(key2, keypitch, seglist.key, seglist.pitch, keypitch, segs.size(), cudaMemcpyDeviceToHost);
	for (unsigned int i = 0; i < segs.size(); i++) {
		for (int j = 0; j < N; j++) { 
//			std::cerr << "Asserting " << REF2D(key, keypitch, i, j) << " == " <<  REF2D(key2, keypitch, i, j) << "\n";
			assert( REF2D(key, keypitch, i, j) ==REF2D(key2, keypitch, i, j) );
		}
	}
	HandleCUDAError(cudaGetLastError(),__FILE__,__LINE__);
	delete [] pattern;
	delete [] key;
	
}
#endif

template<int N, class T>
void displaySegmentList(segment<N,T>* seglist, const Circuit& ckt) {
	int i = 0;
	while (seglist[i].key.num[0] < ckt.size()) {
		std::cout << "Sid " << i << ": (";
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
