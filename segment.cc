#include "segment.cuh"
#include "ckt.h"
#include <vector>

// Rules for segment generation:
// Start at all nodes.
template <int N>
void generateSegmentList(segment<N>** seglist, const Circuit& ckt) {
	std::vector<segment<N> > segs;
	for (int gid = 0; gid < ckt.size(); gid++) {
		int nfo[N];
		int level = 0;
		segment<N> tmp;
		#pragma unroll
		for (int i = 0; i < N; i++) { nfo[i] = 0; }
		tmp.key.num[0] = gid; // start of current segments
		bool isPI = (ckt.at(gid).type == INPT);

		tmp.key.num[level+1] = ckt.at(gid).nfos.at(nfo[level]);

		// go to child node

	}
	
};

