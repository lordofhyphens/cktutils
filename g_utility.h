__device__ inline int find(int* ckt, int tgt) {
	int i = 0;
	while (ckt[i] != -1 && ckt[i] != tgt)
		i++;
	return i;
}

#define KEY_NOT_FOUND -1
__device__ inline int midpoint(int min, int max) {
	return (min + ((max-min)/2));
}
// Performs a reverse-lookup using binary search.
// src is the position in the *subcircuit* the current gate lies.
// limit is either 0 or the size of the subcircuit, indicates 
// whether or not looking for a PI or a PO reference.
// uses a delayed detection of equality to try to keep 
// all threads together
// ckt is the subcircuit
__device__ inline int bin_find(const int ckt[],const int src,const int tgt,const size_t limit) {
	int imin, imax;
	imin = (limit >= src)*(src) + (src >= limit)*(limit);
	imax = (limit >= src)*(limit) + (src >= limit)*(src);
	while (imin < imax) {
		int imid = midpoint(imin, imax);
		imin = (ckt[imid] < tgt)*(imid + 1) + (ckt[imid] >= tgt)*(imid);
		imax = (ckt[imid] < tgt)*(imax) + (ckt[imid] >= tgt)*(imid);
	}
	// check for equality
	return KEY_NOT_FOUND*(ckt[imin] != tgt) + imin*(ckt[imin]);

}

