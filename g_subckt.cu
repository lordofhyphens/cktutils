#include "subckt.h"
#ifndef CPU
void SubCkt::copy() {
	int *test = flat();
	cudaMalloc(&_gpu, sizeof(int)*(size()+1));
	cudaMemcpy(_gpu, test, sizeof(int)*(size()+1), cudaMemcpyHostToDevice);
	delete test;
}

void SubCkt::clear() {
	if (_gpu != NULL) 
		cudaFree(_gpu);
	_gpu = NULL;
}
SubCkt::~SubCkt() {
	if (_gpu != NULL) 
		cudaFree(_gpu);
	delete _levels;
	delete _subckt;
}

#endif // don't compile this is CPU is defined
