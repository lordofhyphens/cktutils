#include "subckt.h"
#ifndef CPU

void SubCkt::copy() {
	cudaError_t resp;
	int32_t* flat = new int32_t[_subckt->size()+1];
	for (unsigned int i = 0; i < _subckt->size(); i++)
		flat[i] = _subckt->at(i);
	flat[_subckt->size()] = -1;

	resp = cudaMalloc(&_gpu, sizeof(int)*(size()+1));
	if (resp != cudaSuccess) { 
		DPRINT("Error allocating memory for GPU copy of subckt\n");
	}
	resp = cudaMemcpy(_gpu, flat, sizeof(int)*(size()+1), cudaMemcpyHostToDevice);
	if (resp != cudaSuccess) { 
		DPRINT("Error copying GPU copy of subckt to GPU\n");
	}
	delete [] flat;
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
