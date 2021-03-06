#include "gpuckt.h"
#include <cuda.h>

uint32_t GPU_Circuit::max_offset() const {
	return this->_max_offset;
}
GPUNODE* GPU_Circuit::gpu() const {
	return this->_gpu_graph;
}
GPU_Circuit::GPU_Circuit() {
	this->_offset = NULL;
	this->_gpu_graph = NULL;
}
GPU_Circuit::~GPU_Circuit() {
	// Clean up GPU memory afterwards.
	if (this->_offset != NULL) {
		cudaFree(this->_offset);
		this->_offset = NULL;
	}
	if (this->_gpu_graph != NULL) {
		cudaFree(this->_gpu_graph);
		this->_gpu_graph = NULL;
	}

}
// Copy the circuit representation to the GPU.
void GPU_Circuit::copy() {
	std::vector<NODEC>* g = this->graph;
	uint32_t *offsets;
	GPUNODE *ggraph = (GPUNODE*)malloc(sizeof(GPUNODE)*g->size());
	uint32_t off = 0;
	uint32_t maxoff = 0;
	for (std::vector<NODEC>::iterator i = g->begin(); i < g->end(); i++) {
		if (i->typ != UNKN) {
			maxoff += (i->nfi + i->nfo);
		}
	}
	offsets = (uint32_t*)malloc(sizeof(uint32_t)*maxoff); 
	for (uint32_t i = 0; i < g->size(); i++) {
		ggraph[i].type = graph->at(i).typ;
		ggraph[i].nfi = graph->at(i).fin.size();
		ggraph[i].nfo = graph->at(i).fot.size();
		ggraph[i].po = graph->at(i).po;
		ggraph[i].level = graph->at(i).level;
		ggraph[i].scratch = graph->at(i).scratch; // copy the scratchpad index
		if (graph->at(i).typ == 0) {
			std::cout << graph->at(i).name << " is type 0?"<<std::endl;
			continue;
		}
		ggraph[i].offset = off;
		// position of a particular node
		for (std::vector<std::pair<std::string,unsigned int> >::iterator fins = graph->at(i).fin.begin(); fins < graph->at(i).fin.end();fins++) {
			offsets[off] = fins->second;//this->id(*fins);
			off++;
		}
		for (std::vector<std::pair<std::string,unsigned int> >::iterator fots = graph->at(i).fot.begin(); fots < graph->at(i).fot.end();fots++) {
			offsets[off] = fots->second;//this->id(*fots);
			off++;
		}
	}
	this->_max_offset = off;
	cudaMalloc(&(this->_gpu_graph), sizeof(GPUNODE)*(g->size()));
	cudaMalloc(&(this->_offset), sizeof(uint32_t)*off);
	cudaMemcpy(this->_gpu_graph, ggraph, sizeof(GPUNODE)*(g->size()),cudaMemcpyHostToDevice);
	cudaMemcpy(this->_offset, offsets, sizeof(uint32_t)*off,cudaMemcpyHostToDevice);
	free(ggraph);
	free(offsets);
}

uint32_t GPU_Circuit::id(std::string name) const {
	return std::count_if(this->graph->begin(), find(this->graph->begin(),this->graph->end(),name), Yes<NODEC>);
}
