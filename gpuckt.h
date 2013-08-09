#ifndef GPUCKT_H
#define GPUCKT_H

#include "ckt.h"
#include <utility>
#include <stdint.h>

typedef struct GPU_NODE_type {
	uint16_t nfi, nfo, level;
	uint8_t type, po;
	uint32_t offset;
	int32_t scratch;
} GPUNODE;
typedef struct GPUCKT_POD_T { GPUNODE* graph; uint32_t* fanout; } GPUCKT;
// subclass of Circuit to provide GPU-friendly representation of the circuit.
class GPU_Circuit : public Circuit { 
	private: 
		uint32_t* _offset;
		GPUNODE* _gpu_graph;
		uint32_t _max_offset;
		uint32_t id(std::string) const;
	public:
		uint32_t* offset() const { return this->_offset;}
		uint32_t max_offset() const;
		GPUNODE* gpu_graph() const;
		void copy();
		~GPU_Circuit();
		GPU_Circuit();
		GPUCKT POD() const { GPUCKT pod; pod.graph = gpu_graph(); pod.fanout = offset(); return pod; }
		
};

template <class T> bool Yes(const T& item) {
	return true;
}

inline const GPUCKT toPod(const GPU_Circuit& ckt) {
	GPUCKT tmp;
	tmp.graph = ckt.gpu_graph();
	tmp.offset = ckt.offset();
	return tmp;
}
#endif //GPUCKT_H
