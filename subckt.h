#ifndef SUBCKT_H
#define SUBCKT_H
#include <vector>
#include <algorithm>
#include <string>
#include <iterator>
#include "ckt.h"

/* SubCkt class
 *
 * Representation of a list of (nominally connected) nodes in a related
 * Circuit, with additional functions to create one from a single node.
 * Functions of note:
 * at() returns an integer index for given node w/r/t the related Circuit.
 */

class SubCkt { 
	private:
		const Circuit& _ckt;
		int _ref_node;
		int *_flat;
		int *_gpu;
		std::vector<int>* _levels;
		std::vector<int>* _subckt;
		void levelize();

		void grow_recurse_back(unsigned int node);
		void grow_recurse_forward(unsigned int node);
	public:	
		~SubCkt();
		std::string save() const;
		void load(const std::string& memfile);
		SubCkt(const Circuit& ckt);
		SubCkt(const SubCkt&);
		SubCkt(const Circuit& ckt, unsigned int node);
		void add(const int& n) { add(this->_ckt, n);}
		void add(const Circuit&, const int&);
		void copy(); // Flattens and copies the subckt to GPU memory.
		void clear(); // Deallocate the copy of the subckt in GPU memory.
		int in(unsigned int) const;
		inline int levels() const { return _levels->size() - 1; }
		inline int levelsize(const unsigned int n) const { return ( n < _levels->size() ? _levels->at(n) : 0); }
		// translates subckt position to ckt position
		inline int ref(const unsigned int n) const { return _subckt->at(n); } 
		// translates ckt position to subckt position
		inline int reverse_ref(const unsigned int n, const unsigned int g) const { 
			std::vector<int>::iterator a = std::find(_subckt->begin(), _subckt->begin()+g,n);
			if (a < _subckt->begin()+g)
				return std::distance(_subckt->begin(), a); 
			else 
				return size()+1;
		}
		inline int reverse_ref(const unsigned int n) const { 
			std::vector<int>::iterator a = std::find(_subckt->begin(), _subckt->end(),n);
			if (a < _subckt->end())
				return std::distance(_subckt->begin(), a); 
			else 
				return size()+1;
		}
		// Read-only copy of a NODEC.
		inline NODEC& operator[](const unsigned int n) const { return _ckt.at(ref(n)); }
		inline NODEC& at(const unsigned int n) const { return _ckt.at(ref(n)); }
		inline const Circuit& ckt() const { return _ckt; }
		bool operator<(const SubCkt&) const;
		bool operator<(const int) const;
		inline int* gpu() { return this->_gpu;}
		int size() const { return this->_subckt->size();}
		std::vector<int>& subckt() const { return *_subckt; }
		const SubCkt operator/(const SubCkt& b) const; // intersection
		SubCkt& operator=(const SubCkt&);
		int root() { return _ref_node;}
		void grow(unsigned int node);
};
struct GPU_SCKT_BATCH {
	int** sckts;
	size_t* sizes;
	GPU_SCKT_BATCH(int** sckt, size_t* size) { this->sckts = sckt; this->sizes = size; }
};
#ifndef CPU

#endif  // CPU 
#endif // SUBCKT_H
