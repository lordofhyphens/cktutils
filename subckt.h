#ifndef SUBCKT_H
#define SUBCKT_H
#include <vector>
#include <algorithm>
#include <string>
#include <iterator>
#include "ckt.h"
class SubCkt { 
	private:
		const Circuit& _ckt;
		int _ref_node;
		int *_flat;
		int *_gpu;
		int* flat(); // get the flat array representation, allocates memory.
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
		void add(const Circuit& ckt, const int& n) { _subckt->push_back(n); }
		void copy();
		void clear();
		int* gpu() { return this->_gpu;}
		int in(unsigned int) const;
		inline int levels() const { return _levels->size() - 1; }
		inline int levelsize(const unsigned int n) const { return ( n < _levels->size() ? _levels->at(n) : 0); }
		inline int at(const unsigned int n) const { return _subckt->at(n); }

		int size() const { return this->_subckt->size();}
		std::vector<int>& subckt() const { return *_subckt; }
		const SubCkt operator/(const SubCkt& b) const; // intersection
		SubCkt& operator=(const SubCkt&);
		void grow(unsigned int node);
};

#endif // SUBCKT_H
