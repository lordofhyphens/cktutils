#ifndef CKT_H
#define CKT_H

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <queue>
#include <list>
#include <algorithm>
#include <cassert>
#include <utility>
#include "defines.h"
#include <stdint.h>

#define FBENCH 1 // iscas89 BENCH format.
// NODE TYPE CONSTANTS 
#define UNKN 0				// unknown node
#define INPT 1				// Primary Input
#define AND  2				// AND 
#define NAND 3				// NAND 
#define OR   4				// OR 
#define NOR  5				// NOR 
#define XOR  6				// XOR 
#define XNOR 7				// XNOR 
#define BUFF 8				// BUFFER 
#define NOT  9				// INVERTER 
#define FROM 10				// STEM BRANCH
#define DFF 11				// Dflipflop

struct NODEC {
	std::string name;
	char typ;
	unsigned int nfi, nfo;
	uint32_t level;
	int32_t scratch; // memory scratchpad position.
	int cur_fo;
	bool po, placed;
	std::string finlist;
	std::vector<std::pair<std::string, uint32_t > > fin;
	std::vector<std::pair<std::string, uint32_t > > fot;
	NODEC() { name = "", typ = 0, nfi = 0, nfo = 0, po = false, finlist="";}
	NODEC(std::string);
	NODEC(std::string, int type);
	NODEC(std::string id, std::string type, int nfi, std::string finlist);
	bool operator==(const std::string& other) const;
	bool operator==(const NODEC& other) const;
	bool operator<(const NODEC& other) const;
	bool operator>(const NODEC& other) const;
	bool operator<=(const NODEC& other) const;
	bool operator>=(const NODEC& other) const;
	private:
		void initialize(std::string id, int type, int nfi, int nfo, bool po, std::string finlist);
		void initialize(std::string id, std::string type, int nfi, int nfo, bool po, std::string finlist);
		void load(std::string attr);
};

bool scratch_compare(const NODEC& a, const NODEC& b);
class Circuit {
	protected:
		std::vector<NODEC>* graph;
		std::string name;
		void levelize();
		void mark_lines();
		unsigned int _levels;
		void annotate(std::vector<NODEC>*);
	public:
		Circuit();
		Circuit(int type, const char* benchfile) {
			this->graph = new std::vector<NODEC>();
			this->_levels = 1;
			if (type == FBENCH)
				this->read_bench(benchfile);
		}
		~Circuit();
		bool nodelevel(unsigned int n, unsigned int m) const;
		void read_bench(const char* benchfile, const char* ext = "");
		void print() const;
		inline NODEC& at(int node) const { return this->graph->at(node);}
		inline size_t levels() const { return this->_levels;}
		size_t max_level_pair();
		size_t out_of_level_nodes(size_t, size_t);
		size_t max_out_of_level_nodes();
		void compute_scratchpad(); // calculates the scratchpad
		// Assuming that the scratchpad has been computed, gives the largest ID required for scratchpad memory.
		inline unsigned int max_scratchpad() { 
			return std::max_element(graph->begin(), graph->end(), scratch_compare)->scratch; 
		}
		unsigned int levelsize(unsigned int) const;
		size_t size() const { return this->graph->size();}
		void save(const char*); // save a copy of the circuit in its current levelized form
		void load(const char* memfile); // load a circuit that has been levelized.
		void load(const char* memfile, const char* ext_id);
		void reannotate();
};

std::ostream& operator<<(std::ostream& outstream, const NODEC& node);
bool isPlaced(const NODEC& node);
bool isInLevel(const NODEC& node, unsigned int N);

unsigned int countInLevel(std::vector<NODEC>& v, unsigned int level);
bool isUnknown(const NODEC& node) ;
bool isDuplicate(const NODEC& a, const NODEC& b);
bool nameSort(const NODEC& a, const NODEC& b);

struct StringFinder
{
  StringFinder(const std::string & st) : s(st) { }
  const std::string s;
  bool operator()(const std::pair<std::string, int>& lhs) const { return lhs.first == s; }
};


template <class T>
bool from_string(T& t, const std::string& s, std::ios_base& (*f)(std::ios_base&))
{
  std::istringstream iss(s);
  return !(iss >> f >> t).fail();
}
template<class Iter, class T>
Iter binary_find(Iter begin, Iter end, T val)
{
    // Finds the lower bound in at most log(last - first) + 1 comparisons
    Iter i = std::lower_bound(begin, end, val);

    if (i != end && *i == val)
        return i; // found
    else
        return end; // not found
}
#endif //CKT_H
