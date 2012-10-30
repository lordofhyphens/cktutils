#include "subckt.h"
#include <algorithm>
#include <parallel/algorithm>

SubCkt::SubCkt(const Circuit& ckt) : _ckt(ckt) {
	_flat = NULL;
	_levels = new std::vector<int>();
	_subckt = new std::vector<int>();
	for (uint32_t  i = 0; i <= _ckt.levels(); i++) {
		_levels->push_back(0); // add more nodes to the levels vector if necessary
	}
}
SubCkt::SubCkt(const Circuit& ckt, unsigned int node) : _ckt(ckt) {
	_flat = NULL;
	_levels = new std::vector<int>();
	_subckt = new std::vector<int>();
	for (uint32_t  i = 0; i <= _ckt.levels(); i++) {
		_levels->push_back(0); // add more nodes to the levels vector if necessary
	}
	grow(node);
}
bool SubCkt::operator<(const SubCkt& other) const {
	return (*this < other.size());
}
bool SubCkt::operator<(int other) const {
	return size() < other;
}
std::string SubCkt::save() const {
	// dump the subckt to a space-separated file, followed by a newline.
	std::stringstream ofile;
	if (_subckt->size() > 0) {
		ofile << _subckt->size() << ":";
		for (unsigned int i = 0; i < this->_subckt->size(); i++) {
			ofile << _subckt->at(i) << " ";
		}
	}
	return ofile.str();
}

void SubCkt::load(const std::string& memfile) {
	//std::cerr << "Subckt: "<< memfile << std::endl;
	std::string data = memfile.substr(memfile.find_first_of(':') + 1);
	std::stringstream t(memfile.substr(memfile.find_first_of(':')));
	std::stringstream ifile(data);
	t >> _ref_node;
	int z;
	while(!ifile.eof()) { ifile >> z; add(z);}
	__gnu_parallel::sort(_subckt->begin(), _subckt->end());
	std::vector<int>::iterator it = unique(_subckt->begin(), _subckt->end());
	_subckt->resize(it - _subckt->begin());
	levelize();
	_ref_node = _subckt->at(0);
}
#ifdef CPU
SubCkt::~SubCkt() {
	delete _levels;
	delete _subckt;
}
#endif
void SubCkt::add(const Circuit& ckt, const int& n) {
	_subckt->push_back(n);
}
void SubCkt::levelize() {
	delete _levels;
	_levels = new std::vector<int>();
	for (uint32_t  i = 0; i <= _ckt.levels(); i++) {
		_levels->push_back(0); // add more nodes to the levels vector if necessary
	}
	for (uint32_t i = 0; i < _subckt->size(); i++) {
		_levels->at(_ckt.at(_subckt->at(i)).level) += 1;
	}
}
void SubCkt::grow(unsigned int node) {
	const NODEC& home_node = _ckt.at(node);
	// starting from the home node, traverse through its fanins and fanouts, adding them one-by-one.
	add(node);
	_ref_node = node;
	for (unsigned int i = 0; i < home_node.fin.size(); i++) {
		grow_recurse_back(home_node.fin.at(i).second);
	}
	for (unsigned int i = 0; i < home_node.fot.size(); i++) {
		grow_recurse_forward(home_node.fot.at(i).second);
	}
	__gnu_parallel::sort(_subckt->begin(),_subckt->end());
	_subckt->resize(std::unique(_subckt->begin(),_subckt->end()) - _subckt->begin());
}

void SubCkt::grow_recurse_back(unsigned int node) {
	const NODEC& home_node = _ckt.at(node);
	if (__gnu_parallel::find(_subckt->begin(),_subckt->end(), node) != _subckt->end())
		return;
	_subckt->push_back(node);
	for (unsigned int i = 0; i < home_node.fin.size(); i++) {
		grow_recurse_back(home_node.fin.at(i).second);
	}
}
void SubCkt::grow_recurse_forward(unsigned int node) {
	const NODEC& home_node = _ckt.at(node);
	if (__gnu_parallel::find(_subckt->begin(),_subckt->end(), node) != _subckt->end())
		return;
	_subckt->push_back(node);
	for (unsigned int i = 0; i < home_node.fot.size(); i++) {
		grow_recurse_forward(home_node.fot.at(i).second);
	}
}


const SubCkt SubCkt::operator/(const SubCkt& b) const {
	SubCkt sc(this->_ckt);

	for (std::vector<int>::iterator i = _subckt->begin(); i < _subckt->end(); i++) {
		for (std::vector<int>::iterator j = b.subckt().begin(); j <  b.subckt().end();j++) {
			if (*i == *j) { sc.add(*i); }
		}
	}
	return sc;
}
int SubCkt::in(unsigned int tgt) const {
	std::vector<int>::iterator is = __gnu_parallel::find(_subckt->begin(), _subckt->end(), tgt);
	if (is == _subckt->end()) {
		return -1;
	} else {
		return std::distance(_subckt->begin(), is);
	}
}
SubCkt::SubCkt(const SubCkt& other) : _ckt(other._ckt){
	_levels = new std::vector<int>();
	_subckt = new std::vector<int>();
	this->_ref_node = other._ref_node;
	this->_subckt->assign(other._subckt->begin(), other._subckt->end());
	this->_levels->assign(other._levels->begin(), other._levels->end());
}
SubCkt & SubCkt::operator=(const SubCkt & rhs) {
	if (this == &rhs)
		return *this;
	this->load(rhs.save());
	this->_ref_node = rhs._ref_node;

	return *this;
}
