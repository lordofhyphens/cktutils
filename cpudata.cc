#include "cpudata.h"
#include "utility.cuh"
/*
static void delete_data(const ARRAY2D<char>& item) {
	delete item.data;
}
*/
CPU_Data::CPU_Data() {
	this->_data = new std::vector<ARRAY2D<uint8_t> >();
    this->_width = 0;
    this->_height =0;
}
CPU_Data::CPU_Data(size_t rows, size_t columns) {
	this->_data = new std::vector<ARRAY2D<uint8_t> >();
	this->initialize(rows, columns);
}

CPU_Data::~CPU_Data() {
	for (std::vector<ARRAY2D<uint8_t> >::iterator i = _data->begin(); i < _data->end(); i++) {
		delete i->data;
	}
	delete this->_data;
}


int CPU_Data::initialize(size_t in_columns, size_t in_rows) {
    this->_data->push_back(ARRAY2D<uint8_t>(in_rows, in_columns, sizeof(uint8_t)*in_columns));
	this->_current = 0;
    this->_width = in_rows;
	this->_height = in_columns;
    return ERR_NONE;
}
std::string CPU_Data::print() const {
	std::stringstream st; 
	for (unsigned int r = 0;r < this->cpu().width; r++) {
		for (unsigned int i = 0; i < this->cpu().height; i++) {
			uint8_t z = REF2D(this->cpu().data, this->cpu().pitch, r, i);
			st << (int)z;
		}
		st << std::endl;
	}
	return st.str();
}
std::string CPU_Data::debug() const {
	std::stringstream st; 
	st << "CPU DATA,width="<<this->width()<<",height="<< this->height()<< ",chunks="<<this->_data->size()<<",current="<<this->_current << std::endl;
	return st.str();
}
