#ifndef CPUDATA_H
#define CPUDATA_H

#include "array2d.h"
#include "errors.h"
#include "defines.h"
#include "array2d.h"
#include "utility.cuh"
#include <iostream> // included for debugging
#include <utility>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <stdint.h>
class CPU_Data {
    protected:
		std::vector<ARRAY2D<uint8_t> >* _data; // variable size CPU memory space.
		size_t _width, _height;
		size_t _current; // current chunk on GPU.
    public:
		size_t height() const { return this->_height;}
		size_t width() const { return this->_width;}
		size_t size() const { return this->_data->size();}
		int initialize(size_t, size_t);
        CPU_Data();
        CPU_Data(size_t, size_t);
        ~CPU_Data();
        std::string debug() const;
		int current() const;
		std::string print() const; 
		ARRAY2D<uint8_t> cpu(int ref) const { return this->_data->at(ref); }
		ARRAY2D<uint8_t> cpu() const { return cpu(this->_current);} // gets the CPU value for current;
};

#endif // CPUDATA_H
