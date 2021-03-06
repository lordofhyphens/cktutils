#include <cuda.h>
#include "gpudata.h"
#include "defines.h"
void HandleGPUDataError( cudaError_t err, const char *file, uint32_t line ) {
    if (err != cudaSuccess) {
		throw std::runtime_error(std::string(cudaGetErrorString( err )));
    }
}
#define HANDLE_ERROR( err ) (HandleGPUDataError( err, __FILE__, __LINE__ ))
GPU_Data::GPU_Data() {
	this->_gpu = new ARRAY2D<uint8_t>();
	this->_block_size = 0;
}
GPU_Data::GPU_Data(size_t rows, size_t columns) {
	this->_gpu = new ARRAY2D<uint8_t>();
	this->initialize(rows+1, columns, rows+1);
}
GPU_Data::GPU_Data(size_t rows, size_t columns, uint32_t blockwidth) {
	this->_gpu = NULL;
	this->initialize(rows+1, columns, blockwidth+1);
}
GPU_Data::~GPU_Data() {
	if (this->_gpu->data != NULL) {
		cudaFree(this->_gpu->data);
	}
	delete _gpu;
} 
ARRAY2D<uint8_t> GPU_Data::gpu(uint32_t ref, bool coherent) {
	if (ref == this->_current) {
		if (this->_gpu->data == NULL) { this->copy(ref); }
		return *(this->_gpu);
	}
	DPRINT("%s:%d - Switching to chunk %d\n", __FILE__, __LINE__, ref);
	uint32_t tmp = this->_current;
	uint32_t err; 
	try {
		err = this->copy(ref, coherent);
	} catch (std::out_of_range& oor) { 
		// handling the problem by returning NULL and ensuring that _current is not changed.
		this->_current = tmp;
		DPRINT("Out of range in swap.\n");
		return ARRAY2D<uint8_t>(NULL,0,0,0);
	}
	if (err != ERR_NONE) {
		DPRINT("Unknown error in swap.\n");
		return ARRAY2D<uint8_t>(NULL,0,0,0);
	}
	return *(this->_gpu);
}

// total size in columns, rows. 
uint32_t GPU_Data::initialize(size_t in_columns, size_t in_rows, uint32_t block_width) {
	rows = in_rows;
	columns = in_columns;
	bl_width = block_width;
	cudaError_t err = cudaSuccess;
	uint32_t chunks = (in_columns / block_width) + ((in_columns % block_width) > 0);

	this->_gpu = new ARRAY2D<uint8_t>(NULL, in_rows, block_width, sizeof(uint8_t)*block_width);
	err = cudaMallocPitch(&(this->_gpu->data), &(this->_gpu->pitch), sizeof(uint8_t)*this->_gpu->width, in_rows);
	if (err != cudaSuccess) { 
		DPRINT("Failed to allocate memory.\n");
		HANDLE_ERROR(err);
		return err;
	}
	uint32_t rem_columns = in_columns;
	for (uint32_t i = 0; i < chunks;i++) {
		uint8_t* data = new uint8_t[in_rows*sizeof(uint8_t)*min(block_width,rem_columns)];
		this->_data->push_back(ARRAY2D<uint8_t>(data, in_rows, min(block_width, rem_columns),sizeof(uint8_t)*min(block_width,rem_columns)));
		assert(this->_data->back().data != NULL);
		if (rem_columns > block_width) {
			rem_columns -= block_width;
		}
		
	}
	assert(_gpu->data != NULL);
	this->_current = 0;
	this->_block_size = block_width;
	this->_width = in_columns;
	this->_height = in_rows;
	return ERR_NONE;
}

// performs a swap-out of GPU memory. 
uint32_t GPU_Data::copy(uint32_t ref, bool coherent) {
	uint32_t error;
	bool reallocate = false;
//	DPRINT("%s:%d - Copying chunk %d from GPU, %d to GPU.\n",__FILE__,__LINE__,_current, ref);
	if (this->_gpu->data == NULL) { // Re-allocate GPU memory.
		error = cudaMallocPitch(&(this->_gpu->data), &(this->_gpu->pitch), sizeof(uint8_t)*this->_gpu->width, rows);
		if (error != cudaSuccess) { DPRINT("Failed to allocate memory."); }
		reallocate = true;
	}
	ARRAY2D<uint8_t>* cpu = &(this->_data->at(this->_current));
	ARRAY2D<uint8_t>* gpu = this->_gpu;
//	DPRINT("%s:%d - Memcpy from GPU\n", __FILE__,__LINE__);
	if (coherent && !reallocate) 
		cudaMemcpy2D(cpu->data, cpu->pitch, gpu->data, gpu->pitch, cpu->width * sizeof(uint8_t), cpu->height, cudaMemcpyDeviceToHost);
//	DPRINT("%s:%d - Memcpy from GPU\n", __FILE__,__LINE__);
	error = cudaGetLastError();
//	DPRINT("%s:%d - Getting reference to CPU", __FILE__,__LINE__);
	cpu = &(this->_data->at(ref));
	cudaMemcpy2D(gpu->data, gpu->pitch, cpu->data, cpu->pitch, cpu->width * sizeof(uint8_t), cpu->height, cudaMemcpyHostToDevice);
	gpu->width = cpu->width;
	gpu->height = cpu->height;
	error = cudaGetLastError();
	this->_current = ref;
//	DPRINT("%s:%d - Finished copy.\n", __FILE__,__LINE__);
	if (error != cudaSuccess)
		return ERR_NONE;
	return error;
}
// copy the GPU memory space back to the CPU.
void GPU_Data::unload() {
	ARRAY2D<uint8_t>* cpu = &(this->_data->at(this->_current));
	ARRAY2D<uint8_t>* gpu = this->_gpu;
	if (gpu != NULL)
		cudaMemcpy2D(cpu->data, cpu->pitch, gpu->data, gpu->pitch, cpu->width * sizeof(uint8_t), cpu->height, cudaMemcpyDeviceToHost);
}
uint32_t GPU_Data::refresh() {
	uint32_t error;
	if (this->_gpu->data == NULL) { // Re-allocate GPU memory.
		error = cudaMallocPitch(&(this->_gpu->data), &(this->_gpu->pitch), sizeof(uint8_t)*this->_gpu->width, rows);
		if (error != cudaSuccess) { DPRINT("Failed to allocate memory."); }
	}
	ARRAY2D<uint8_t>* cpu = &(this->_data->at(this->_current));
	ARRAY2D<uint8_t>* gpu = this->_gpu;
	cudaMemcpy2D(gpu->data, gpu->pitch, cpu->data, cpu->pitch, cpu->width*sizeof(uint8_t), cpu->height, cudaMemcpyHostToDevice);
	error = cudaGetLastError();
	if (error != cudaSuccess)
		return ERR_NONE;
	return error;
}
void GPU_Data::clear() {
	if (this->_gpu->data != NULL) { cudaFree(this->_gpu->data); }
	this->_gpu->data = NULL;
} // deletes copy of data on GPU
std::string GPU_Data::debug() {
	std::stringstream st; 
	if (_data != NULL && _gpu != NULL) {
		st << "GPU DATA,width="<<this->width();
		st 	<<",height="<< this->height();
		st	<< ",pitch=" << this->gpu().pitch;
		st	<<",blocksize=" << this->_block_size ;
		st	<< ",chunks=" <<this->_data->size() ;
		st	<<",current="<<this->_current << std::endl;;
	} else {
		st << "GPU DATA: Internal data item is null." << std::endl;
	}
	return st.str();
}

void debugDataOutput(GPU_Data& results, std::string outfile = "simdata.log") {
#ifndef NDEBUG
	uint8_t *lvalues;
	std::ofstream ofile(outfile.c_str());

	lvalues = (uint8_t*)malloc((results.gpu().height)*results.gpu().pitch);
	cudaError_t err = cudaMemcpy2D(lvalues,results.gpu().pitch,results.gpu().data,results.gpu().pitch,results.gpu().width,results.gpu().height,cudaMemcpyDeviceToHost);
	assert (err == cudaSuccess);
	for (uint32_t r = 0;r < results.width(); r++) {
		for (uint32_t i = 0; i < results.height(); i++) {
			uint8_t z = REF2D( lvalues, results.gpu().pitch, r, i);
			ofile << (uint32_t)z;
		}
		ofile << std::endl;
	}
	free(lvalues);
	ofile.close();
#endif
}

void debugDataOutput(ARRAY2D<uint8_t> results, std::string outfile = "simdata.log") {
#ifndef NDEBUG
	uint8_t *lvalues;
	std::ofstream ofile(outfile.c_str());

	lvalues = (uint8_t*)malloc((results.height)*results.pitch);
	cudaError_t err = cudaMemcpy2D(lvalues,results.pitch,results.data,results.pitch,results.width,results.height,cudaMemcpyDeviceToHost);
	assert (err == cudaSuccess);
	for (uint32_t r = 0;r < results.width; r++) {
		for (uint32_t i = 0; i < results.height; i++) {
			uint8_t z = REF2D( lvalues, results.pitch, r, i);
			ofile << (uint32_t)z;
		}
		ofile << std::endl;
	}
	free(lvalues);
	ofile.close();
#endif
}
HOST_DEVICE coalesce_t vectAND(coalesce_t a, uint32_t b) { a.rows[0] &= b; a.rows[1]&=b;a.rows[2]&=b;a.rows[3]&=b; return a;}
