#include "defines.h"
#include "errors.h"
#include "gpudata.h"
#include <cassert>
#include <string>
#include <sstream>
#include <fstream>
#include <utility>
#include <vector>

std::pair<size_t, size_t> get_vector_dim(const char fvec[]);
int read_vectors(GPU_Data& pack, const char* fvec, int chunksize);
int read_vectors(CPU_Data& pack, const char* fvec, int chunksize);
int read_vectors(std::vector<std::vector<bool> >& vec, const char* fvec);
