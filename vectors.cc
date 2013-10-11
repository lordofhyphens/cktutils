#include "vectors.h"
#include "gzstream.h"
#include <cstring>
#include <vector>

/* Read a simple text file formatted with input patterns.
* This modifies the array in vecs, allocating it. 
* It returns the count of input patterns. All don'tcares 
* are set to '0'.
*/
std::pair<size_t, size_t> get_vector_dim(const char *fvec) {
	std::string str1;
	std::istream* tfile;
	if (strstr(fvec,".gz") == NULL) {
		std::cerr << " (uncompressed) ";
		tfile = new std::ifstream(fvec);
	} else {
		std::cerr << " (compressed) ";
		tfile = new igzstream(fvec);
	}

	if (!(tfile->good())) { 
		DPRINT("%s, %d: Failed to open file %s.\n", __FILE__, __LINE__, fvec);
		return std::make_pair(0,0); //early abort
	}
	size_t lines = 0;
	size_t inputs = 0;
	getline((*tfile),str1);
	while(getline((*tfile),str1)) {
		lines++;
		inputs = str1.size();
	}
	assert(lines > 0);
	assert(inputs > 0);
	delete tfile;
	return std::make_pair(lines, inputs);
}
#ifndef CPU
int read_vectors(GPU_Data& pack,const char* fvec, int chunksize, int height) {
	std::string str1;
	std::istream* tfile;
	if (strstr(fvec,".gz") == NULL) {
		tfile = new std::ifstream(fvec);
	} else {
		tfile = new igzstream(fvec);
	}
	int chunk = 0;
	int lines = 0;
	if (!*tfile) { 
		DPRINT("%s, %d: Failed to open file %s.\n", __FILE__, __LINE__, fvec);
	}
	while(getline(*tfile,str1)) {
		if (str1.find("#") != std::string::npos) 
			continue; // ignore comment lines
		// for every character in the string, 
		// determine the placement in the array, using
		// REF2D.
		//std::cout << str1 << std::endl;
		uint8_t* data = pack.cpu(chunk).data;
		#pragma omp parallel for shared(data)
		for (unsigned int j = 0; j < str1.size(); j++) { 
			REF2D( data,pack.cpu(chunk).pitch,lines, j) = ((str1[j] == '0') ? 0 : 1);
			//DPRINT("%2d ",REF2D( pack.cpu(chunk).data,pack.cpu().pitch,lines, j) );
		}
		lines++;
	//	if (lines > chunksize) {
	//		lines = 0;
	//		chunk++;
	//	}
	}
	assert(lines == pack.width());
	std::cerr << " All vectors have been read." << std::endl;
	delete tfile;
	pack.unload();
	pack.refresh();
	return ERR_NONE;
}
#endif 
int read_vectors(CPU_Data& pack,const char* fvec, int chunksize) {
	std::string str1;
	std::istream* tfile;
	if (strstr(fvec,".gz") == NULL) {
		tfile = new std::ifstream(fvec);
	} else {
		tfile = new igzstream(fvec);
	}
	int chunk = 0;
	int lines = 0;
	while(getline(*tfile,str1)) {
		if (str1.find("#") != std::string::npos) 
			continue; // ignore comment lines
		// for every character in the string, 
		// determine the placement in the array, using
		// REF2D.
		for (unsigned int j = 0; j < str1.size(); j++) { 
			REF2D( pack.cpu(chunk).data,pack.cpu(chunk).pitch,lines, j) = ((str1[j] == '0') ? 0 : 1);
		}
		lines++;
		if (lines > chunksize) {
			lines = 0;
			chunk++;
		}
	}
	std::cerr << " All vectors have been read." << std::endl;
	delete tfile;
	return ERR_NONE;
}

int read_vectors(std::vector<std::vector<bool> >& vec, const char* fvec) {
	std::string str1;
	std::ifstream tfile(fvec);
	int lines = 0;
	while(getline(tfile,str1)) {
		if (str1.find("#") != std::string::npos) 
			continue; // ignore comment lines
		std::vector<bool> z(str1.size(),0);
		// for every character in the string, 
		// determine the placement in the array
		for (unsigned int j = 0; j < str1.size(); j++) {
			z[j] = ((str1[j] == '0') ? 0 : 1);
		}
		lines++;
		vec.push_back(z);
	}
	std::cerr << " All vectors have been read." << std::endl;
	tfile.close();
	return ERR_NONE;
}
