#include "vectors.h"
#include "gzstream.h"
#include <cstring>
#include <vector>
#include <map>
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
	assert((unsigned)lines == pack.width());
	std::cerr << " All vectors have been read." << std::endl;
	delete tfile;
	pack.unload();
	pack.refresh();
	return ERR_NONE;
}
#endif 
