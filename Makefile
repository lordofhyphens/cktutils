objs=ckt.o cpudata.o iscas.o node.o sort.o subckt.o utility.o vectors.o mergestate.o
gobjs=gutility.o gpuckt.o gpudata.o g_subckt.o
CXX=g++-4.7
ifndef CUDA_DIR
	CUDA_DIR=/opt/net/apps/cuda-5.0
endif
ifndef GPCXX
	GPCXX=${CUDA_DIR}/bin/nvcc
endif
ifndef NVCFLAGS 
	NVCFLAGS=-arch=sm_20 -I${CUDA_DIR}/include -ccbin g++-4.4
endif
.SUFFIXES:
.SUFFIXES: .o .cu .cc
.cc.o: $(objs:.o=.cc) $(objs:.o=.h)
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) -std=c++11 -march=native -fopenmp -o $@ $<
.cu.o: 
	$(GPCXX) -c $(NVCFLAGS) -o $@ $<

all: $(objs) 
gpu: $(gobjs)
clean:
	rm -f *.o
