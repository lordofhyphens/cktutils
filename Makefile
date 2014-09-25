objs=ckt.o cpudata.o iscas.o node.o sort.o subckt.o utility.o vectors.o mergestate.o gzstream.o
gobjs=gutility.o gpuckt.o gpudata.o g_subckt.o lookup.o 
CXX=g++
CC=gcc
ifndef CUDA_DIR
	CUDA_DIR=/opt/net/apps/cuda-5.0
endif
ifndef GPCXX
	GPCXX=${CUDA_DIR}/bin/nvcc
endif
ifndef NVCFLAGS 
	NVCFLAGS=-arch=sm_20 -I${CUDA_DIR}/include -ccbin g++-4.6
endif
.SUFFIXES:
.SUFFIXES: .o .cu .cc
.cc.o: $(objs:.o=.cc) $(objs:.o=.h)
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) -std=c++11 -march=native -fopenmp -o $@ $<

.cu.o: 
	$(GPCXX) -c -dc $(NVCFLAGS) -o $@ $<

all: $(objs) 
gpu: $(gobjs)
libs: libcktutil.a

libcktutil.a: ckt.o node.o utility.o iscas.o gzstream.o
	ar rv $@ $?
test: minimum_example.cc ckt.cc node.cc
	$(CXX) $(CFLAGS) -fopenmp -std=c++11 -o $@ minimum_example.cc ckt.cc node.cc
	
clean:
	rm -f *.o *.a
