#CUDD
CUDD:=./cudd
CUDDLIBS=obj dddmp cudd epd mtr util st 
CUDDLIB_SEARCH=$(foreach libs,$(CUDDLIBS),-L$(CUDD)/$(libs)) 
CUDDLIB_FLAGS=$(foreach libs,$(CUDDLIBS),-l$(libs))
CUDDFLAGS:=-I$(CUDD)/include


# CPPUTEST 
CPPUTEST_FLAGS:=-I$(CPPUTEST_HOME)/include 
CPPUTEST_LIBS:=-lCppUTest -lCppUTestExt

LIBS:=$(CUDDLIB_FLAGS) -lm -lz
#objs=ckt.o cpudata.o iscas.o node.o sort.o subckt.o utility.o vectors.o mergestate.o gzstream.o
objs=logic.o circuit.o bdd_circuit.o

gobjs=gutility.o gpuckt.o gpudata.o g_subckt.o lookup.o g_vectors.o 
CXX=clang++
CC=clang

CXXFLAGS+= -O3 -mtune=native $(shell $$CXXFLAGS) -g -Wall -std=c++11 -march=native -fopenmp $(CPPUTEST_FLAGS) $(CUDDFLAGS)
LDFLAGS:=$(CUDDLIB_SEARCH)

TEST=$(foreach test,$(objs:.o=_test.cpp) AllTests.cpp,tests/${test})
TEST_H=$(objs:.o=.h)


ifndef CUDA_DIR
	CUDA_DIR=/opt/net/apps/cuda-5.0
endif
ifndef GPCXX
	GPCXX=${CUDA_DIR}/bin/nvcc
endif
ifndef NVCFLAGS 
	NVCFLAGS=-arch=sm_20 -I${CUDA_DIR}/include -ccbin g++-4.6
	CFLAGS:=-DCPU
endif
.SUFFIXES:
.SUFFIXES: .o .cu .cpp

.cpp.o: $(objs:.o=.cpp) $(objs:.o=.h)
	$(CXX) -c $(CFLAGS) $(CXXFLAGS)  -o $@ $<

.cu.o: 
	$(GPCXX) -c -dc $(NVCFLAGS) -o $@ $<

AllTests.o: $(TEST)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CPPUTEST_FLAGS) -c $^ -o $@

tests/logic_cpp.o: tests/logic_cpp.cpp logic.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CPPUTEST_FLAGS) -L${CPPUTEST_HOME}/lib $^ $(LIBS) $(CPPUTEST_LIBS) -o $^ 

all: $(objs) 
gpu: $(gobjs)
libs: libcktutil.a

libcktutil.a: subckt.o ckt.o node.o utility.o iscas.o gzstream.o vectors.o
	ar rv $@ $?
	
testsuite: $(subst explore.o,,$(objs)) $(TEST:.cpp=.o)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CPPUTEST_FLAGS) $(LDFLAGS) -L${CPPUTEST_HOME}/lib $^ $(LIBS) $(CPPUTEST_LIBS) -o $@ 
clean:
	rm -f *.o *.a tests/*.o testsuite

test: testsuite
	./$<
