objs=ckt.o cpudata.o iscas.o node.o sort.o subckt.o utility.o vectors.o mergestate.o
gobjs=gutility.o gpuckt.o gpudata.o g_subckt.o
NVCC=nvcc
NVCFLAGS=-arch=sm_20 -I/opt/net/apps/cuda/include -ccbin g++-4.4
.SUFFIXES:
.SUFFIXES: .o .cu .cc
.cc.o: $(objs:.o=.cc) $(objs:.o=.h)
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) -o $@ $<
.cu.o: 
	$(NVCC) -c $(NVCFLAGS) -o $@ $<

all: $(objs) 
gpu: $(gobjs)
clean:
	rm -f *.o
