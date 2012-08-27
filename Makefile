objs: ckt.o cpudata.o iscas.o node.o sort.o subckt.o utility.o vectors.o mergestate.o
all: $(objs) 

.cc.o: $(objs:.o=.cc) $(objs:.o=.h)
	$(CXX) -c $(CFLAGS) $(CPPFLAGS) -o $@ $<

clean:
	rm -f *.o
