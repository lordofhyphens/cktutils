#ifndef DEFINES_H
#define DEFINES_H
#define HASHLENGTH 6
#define UNINITIALIZED -1
#ifdef NDEBUG 
	#define DPRINT(...) 
#else
	#include <cstdio>
	#include <iomanip>
	#include <iostream>
	#include <fstream>
	#define DPRINT(...) if (verbose_flag) fprintf(stderr, __VA_ARGS__)
#endif // DPRINT definition
#define LOGEXEC 
#ifdef NDEBUG 
	#define GPRINT(A, ...) 
#else
	#define GPRINT(...) (if (tid == 0) { printf( __VA_ARGS__); })
#endif // GPRINT definition

#ifdef NTIMING
	#define TPRINT(...) 
#else
	#define TPRINT(...) fprintf(stderr, __VA_ARGS__)
#endif // TPRINT definition
#define HASH_COUNT 10
#define S0 0
#define S1 1
#define T0 2 // 1->0 transition, or "Transition to 0"
#define T1 3 // 0->1 transition, or "Transition to 1"

#define NOT_IN(A) ( A >= T0 )
#define AND_OUT(A,B) ((B >= T0)*(A != S0) + (B==S1)*(A==S1)*2 + (B==S1)*(A>=T0))
#define AND_IN(A,B) ((A==T0)*(B>S0)+(A==T1)*((B==S1)+(B==T1)))
#define OR_IN(A,B) ( (A>=T0)*((B==S0)+(B==T0))+(A==T1)*(B==T1) )
#define OR_OUT(A,B) ((A>=T0)*(B!=S1)+2*(B==S0)*(A==S0)+(A==S0)*(B>=T0)) 
#define XOR_IN(A,B) ( (A>=T0)*(B<=S1) )
#define XOR_OUT(A,B) ( (A>=T0)*(B<=S1) + (A<=S1)*(B>=T0) )


// utility macros to makke addressing a little easier.
// byte-based offsets, pitch is the width of the linear array in bytes.
//#define REF2D(TYPE,ARRAY,PITCH,X,Y) ( (((TYPE*)((char*)ARRAY + Y*PITCH))[X] ))
//#define ADDR2D(TYPE,ARRAY,PITCH,X,Y) ( (((TYPE*)((char*)ARRAY + Y*PITCH))+X ))
//Usage: REF2D(int,o_count,p_count,pid,gid)
#define GREF(GRAPH,SUB,OFFSET, X) ( GRAPH[SUB[OFFSET+X]] )
#define FIN(AR, OFFSET, ID) ( AR[OFFSET+ID] ) 

#define FREF(AR, GATE, FIN, REF) ((AR[GATE.FIN.at(REF).second]))
#define FANID(GATE, FIN, REF) ((GATE.FIN.at(REF).second))
#define FADDR(AR,GATE,FIN,REF) (AR+GATE.FIN.at(REF).second)
#define NOTMARKED(MARK, HIST, GATE) ((MARK[GATE] > 0 && HIST[GATE] < 1))
#define NMARKEDG(MARK,HIST,G,PID) ((MARK > 0)*(PID < HIST[G])*(HIST[G] >= 0))

#define TID (((blockIdx.y * blockDim.y) + threadIdx.x))
#define GID(OFFSET) (blockIdx.x + OFFSET)

//utility macro that serve same function as the stability lookup table.

#define STABLE(P, N) (3*(!P & N) + 2*(P & !N) + (P & N))
#define BIN(V) ( (V==T1 || V==S1) )
//thread-per-block sizes, per kernel.
#define THREAD_SHIFT 1024

const unsigned int MAX_BLOCKS = 65535;
const unsigned int SEGMENT_LENGTH = 6;
extern int verbose_flag;
#define OUTJUST 3

#ifdef __CUDACC__ 
	#define HOST_DEVICE __device__ __host__ __forceinline__
	#define DEVICE __device__ __forceinline__
#else 
	#define HOST_DEVICE inline
#endif 

#endif // include guard.
