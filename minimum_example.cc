#include "utility.h"
#include "ckt.h"
#include "defines.h"
#include <utility>
#include <iostream>
#include <fstream>
#include <getopt.h>
#include <string>
using namespace std;
int verbose_flag;

int main(int argc, char* argv[]) {
	int arg;
	extern int optind;
	Circuit ckt;
	timespec start, stop;
	float elapsed = 0.0;
	std::string infile, scktfile = "";
	std::vector<string> benchmarks;
	int option_index = 0;
  // The while loop is just to deal with command-line options. 
  // feel free to delete it and just read argv[] manually.
  // requires <getopt.h>

	while (1)
	{
		static struct option long_options[] =
		{
			/* These options set a flag. */
			{"verbose", no_argument,       &verbose_flag, 1},
			/* These options don't set a flag.
			   We distinguish them by their indices. */
			{"help",     no_argument,       0, 'h'},
			{"bench",     required_argument,       0, 'b'},
			{"sckt", required_argument, 0, 't'},
			{0, 0}
		};
		/* getopt_long stores the option index here. */
		arg = getopt_long (argc, argv, "b:s:",
				long_options, &option_index);

		/* Detect the end of the options. */
		if (arg == -1)
			break;

		switch (arg)
		{
			case 0:
				/* If this option set a flag, do nothing else now. */
				if (long_options[option_index].flag != 0)
					break;
				printf ("option %s", long_options[option_index].name);
				if (optarg)
					printf (" with arg %s", optarg);
				printf ("\n");
				break;

			case 'b':
				infile = std::string(optarg);
				break;

			case 't':
				scktfile = std::string(optarg);
				break;
      
      case 'h':
				printf("Usage: %s (options) /path/to/benchmark1 /path/to/benchmark2\n", argv[0]);
				printf("\t--segs N : Run fault grading over segments of length N, where 1 <= N <= 8\n");
				printf("\t--bench /path/to/ckt : A circuit to apply benchmarks.\n");
				abort();
			case '?':
				/* getopt_long already printed an error message. */
				break;

			default:
				abort ();
		}
	}
	if (optind < argc) 
		/* these are the arguments after the command-line options */ 
		for (; optind < argc; optind++) 
				benchmarks.push_back(std::string(argv[optind]));
	else { 
		printf("no arguments left to process\n"); 
	}
	// done with getopts, rest are benchmark vectors.
	if (infile.empty()) {
		printf("--bench argument is required.");
		abort();
	}
	std::cerr << "Reading circuit file " << infile << "....";

// read_bench routine is actually what reads and processes a BENCH format
// the load() function is for an internal, compact netlist format.
	if (infile.find("bench") != std::string::npos) {
		ckt.read_bench(infile.c_str());
	} else {
			std::clog << "presorted benchmark " << infile << " ";
		ckt.load(infile.c_str());
	}
	ckt.print();

	return 0;
}
