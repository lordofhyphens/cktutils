#include "utility.h"
timespec diff(timespec start, timespec end) {
	timespec temp;
	if ((end.tv_nsec-start.tv_nsec) < 0) {
		temp.tv_sec = end.tv_sec-start.tv_sec-1;
		temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
	} else {
		temp.tv_sec = end.tv_sec-start.tv_sec;
		temp.tv_nsec = end.tv_nsec-start.tv_nsec;
	}
	return temp;
}

float floattime(timespec time) {
	float temp =0.0; 
	temp = (time.tv_sec * pow(10,3)) + (time.tv_nsec / pow(10,6));
	return temp;
}
int largest_size(std::vector<SubCkt>::iterator& start, std::vector<SubCkt>::iterator& end, int level) {
	int res = 0; 
	for (int i = 0; (start+i) < end; i++) { if (res < (start+i)->levelsize(i)) { res = (start+i)->levelsize(i); } }
	return res;
}
