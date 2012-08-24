#!/usr/bin/python

import argparse
import string

class subckts:
	def __init__(self):
		count = 0
		ckts = None
def spaced(a):
	ret = ""
	for i in a:
		ret += str(a)
	return ret
parser = argparse.ArgumentParser(description='Partition a subcircuit list into similarly-sized')
parser.add_argument('benchmark', metavar='IN', type=str,
                   help='input')
parser.add_argument('parts', metavar='N', type=int,
                   help='input')
parser.add_argument('output', metavar='OUT', type=str, help='output, basic name')
args = parser.parse_args()

f = open(args.benchmark)
lists = list()
for i in range(0, args.parts):
	lists.append(subckts());
	lists[i].count = 0;
	lists[i].ckts = []
for line in f:
	items = map(lambda x: int(x),string.replace(line," \n","").split(":")[1].split(" "))
	min(lists,key=lambda x: x.count).ckts.append(items)
	min(lists,key=lambda x: x.count).count += len(items)
inc = 0 
for i in lists:
	print i.count
	outfile = open(args.output + "-" + str(inc) + ".sub", 'w')
	for ckt in i.ckts:
		outfile.write("0:")
		for j in ckt:
			outfile.write(str(j) + " ")
		outfile.write("\n")
	outfile.close()
	inc += 1
