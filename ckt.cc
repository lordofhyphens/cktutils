#include "ckt.h"
#include <functional>
#include <parallel/algorithm>
typedef std::vector<NODEC>::iterator nodeiter;
Circuit::Circuit() {
	this->graph = new std::vector<NODEC>();
	this->_levels = 1;
	this->__cached_levelsize = -1;
} 
Circuit::~Circuit() {
	delete this->graph;
}
// Saves in the following format:
// pos:name:type:po:level:nfi:fin1:...:finn:nfo:fot1:...:fotn \n
void Circuit::save(const char* memfile) {
	std::ofstream ofile(memfile);
	unsigned long j = 0;
	for (nodeiter i = this->graph->begin(); i < this->graph->end(); i++) {
		ofile << j << " " << i->name << " " << (int)i->typ << " " << i->po << " " << i->level << " " << i->fin.size() << " ";
		for (std::vector<std::pair<std::string, uint32_t > >::iterator fin = i->fin.begin(); fin < i->fin.end(); fin++) {
			ofile << fin->first << "," << fin->second << " ";
		}
		ofile << i->fot.size();
		for (std::vector<std::pair<std::string, uint32_t > >::iterator fot = i->fot.begin(); fot < i->fot.end(); fot++) {
			ofile << " " << fot->first << "," << fot->second;
		}
		ofile << std::endl;
		j++;
	}
	ofile.close();
}
// pos name type po level nfi fin1 ... finn nfo fot1 ... fotn\n
void Circuit::load(const char* memfile) {
	std::ifstream ifile(memfile);
	int type;
	int nfos = 0;
	unsigned int max_nfo = 0;
	std::string strbuf;
	std::string name;
	while (!ifile.eof()) {
		NODEC node; 
		std::getline (ifile,strbuf);
		if (strbuf.size() < 5) {
			continue;
		}
		std::stringstream buf(strbuf);
		buf.ignore(300, ' ');
		buf >> node.name >> type >> node.po >> node.level >> node.nfi;
		node.typ = type;
		for (unsigned int i = 0; i < node.nfi; i++) {
			std::string temp;
			int id;
			size_t p;
			buf >> temp;
			p = temp.find(",");
			node.finlist.append(temp.substr(0,p));
			if (i < node.nfi-1)
				node.finlist.append(",");

			std::stringstream fnum(temp.substr(p+1));
			from_string<int>(id, temp.substr(p+1), std::dec);
			node.fin.push_back(std::make_pair(temp.substr(0,p),id));
		}
		buf >> node.nfo;
		max_nfo = (node.nfo > max_nfo ? node.nfo : max_nfo);
		nfos += node.nfo;
		for (unsigned int i = 0; i < node.nfo; i++) {
			std::string temp;
			int id;
			size_t p;
			buf >> temp;
			p = temp.find(",");
			std::stringstream fnum(temp.substr(p+1));
			from_string<int>(id, temp.substr(p+1), std::dec);
			node.fot.push_back(std::make_pair(temp.substr(0,p),id));
		}
		this->graph->push_back(node);
	}
	this->_levels = 1;
	for (std::vector<NODEC>::iterator a = this->graph->begin(); a < this->graph->end(); a++) {
		this->_levels = std::max(this->_levels, a->level);
	}
	_avg_nfo = (double)nfos / size();
	_max_nfo = max_nfo;
	std::cerr << nfos << "\n";
}
void Circuit::load(const char* memfile, const char * ext_id) {
	std::ifstream ifile(memfile);
	int nfos = 0;
	unsigned int max_nfo = 0;
	int type;
	std::string strbuf;
	std::string name;
	std::vector<NODEC> *g = new std::vector<NODEC>();
	const std::string cktid(ext_id);
	while (!ifile.eof()) {
		NODEC node; 
		std::getline (ifile,strbuf);
		if (strbuf.size() < 5) {
			continue;
		}
		std::stringstream buf(strbuf);
		buf.ignore(300, ' ');
		buf >> node.name >> type >> node.po >> node.level >> node.nfi;
		std::string tmp_node = node.name + cktid;
		node.name = tmp_node;
		node.typ = type;
		for (unsigned int i = 0; i < node.nfi; i++) {
			std::string temp;
			int id;
			size_t p;
			buf >> temp;
			p = temp.find(",");
			std::string tmp = temp.substr(0,p).append(cktid);
			node.finlist.append(tmp);
			if (i < node.nfi-1)
				node.finlist.append(",");
			from_string<int>(id, temp.substr(p+1), std::dec);
			node.fin.push_back(std::make_pair(tmp,0));
		}
		buf >> node.nfo;
		max_nfo = (node.nfo >= max_nfo ? node.nfo : max_nfo);
		nfos += node.nfo;
		for (unsigned int i = 0; i < node.nfo; i++) {
			std::string temp;
			size_t p;
			buf >> temp;
			p = temp.find(",");
			std::string tmp(temp.substr(0,p));
			std::string tmp2 = tmp+cktid;
			node.fot.push_back(std::make_pair(tmp2,0));
		}
		g->push_back(node);
	}
	this->graph->insert(graph->end(),g->begin(),g->end());
	delete g;
	
	this->_levels = 1;
	for (std::vector<NODEC>::iterator a = this->graph->begin(); a < this->graph->end(); a++) {
		this->_levels = std::max(this->_levels, a->level);
	}
	_avg_nfo = (double)nfos / size();
	std::cerr << nfos << "\n";
	_max_nfo = max_nfo;

}
void Circuit::read_bench(const char* benchfile, const char* ext) {
	std::ifstream tfile(benchfile);
	this->name = benchfile;
	this->name.erase(std::remove_if(this->name.begin(), this->name.end(),isspace),this->name.end());
	this->name.erase(__gnu_parallel::find(this->name.begin(),this->name.end(),'.'),this->name.end());
	std::vector<NODEC>* g = new std::vector<NODEC>();
	std::string buffer, id;
	std::stringstream node;
	int front, back;
	while (getline(tfile,buffer)) {
		node.str(buffer);
		if (buffer.find("#") != std::string::npos) 
			continue;
		else if (buffer.find("INPUT") != std::string::npos) {
			front = buffer.find("(");
			back = buffer.find(")");
			id = buffer.substr(front+1, back - (front+1));
			id.append(ext);
			g->push_back(NODEC(id, INPT));
		} else if (buffer.find("OUTPUT") != std::string::npos) {
			front = buffer.find("(");
			back = buffer.find(")");
			id = buffer.substr(front+1, back - (front+1));
			id.append(ext);
			g->push_back(NODEC(id));
			g->back().po = true;
		} else if (buffer.find("DFF") != std::string::npos) {
      // this needs to become 2 separate nodes, one for the input and one
      // for the output.
			id = buffer.substr(0,buffer.find("=")); // an input, has no fins
			id.erase(std::remove_if(id.begin(), id.end(),isspace),id.end());
      nodeiter iter;
      if ((iter = __gnu_parallel::find(g->begin(), g->end(), id)) == g->end()) { 
        g->push_back(NODEC(id,DFF));
			} else {
				// modify the pre-existing node. Node type should be unknown, and PO should be set.
       	assert(iter->po == true);
				assert(iter->typ == UNKN);
				*iter = NODEC(id, DFF);
      }

			id = buffer.substr(0,buffer.find("=")) + "_in"; // a PO, has fins
			id.erase(std::remove_if(id.begin(), id.end(),isspace),id.end());
			id.append(ext);
			front = buffer.find("(");
			back = buffer.find(")");
			std::string finlist = buffer.substr(front+1, back - (front+1));
			int nfi = count_if(finlist.begin(), finlist.end(), ispunct) + 1;

			if (__gnu_parallel::find(g->begin(), g->end(), id) == g->end()) { 
				g->push_back(NODEC(id, DFF_IN, nfi, finlist));
        g->back().po = true;
			}
      
    } else if (buffer.find("=") != std::string::npos) {
			id = buffer.substr(0,buffer.find("="));
			id.erase(std::remove_if(id.begin(), id.end(),isspace),id.end());
			id.append(ext);
			front = buffer.find("(");
			back = buffer.find(")");
			std::string finlist = buffer.substr(front+1, back - (front+1));
			std::string gatetype = buffer.substr(buffer.find("=")+1,front - (buffer.find("=")+1));
			int nfi = count_if(finlist.begin(), finlist.end(), ispunct) + 1;
			if (__gnu_parallel::find(g->begin(), g->end(), id) == g->end()) { 
				g->push_back(NODEC(id, gatetype, nfi, finlist));
			} else {
				// modify the pre-existing node. Node type should be unknown, and PO should be set.
				nodeiter iter = find(g->begin(), g->end(), id);
				assert(iter->po == true);
				assert(iter->typ == UNKN);
				*iter = NODEC(id, gatetype, nfi, finlist);
				iter->po = true;
			}
		} else {
			continue;
		}
	}
	std::clog << "Finished reading " << g->size() << " lines from file." <<std::endl;
	for (nodeiter iter = g->begin(); iter < g->end(); iter++) {
		if (iter->finlist == "")
			continue;
		node.str(iter->finlist);
		node.clear();
		while (getline(node,buffer,',')) {
			// figure out which which node has this as a fanout.
			buffer.append(ext);
			nodeiter j = __gnu_parallel::find(g->begin(), g->end(), buffer);
      if (j == g->end())
        j = __gnu_parallel::find(g->begin(), g->end(), buffer+"_in");
			j->nfo++;
		}
	}
	std::vector<NODEC> temp_batch;
	for (nodeiter iter = g->begin(); iter < g->end(); iter++) {
		node.str(iter->finlist);
		node.clear();
		std::string newfin = "";
		while (getline(node,buffer,',')) {
			buffer.append(ext);
			nodeiter j = __gnu_parallel::find(g->begin(), g->end(), buffer);
      if (j == g->end())
        j = __gnu_parallel::find(g->begin(), g->end(), buffer+"_in");
			if (j->nfo < 2) {
				iter->fin.push_back(std::make_pair(j->name, -1));
				j->fot.push_back(std::make_pair(iter->name, -1));
				if (newfin == "") {
					newfin += j->name;
				} else {
					newfin += "," + j->name;
				}
			} else {
				std::stringstream tmp;
				tmp << j->cur_fo;
				j->cur_fo+=1;
				temp_batch.push_back(NODEC((j->name+"fan"+tmp.str()),"FROM",1,j->name));
				temp_batch.back().fot.push_back(std::make_pair(iter->name,-1));
				temp_batch.back().fin.push_back(std::make_pair(j->name,-1));
				temp_batch.back().nfo = 1;
				j->fot.push_back(std::make_pair(temp_batch.back().name,-1));
				iter->fin.push_back(std::make_pair(j->name+"fan"+tmp.str(),-1));
				if (newfin == "") {
					newfin += j->name+"fan"+tmp.str();
				} else {
					newfin += "," + j->name+"fan"+tmp.str();
				}
			}
		}
		iter->finlist = newfin;
	}
	for (nodeiter iter = temp_batch.begin(); iter < temp_batch.end(); iter++) {
		g->push_back(*iter);
	}
	std::clog << "Removing empty nodes." <<std::endl;
	remove_if(g->begin(),g->end(),isUnknown);


	this->graph->insert(this->graph->end(), g->begin(), g->end());
	delete g;
	g = this->graph;

	std::clog << "Sorting circuit." << std::endl;
	__gnu_parallel::sort(g->begin(), g->end(),nameSort);
	std::clog << "Removing duplicate nodes." << std::endl;
	std::vector<NODEC>::iterator it = unique(g->begin(),g->end(),isDuplicate);
	g->resize(it - g->begin());
	
	std::clog << "Annotating circuit." << std::endl;
	annotate(g);

	std::clog << "Levelizing circuit." << std::endl;
	this->levelize();
	std::clog << "Sorting circuit." << std::endl;
	__gnu_parallel::sort(g->begin(), g->end());
	std::clog << "Annotating circuit." << std::endl;
	annotate(g);
}
bool isPlaced(const NODEC& node) {
	return (node.placed == 0);
}
inline bool Yes(const NODEC& node) {
	return true;
}
// levelize the circuit.
void Circuit::levelize() {
	std::vector<NODEC>* g = this->graph;

	while (count_if(g->begin(),g->end(), isPlaced) > 0) {
		for (nodeiter iter = g->begin(); iter < g->end(); iter++) {
			if (iter->placed == false) {
				if (iter->typ == INPT || iter->typ == DFF)  {
					iter->level = 0;
					iter->placed = true;
				} else {
          if (verbose_flag) {
            std::cerr << "Trying to place " << iter->name << "\n";
          }
					bool allplaced = true;
					unsigned int level = 0;
					for (unsigned int i = 0; i < iter->fin.size(); i++) {
						allplaced = allplaced && (g->at(iter->fin[i].second).placed || g->at(iter->fin[i].second).typ == DFF);
						if (level < g->at(iter->fin[i].second).level)
							level = g->at(iter->fin[i].second).level;
					}
					if (allplaced == true) { 
						iter->level = ++level;
						iter->placed = true;
						if (level+1 > this->_levels)
							this->_levels = level;
						iter->nfi = iter->fin.size();
						iter->nfo = iter->fot.size();
					}

				}
			}
		}
	}
}
void Circuit::print() const {
	std::vector<NODEC>* g = this->graph;
	std::cout << "Circuit: " << this->name << std::endl;
	std::cout << "Name\tType\tPO?\tLevel\tNFI\tNFO\tFinlist\t\tFin | Fot" << std::endl;
	for (nodeiter iter = g->begin(); iter < g->end(); iter++) {
		std::cout << *iter;
	}
}

unsigned int Circuit::levelsize(unsigned int l) const {
	return countInLevel(*graph, l);
}


// labels each fanin of each circuit 
void Circuit::annotate(std::vector<NODEC>* g) {
	for (std::vector<NODEC>::iterator iter = g->begin(); iter < g->end(); iter++) {
		for (std::vector<std::pair<std::string, uint32_t> >::iterator i = iter->fin.begin(); i < iter->fin.end(); i++) {
			const std::vector<NODEC>::iterator a =  __gnu_parallel::find(g->begin(),g->end(),i->first);
			i->second = __gnu_parallel::count_if(g->begin(), a, Yes);
		}
		for (std::vector<std::pair<std::string, uint32_t> >::iterator i = iter->fot.begin(); i < iter->fot.end(); i++) {
			const std::vector<NODEC>::iterator a =  __gnu_parallel::find(g->begin(),g->end(),i->first);
			i->second = __gnu_parallel::count_if(g->begin(), a, Yes);
		}
		DPRINT("Finished node %s, %lu/%lu\n", iter->name.c_str(), std::distance(g->begin(),iter), g->size());
	}
}
inline bool isInLevel(const NODEC& node, const unsigned int& N) { return node.level == N; }

unsigned int countInLevel(const std::vector<NODEC>& v, const unsigned int& level)  {
		unsigned int cnt = 0;
		cnt = std::count_if(v.begin(), v.end(), std::bind(isInLevel, std::placeholders::_1, level)); 
/*		for (std::vector<NODEC>::const_iterator iter = v.begin(); iter < v.end(); iter++) {
			if (isInLevel(*iter, level)) 
				cnt = cnt + 1;
		}*/

	return cnt;
}
bool Circuit::nodelevel(unsigned int n, unsigned int m) const {
	return graph->at(n).level < graph->at(m).level;
}
bool isUnknown(const NODEC& node) {
	return node.typ == UNKN;
}


bool isDuplicate(const NODEC& a, const NODEC& b) {
	return (a.name == b.name && a.typ == b.typ);
}
inline bool nameSort(const NODEC& a, const NODEC& b) { return (a.name < b.name); }

size_t Circuit::max_level_pair() {
	size_t max = 0;
	for (size_t i = 0; i < levels() - 1; i++) {
		size_t a = levelsize(i) + levelsize(i+1);
		if (max < a) max = a;
	}
	return max;
}

// Returns the maximum number of nodes not in i that are in j's fan-in
size_t Circuit::out_of_level_nodes(size_t i, size_t j) {
	unsigned int ref = 0, count = 0;
	unsigned int index = 0;
	while (at(index).level != j && index < size()) index++;
	ref = index;
	for (index = ref; index < ref+levelsize(j); index++) {
		for (unsigned int fin = 0; fin < at(index).nfi; fin++) {
			if (at(at(index).fin.at(fin).second).level != j) { count++;}
		}
	}
	return count;
}

// Check every pair-wise level for the maximum # of nodes that are not in a previous level's fan-in
size_t Circuit::max_out_of_level_nodes() {
	size_t max = 0;
	for (size_t i = 0; i < levels() - 1; i++) {
		size_t a = out_of_level_nodes(i, i+1);
		if (max < a) max = a;
	}
	return max;
}

// Compute for every node where it sits in the scratchpad memory.  A node gets
// a placement in the scratchpad if and only if it has a reference from another
// node that is in a non-adjacent level.
// To determine this, only need to check the level of the fan-outs of the current node. 
// Keep a separate list (temporary) for all nodes and where they are. 
// At every node, check the fan-ins. If a node in the fan-ins is in the scratchpad, add 
void Circuit::compute_scratchpad() {
	std::queue<uint32_t> next_scratch;
	unsigned int i = 0;
	next_scratch.push(0);
	for (std::vector<NODEC>::iterator it = graph->begin(); it < graph->end(); it++) {
		// go through fan-ins, add those ids to next_scratch if they are a fan-in of this node.
		for (unsigned int fin = 0; fin < it->fin.size(); fin++) {
			if (at(it->fin.at(fin).second).scratch >= 0) {
				next_scratch.push(at(it->fin.at(fin).second).scratch);
			}
		}
		for (unsigned int fot = 0; fot < it->fot.size(); fot++) {
			if (at(it->fot.at(fot).second).level != (it->level + 1) ) {
				it->scratch = next_scratch.front();next_scratch.pop();
				next_scratch.push(++i);
				// only need in scratchpad once. 
				fot = it->nfo;
				continue;
			}
		}
	}
}
bool scratch_compare(const NODEC& a, const NODEC& b) {
	return a.scratch < b.scratch;
}
void Circuit::reannotate() { __gnu_parallel::sort(this->graph->begin(), this->graph->end()); DPRINT("Annotating.\n"); annotate(this->graph);}
