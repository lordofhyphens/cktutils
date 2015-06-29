#ifndef BDD_CIRCUIT_H
#define BDD_CIRCUIT_H


#include <fstream> // needed before cudd for some reason
#include <cuddObj.hh>
#include <cudd.h>
#include <vector>

#include "circuit.h"

using std::vector;
using std::string;

class BDDCircuit : public Circuit
{
  public:
    BDDCircuit(string name) : Circuit(name), manager(Cudd()) {}
    Cudd manager;
    vector<std::pair<BDD,BDD>> bdd_flops;
    vector<BDD> bdd_pi;
    vector<BDD> bdd_po;

    void to_bdd();
  private:
    vector<BDD> bdd_netlist; 
};



#endif // BDD_CIRCUIT_H
