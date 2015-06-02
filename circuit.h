#ifndef CIRCUIT_H
#define CIRCUIT_H

#include <string>
#include <vector>

#include "logic.h"

using std::string;
using std::vector;


// Circuit, a collection of LogicBlocks
//

class Circuit
{
  public:
    Circuit(const string& name) : _name(name) {}
    string name() const { return _name; }

    bool operator==(const Circuit& other) { return _name == other._name && netlist == other.netlist; }

    Circuit& operator=(const Circuit&) = default; 
    Circuit(const Circuit&) = default;

    vector<LogicBlock> netlist; 
  protected:
    string _name;
};

#endif // CIRCUIT_H
