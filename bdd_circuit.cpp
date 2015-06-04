#include "bdd_circuit.h"


using std::vector;

void BDDCircuit::to_bdd() 
{
  // start making BDDs.
  for(const auto &i : pi)
  {
    bdd_pi.emplace_back(manager.bddVar());
  }
  for(const auto &i : flops)
  {
    bdd_flops.emplace_back(manager.bddZero(), manager.bddVar());
  }


// should happen last
  for (const auto &i : po)
  {
    bdd_po.emplace_back(manager.bddZero());
  }
}
