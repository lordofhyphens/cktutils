#include "bdd_circuit.h"


using std::vector;

void BDDCircuit::to_bdd() 
{
  // start making BDDs.
  for (size_t i = 0; i < netlist.size(); i++)
  {
    if (netlist.at(i).fin.size() == 1)
    {
      auto it = std::find(netlist.begin(), netlist.end(), netlist.at(i).fin.at(0));
      assert (it != netlist.end());
      if (netlist.at(i).type == LogicType::Not)
        bdd_netlist.emplace_back(!bdd_netlist.at(std::distance(netlist.begin(), it)));
      else
        bdd_netlist.emplace_back(bdd_netlist.at(std::distance(netlist.begin(), it)));
      continue;
    }
    switch(netlist.at(i).type)
    {
      case LogicType::DFF:
      case LogicType::Input:
        bdd_netlist.emplace_back(manager.bddVar());
        if (netlist.at(i).type == LogicType::DFF)
          manager.bddSetNsVar(manager.ReadSize()-1);
        else
          manager.bddSetPiVar(manager.ReadSize()-1);
        break;
      case LogicType::Unknown: break;
      default: 
        {
          auto& gate = netlist.at(i);
          assert(gate.fin.size() > 0);
          auto fin = gate.fin.begin();
          auto it = std::find(netlist.begin(), netlist.end(), *fin);
          BDD result = bdd_netlist.at(std::distance(netlist.begin(), it));
          fin++;
          for (; fin < gate.fin.end(); fin++) 
          {
            it = std::find(netlist.begin(), netlist.end(), *fin);
            assert(it != netlist.end());
            auto pos = std::distance(netlist.begin(), it);
            switch(netlist.at(i).type)
            {
              case LogicType::And:
                result = result.And(bdd_netlist.at(pos));
                break;
              case LogicType::Nand:
                result = result.Nand(bdd_netlist.at(pos));
                break;
              case LogicType::Or:
                result = result.Or(bdd_netlist.at(pos));
                break;

              case LogicType::Nor:
                result = result.Nor(bdd_netlist.at(pos));
                break;
              case LogicType::Xor:
                result = result.Xor(bdd_netlist.at(pos));
                break;
              case LogicType::Xnor:
                result = result.Xnor(bdd_netlist.at(pos));
                break;
            }
            bdd_netlist.emplace_back(std::move(result));
          }
        }
    }
  }
  for(const auto &i : pi)
  {
    auto pos = (std::distance(netlist.begin(), std::find(netlist.begin(), netlist.end(), i)));
    bdd_pi.emplace_back(bdd_netlist.at(pos));
    // find the corresponding entry in netlist 
    // and link it. 


  }
  for(const auto &i : flops)
  {
    auto pos1 = (std::distance(netlist.begin(), std::find(netlist.begin(), netlist.end(), i.first)));
    auto pos2 = (std::distance(netlist.begin(), std::find(netlist.begin(), netlist.end(), i.second)));
    bdd_flops.emplace_back(bdd_netlist.at(pos1), bdd_netlist.at(pos2));
  }

  // iterate through the rest of the netlist, ignoring DFFs and Inputs

  // should happen last
  for (const auto &i : po)
  {
    auto pos = (std::distance(netlist.begin(), std::find(netlist.begin(), netlist.end(), i)));
    bdd_po.emplace_back(bdd_netlist.at(pos));
  }
}
