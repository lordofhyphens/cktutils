#include "circuit.h"
#include "logic.h"
#include <vector>
#include <fstream>
#include <string>
#include <tuple>
#include "line_iterator.h"

using std::ifstream;
using std::string;
using std::get;
using std::make_tuple;


std::vector<std::string> delimited_string(const std::string& line,  const std::string& separator, size_t start) 
{
  auto pos = start;
  std::vector<std::string> result;
  std::string cleaned_line = line;
  auto new_end = std::unique(cleaned_line.begin(), cleaned_line.end(), [=](char& c1, char&c2) -> bool { return c1 == ' ' && c2 == ' ' && c1 == c2;});
  cleaned_line.erase(new_end, cleaned_line.end());

  std::vector<LogicBlock> sum;
  while (pos < line.size())
  {
    auto gname = cleaned_line.substr(pos, cleaned_line.find(separator, pos+1)-pos);
    gname.erase(std::remove_if(gname.begin(), gname.end(),
    [&separator](char &c) -> bool { 
      return c == *(separator.c_str());
    }),gname.end());
    pos = cleaned_line.find(separator, pos+1);
    result.push_back(gname);
  }
  return std::move(result);
}
std::string normalize(std::string in)
{
  std::string t = in;
  while (t.find("\t") != std::string::npos)
  {
    auto pos = t.find("\t", 0);
    t = t.replace(pos, 1, " ");
  }
  while (t.find("  ") != std::string::npos)
  {
    auto pos = t.find("  ", 0);
    t = t.erase(pos, 1);
  }
  return t;
}

void Circuit::read_blif(const string& filename) 
{
  // parse a simplistic BLIF file. We assume that inputs, outputs are
  // specified. We ignore model and that the file is self-sufficient (no
  // submodels for now)

  // BLIF lines starting with # are ignored until linebreak. 
  ifstream file(filename);
  bool had_minterm = false;
  std::vector<std::string> minterm_list;
  std::tuple<int, std::string> products = make_tuple(0, std::string(""));
  std::string outname = "";
  std::vector<LogicBlock> sum;
  bool single_minterm_product = false;
  for( auto& line_orig: lines(file) )
  {
    auto line = normalize(line_orig);
    // check for a leading #
    if (line.find("#", 0) == 0) continue;
    if (line.find(".model", 0) == 0) 
    {
      continue;
    }
    if (line.find(".latch", 0) == 0)
    {
      auto pos = line.find(" ", line.find(".latch",0));
      std::string src = "";
      size_t i = 0;
      for (auto &gname : delimited_string(line, " ", pos)) {
        switch(i)
        {
          case 0:
            netlist.emplace_back(LogicBlock((gname))); // add a ref to the node this one is referring to
            src = gname;
            break;
          case 1:
            {
              netlist.emplace_back(LogicBlock(gname+"_IN", LogicType::DFF_in)); // actually add the output node
              netlist.back().primary_out = true;
              netlist.back().add_fanin(src); 
              auto it = std::find(netlist.begin(), netlist.end(), LogicBlock(gname, LogicType::Unknown));
              // check to make sure 
              if (it == netlist.end()) 
                netlist.emplace_back(gname, LogicType::DFF);
              else
              {
                it->type = LogicType::DFF;
              }
            }
            break;
          default: 
            // the optionals, ignore for now?
            break;
        }
        i++;
      }
      continue;
    }
    if (line.find(".end", 0) == 0) 
    {
      continue;
    }
    // if .inputs is encountered, add an input node for every name.
    if (line.find(".inputs", 0) == 0)
    {
      // split the rest of the line
      auto pos = line.find(" ", line.find(".inputs",0));
      for (auto &gname : delimited_string(line, " ", pos+1)) {
        if (gname == "" || gname == " ") continue;
        netlist.emplace_back(LogicBlock(gname, LogicType::Input));
        netlist.emplace_back(LogicBlock(gname+"_NOT", LogicType::Not));
        netlist.back().add_fanin(gname);
      }
      continue;
    }
    // if .names is encountered, 
    if (line.find(".outputs", 0) == 0)
    {
      // split the rest of the line
      auto pos = line.find(" ", line.find(".outputs",0));
      for (auto &gname : delimited_string(line, " ", pos)) {
        netlist.emplace_back(LogicBlock(gname, LogicType::Unknown));
        netlist.back().primary_out = true;
      }
      continue;
    }
    if (line.find(".names", 0) == 0)
    {
      auto pos = line.find(" ", line.find(".names",0));
      for (auto &gname : delimited_string(line, " ", pos))
      {
        minterm_list.push_back(gname);
      }
      outname = minterm_list.back();
      minterm_list.pop_back();
      had_minterm = true;
      get<0>(products) = 0;
      continue;
    }
    // otherwise, we are tracking minterms from a previous gate
    auto term_it = minterm_list.cbegin();
    vector<string> product;
    for(auto c : line)
    {
      switch (c)
      {
        case '0': // inverted var/gate appears
          {
            product.emplace_back(LogicBlock{*term_it+"_NOT"});
            break;
          }
        case '1': // positive var/gate appears
          {
            product.emplace_back(LogicBlock{*term_it});
            break;
          }
        case default: // ignore, var doesn't appear
          break;
      }
      term_it++;
    }

    sum.emplace_back(product);
    for (auto p : product) { sum.back().add_fanin(p); }

  } 
  std::sort(netlist.begin(), netlist.end());
}
