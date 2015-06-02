#ifndef LOGIC_H
#define LOGIC_H
#include <string>
#include <algorithm>
#include <vector>

using std::vector;
using std::begin;
using std::end;
using std::move;
using std::forward;

enum class LogicType { And, Or, Not, Xor, Xnor, Buff, Inpt, DFF, DFF_in, Unknown};
using logic_t = LogicType;

using LogicType::Unknown;

class LogicBlock
{
  public:
    LogicBlock(std::string n) : type(Unknown), primary_out(false), _name(n) { }
    LogicBlock(std::string n, logic_t type) : type(type), primary_out(false), _name(n) { }
    std::string name() const { return _name; }
    void add_fanin(const LogicBlock& o) { fin.push_back(o._name); }

    std::vector<std::string> fin;

    inline size_t nfi() { return fin.size(); }
    bool operator<(const LogicBlock& z) const { return std::find(begin(z.fin), end(z.fin), _name) != end(z.fin);}
    bool operator==(const LogicBlock& z) const { return _name == z._name && fin == z.fin; }
    bool operator==(const std::string& z) const { return _name == z; }

    LogicBlock(LogicBlock&& other) : fin(move(other.fin)), type(move(other.type)), primary_out(move(other.primary_out)), _name(std::move(other._name)) 
    { other.type = logic_t::Unknown; }

    LogicBlock(const LogicBlock&) = default; // copy constructor
    LogicBlock& operator=(const LogicBlock&) = default; // copy assignment

    LogicType type;
    bool primary_out;
  protected:    
    const std::string _name;
};

#endif // LOGIC_H
