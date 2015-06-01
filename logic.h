#include <string>
#include <vector>

using std::vector;

class LogicBlock
{
  public:
    LogicBlock(std::string n) : _name(n) { }
    std::string name() const { return _name; }
    void add_fanin(const LogicBlock& o) { fin.push_back(o._name); }

    std::vector<std::string> fin;

    inline size_t nfi() { return fin.size();}
  private:    
    const std::string _name;
};
