#include <string>

class LogicBlock
{
  public:
    LogicBlock(std::string n) : _name(n) { }
    std::string name() const { return _name; }
  private:
    const std::string _name;
};
