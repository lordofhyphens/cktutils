
// needed for CppUtest
#include "CppUTest/TestHarness.h"
#include "CppUTest/TestOutput.h"

#include "../circuit.h"

TEST_GROUP(CKT_BASIC)
{
  std::unique_ptr<Circuit> ckt= nullptr;
  void setup()
  {
    ckt = std::unique_ptr<Circuit>(new Circuit("c17"));
    
  }
  void teardown()
  {
    ckt = nullptr;
  }
};

TEST(CKT_BASIC, SayMyName)
{
  CHECK_EQUAL("c17", ckt->name());
}

TEST(CKT_BASIC, equality)
{
  Circuit ckt2("c17");
  CHECK(ckt2 == *ckt);
}

TEST(CKT_BASIC, add)
{
  LogicBlock blk("G0", logic_t::Input);
  ckt->add(blk);
  CHECK(ckt->at(0) == blk);
}

TEST(CKT_BASIC, emplace)
{
  LogicBlock blk("G0", logic_t::Input);
  ckt->emplace({"G0", logic_t::Input});
  CHECK(ckt->at(0) == blk);
}
