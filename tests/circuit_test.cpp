
// needed for CppUtest
#include "CppUTest/TestHarness.h"
#include "CppUTest/TestOutput.h"

#include "../circuit.h"
#include <utility>
#include <iostream>
using std::make_pair;
using std::string;
using std::pair;

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
  ckt->emplace_back({"G0", logic_t::Input});
  CHECK(ckt->at(0) == blk);
}

TEST_GROUP(CKT_READ)
{
  std::unique_ptr<Circuit> ckt= nullptr;
  void setup()
  {
    ckt = std::unique_ptr<Circuit>(new Circuit("s27"));
    
  }
  void teardown()
  {
    ckt = nullptr;
  }
};

// S27 BLIF format
// 4 inputs
// 3 latches
// 1 output
// 6 intermediate gates
TEST(CKT_READ, read_blif_s27)
{
  ckt->read_blif("tests/s27.blif");
  for (auto i : *ckt)
  {
    std::cerr << i.name() << "\n";
  }

  CHECK_EQUAL(35, ckt->size());
  auto flop_it = ckt->flops.cbegin();
  CHECK(*flop_it == (std::pair<string,string>{"G5", "G10"}));
  flop_it++;
  CHECK(*flop_it == (std::pair<string,string>{"G6", "G11"}));
  flop_it++;
  CHECK(*flop_it == (std::pair<string,string>{"G7", "G13"}));


  auto po_it = ckt->po.cbegin();
  CHECK_EQUAL("G17", *po_it);
  
  
  auto pi_it = ckt->pi.cbegin();
  CHECK_EQUAL("G0", *pi_it);
  pi_it++;
  CHECK_EQUAL("G1", *pi_it);
  pi_it++;
  CHECK_EQUAL("G2", *pi_it);
  pi_it++;
  CHECK_EQUAL("G3", *pi_it);
}
