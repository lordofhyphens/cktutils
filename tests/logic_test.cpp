#include "../logic.h"
// needed for CppUtest
#include "CppUTest/TestHarness.h"
#include "CppUTest/TestOutput.h"


// Test the basic functionality of LogicBlocks
// including operators overloaded for them, 
// getter and setter functions, etc.
// 3 functions. Block1 is a primary input G0,
// Block2 has G0 as a fanin, and block 3 has Block2
// as a fanin

TEST_GROUP(LB_BASIC)
{
  std::unique_ptr<LogicBlock> block1= nullptr;
  std::unique_ptr<LogicBlock> block2= nullptr;
  std::unique_ptr<LogicBlock> block3= nullptr;

  void setup()
  {
    block1 = std::unique_ptr<LogicBlock>(new LogicBlock("G0"));
    block2 = std::unique_ptr<LogicBlock>(new LogicBlock("G1"));
    block3 = std::unique_ptr<LogicBlock>(new LogicBlock("G2"));
    block2->add_fanin(*block1);
    block3->add_fanin(*block2);
  }

  void teardown()
  {
  }
};

TEST(LB_BASIC, get_name) {
  CHECK_EQUAL("G0", block1->name());
  CHECK_EQUAL("G1", block2->name());
  CHECK_EQUAL("G2", block3->name());
}

TEST(LB_BASIC, fanins) {
  CHECK_EQUAL(0, block1->nfi());
  CHECK_EQUAL(1, block2->nfi());

  CHECK_EQUAL("G0", block2->fin.at(0));

  CHECK_EQUAL(1, block3->nfi());
  CHECK_EQUAL("G1", block3->fin.at(0));
}
