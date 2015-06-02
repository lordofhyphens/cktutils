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
    block2 = std::unique_ptr<LogicBlock>(new LogicBlock("G1", logic_t::Not));
    block3 = std::unique_ptr<LogicBlock>(new LogicBlock("G2", logic_t::Buff));
    block2->add_fanin(*block1);
    block3->add_fanin(*block2);
    block3->primary_out = true;
  }

  void teardown()
  {
    block1 = nullptr;
    block2 = nullptr;
    block3 = nullptr;
  }
};

TEST(LB_BASIC, constructor) 
{
  CHECK(Unknown == block1->type);
  CHECK(logic_t::Not == block2->type);
  CHECK(logic_t::Buff == block3->type);
}

TEST(LB_BASIC, primaryout)
{
  CHECK(block3->primary_out);
  CHECK_FALSE(block1->primary_out);
}

TEST(LB_BASIC, get_name) 
{
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

TEST(LB_BASIC, ordering)
{
  CHECK(*block1 < *block2);
  CHECK(*block2 < *block3);

  CHECK_FALSE(*block1 < *block3);
}

TEST(LB_BASIC, equality)
{
  CHECK(*block1 == *block1);
  CHECK_FALSE(*block2 == *block3);
  CHECK_FALSE(*block1 == *block3);
  CHECK_FALSE(*block2 == *block1);
}

TEST(LB_BASIC, move)
{
  LogicBlock t3 = std::move(*block3);

  CHECK_EQUAL("G2", t3.name());
  CHECK_EQUAL(1, t3.nfi());
  CHECK_EQUAL("G1", t3.fin.at(0));

  CHECK_EQUAL(0, block3->nfi());
  CHECK(Unknown == block3->type);
  CHECK(t3.primary_out);
}
TEST(LB_BASIC, copy)
{
  LogicBlock t3 = (*block3);

  CHECK_EQUAL("G2", t3.name());
  CHECK_EQUAL(1, t3.nfi());
  CHECK_EQUAL("G1", t3.fin.at(0));

  CHECK_EQUAL(1, block3->nfi());
  CHECK(logic_t::Buff == block3->type);
  CHECK(t3.primary_out);
}
