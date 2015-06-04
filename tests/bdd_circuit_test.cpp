// needed for CppUtest
#include "CppUTest/TestHarness.h"
#include "CppUTest/TestOutput.h"

#include "../bdd_circuit.h"
#include <utility>
#include <iostream>

using std::get;

TEST_GROUP(BDDCKT_BASIC)
{
  std::unique_ptr<BDDCircuit> ckt= nullptr;
  void setup()
  {
    ckt = std::unique_ptr<BDDCircuit>(new BDDCircuit("s27"));
    ckt->read_blif("tests/s27.blif");
  }
  void teardown()
  {
    ckt = nullptr;
  }
};

TEST(BDDCKT_BASIC, TestBDD_PI_Assignments)
{
  ckt->to_bdd();
  // should have 4 BDDs for the PIs, vars 0-3 in this
  // ckt correspond to G0-G3
  CHECK_EQUAL(4,  ckt->bdd_pi.size());
  CHECK(ckt->manager.bddVar(0) == ckt->bdd_pi.at(0));
  CHECK(ckt->manager.bddVar(1) == ckt->bdd_pi.at(1));
  CHECK(ckt->manager.bddVar(2) == ckt->bdd_pi.at(2));
  CHECK(ckt->manager.bddVar(3) == ckt->bdd_pi.at(3));
}

TEST(BDDCKT_BASIC, TestBDD_PO_Assignments)
{
  // should be 1 PO.
  ckt->to_bdd();
  CHECK_EQUAL(1,  ckt->bdd_po.size());
  CHECK_FALSE(ckt->bdd_po.at(0).IsZero());
}

TEST(BDDCKT_BASIC, TestBDD_DFF_Assignments)
{
  ckt->to_bdd();
  CHECK_EQUAL(3,  ckt->bdd_flops.size());
  // 3 variables for current state, and 3 output BDDs.
  // start counting from var x4, as x0-x3 are pis
  CHECK(ckt->manager.bddVar(4) == get<1>(ckt->bdd_flops.at(0)));
  CHECK(ckt->manager.bddVar(5) == get<1>(ckt->bdd_flops.at(1)));
  CHECK(ckt->manager.bddVar(6) == get<1>(ckt->bdd_flops.at(2)));
}
