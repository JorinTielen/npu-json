#include <cstdint>
#include <cstring>
#include <memory>

#include <catch2/catch_all.hpp>

#include <npu-json/engine.hpp>
#include <npu-json/npu/indexer.hpp>

TEST_CASE("detects structural characters and builds index") {
  auto chunk = new char[Engine::CHUNK_SIZE];
  memset(chunk, ' ', Engine::CHUNK_SIZE);

  auto json = std::string("{\"asdf\": 1234, \"arrays\": [[1], [2]]}");

  memcpy(chunk, json.c_str(), json.length());

  auto indexer = std::make_unique<npu::StructuralIndexer>("test.xclbin", "test-insts.txt", false);

  auto structural_index = indexer->construct_structural_index(chunk, false, false);

  auto structural_char = structural_index->get_next_structural_character();
  REQUIRE(structural_char.has_value());
  REQUIRE(structural_char.value().c == '{');
  REQUIRE(structural_char.value().pos == 0);

  structural_char = structural_index->get_next_structural_character();
  REQUIRE(structural_char.has_value());
  REQUIRE(structural_char.value().c == ':');
  REQUIRE(structural_char.value().pos == 7);

  structural_char = structural_index->get_next_structural_character();
  REQUIRE(structural_char.has_value());
  REQUIRE(structural_char.value().c == ',');
  REQUIRE(structural_char.value().pos == 13);

  structural_char = structural_index->get_next_structural_character();
  REQUIRE(structural_char.has_value());
  REQUIRE(structural_char.value().c == ':');
  REQUIRE(structural_char.value().pos == 23);

  structural_char = structural_index->get_next_structural_character();
  REQUIRE(structural_char.has_value());
  REQUIRE(structural_char.value().c == '[');
  REQUIRE(structural_char.value().pos == 25);

  structural_char = structural_index->get_next_structural_character();
  REQUIRE(structural_char.has_value());
  REQUIRE(structural_char.value().c == '[');
  REQUIRE(structural_char.value().pos == 26);

  structural_char = structural_index->get_next_structural_character();
  REQUIRE(structural_char.has_value());
  REQUIRE(structural_char.value().c == ']');
  REQUIRE(structural_char.value().pos == 28);

  structural_char = structural_index->get_next_structural_character();
  REQUIRE(structural_char.has_value());
  REQUIRE(structural_char.value().c == ',');
  REQUIRE(structural_char.value().pos == 29);

  structural_char = structural_index->get_next_structural_character();
  REQUIRE(structural_char.has_value());
  REQUIRE(structural_char.value().c == '[');
  REQUIRE(structural_char.value().pos == 31);

  structural_char = structural_index->get_next_structural_character();
  REQUIRE(structural_char.has_value());
  REQUIRE(structural_char.value().c == ']');
  REQUIRE(structural_char.value().pos == 33);

  structural_char = structural_index->get_next_structural_character();
  REQUIRE(structural_char.has_value());
  REQUIRE(structural_char.value().c == ']');
  REQUIRE(structural_char.value().pos == 34);

  structural_char = structural_index->get_next_structural_character();
  REQUIRE(structural_char.has_value());
  REQUIRE(structural_char.value().c == '}');
  REQUIRE(structural_char.value().pos == 35);
}

TEST_CASE("works across block boundaries") {
  auto chunk = new char[Engine::CHUNK_SIZE];
  memset(chunk, ' ', Engine::CHUNK_SIZE);

  auto json = std::string(60, ' ').append("{\"asdf\": 1234}");

  memcpy(chunk, json.c_str(), json.length());

  auto indexer = std::make_unique<npu::StructuralIndexer>("test.xclbin", "test-insts.txt", false);

  auto structural_index = indexer->construct_structural_index(chunk, false, false);

  auto structural_char = structural_index->get_next_structural_character();
  REQUIRE(structural_char.has_value());
  REQUIRE(structural_char.value().c == '{');
  REQUIRE(structural_char.value().pos == 60);

  structural_char = structural_index->get_next_structural_character();
  REQUIRE(structural_char.has_value());
  REQUIRE(structural_char.value().c == ':');
  REQUIRE(structural_char.value().pos == 67);

  structural_char = structural_index->get_next_structural_character();
  REQUIRE(structural_char.has_value());
  REQUIRE(structural_char.value().c == '}');
  REQUIRE(structural_char.value().pos == 73);
}
