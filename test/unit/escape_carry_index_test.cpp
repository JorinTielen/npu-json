#include <cstdint>
#include <cstring>
#include <memory>

#include <catch2/catch_all.hpp>

#include <npu-json/engine.hpp>
#include <npu-json/npu/indexer.hpp>

TEST_CASE("sets index bit when block ends on escape") {
  auto chunk = new char[Engine::CHUNK_SIZE];
  memset(chunk, ' ', Engine::CHUNK_SIZE);

  auto blocks_in_chunk = Engine::CHUNK_SIZE / Engine::BLOCK_SIZE;
  for (size_t block = 1; block < blocks_in_chunk; block++) {
    auto end_of_block_idx = block * Engine::BLOCK_SIZE - 1;

    // Put a backslash at the end of all odd chunks
    auto has_backslash_at_end = block % 2;
    chunk[end_of_block_idx] = has_backslash_at_end ? '\\' : ' ';
  }

  auto indexer = std::make_unique<npu::StructuralIndexer>("test.xclbin", "test-insts.txt", false);

  auto structural_index = indexer->construct_structural_index(chunk, false, false, 0);

  REQUIRE(structural_index->escape_carry_index[0] == false);
  REQUIRE(structural_index->escape_carry_index[1] == true);
  REQUIRE(structural_index->escape_carry_index[2] == false);
  REQUIRE(structural_index->escape_carry_index[blocks_in_chunk - 2] == false);
  REQUIRE(structural_index->escape_carry_index[blocks_in_chunk - 1] == true);

  delete[] chunk;
}

TEST_CASE("sets first index bit when carry boolean is passed") {
  auto chunk = new char[Engine::CHUNK_SIZE];
  memset(chunk, ' ', Engine::CHUNK_SIZE);

  auto indexer = std::make_unique<npu::StructuralIndexer>("test.xclbin", "test-insts.txt", false);

  auto structural_index = indexer->construct_structural_index(chunk, true, false, 0);

  REQUIRE(structural_index->escape_carry_index[0] == true);

  delete[] chunk;
}

TEST_CASE("sets index bit only for odd sequences of backslashes") {
  auto chunk = new char[Engine::CHUNK_SIZE];
  memset(chunk, ' ', Engine::CHUNK_SIZE);

  auto first_end_of_block_idx = 1 * Engine::BLOCK_SIZE - 1;

  chunk[first_end_of_block_idx - 2] = '\\';
  chunk[first_end_of_block_idx - 1] = '\\';
  chunk[first_end_of_block_idx] = '\\';

  auto second_end_of_block_idx = 2 * Engine::BLOCK_SIZE - 1;

  chunk[second_end_of_block_idx - 1] = '\\';
  chunk[second_end_of_block_idx] = '\\';

  auto indexer = std::make_unique<npu::StructuralIndexer>("test.xclbin", "test-insts.txt", false);

  auto structural_index = indexer->construct_structural_index(chunk, false, false, 0);

  REQUIRE(structural_index->escape_carry_index[1] == true);
  REQUIRE(structural_index->escape_carry_index[2] == false);

  delete[] chunk;
}

TEST_CASE("sets index bit when chunk ends on escape (last block)") {
  auto chunk = new char[Engine::CHUNK_SIZE];
  memset(chunk, ' ', Engine::CHUNK_SIZE);

  auto end_of_chunk_idx = Engine::CHUNK_SIZE - 1;
  chunk[end_of_chunk_idx] = '\\';

  auto indexer = std::make_unique<npu::StructuralIndexer>("test.xclbin", "test-insts.txt", false);

  auto structural_index = indexer->construct_structural_index(chunk, false, false, 0);

  auto blocks_in_chunk = Engine::CHUNK_SIZE / Engine::BLOCK_SIZE;
  REQUIRE(structural_index->escape_carry_index[blocks_in_chunk] == true);

  delete[] chunk;
}
