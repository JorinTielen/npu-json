#include <cstdint>
#include <cstring>
#include <memory>

#include <catch2/catch_all.hpp>

#include <npu-json/engine.hpp>
#include <npu-json/npu/chunk-index.hpp>
#include <npu-json/npu/kernel.hpp>

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

  auto chunk_index = new npu::ChunkIndex();
  npu::construct_escape_carry_index(chunk, *chunk_index, false);

  REQUIRE(chunk_index->escape_carry_index[0] == false);
  REQUIRE(chunk_index->escape_carry_index[1] == true);
  REQUIRE(chunk_index->escape_carry_index[2] == false);
  REQUIRE(chunk_index->escape_carry_index[blocks_in_chunk - 2] == false);
  REQUIRE(chunk_index->escape_carry_index[blocks_in_chunk - 1] == true);

  delete chunk_index;
  delete[] chunk;
}

TEST_CASE("sets first index bit when carry boolean is passed") {
  auto chunk = new char[Engine::CHUNK_SIZE];
  memset(chunk, ' ', Engine::CHUNK_SIZE);

  auto chunk_index = new npu::ChunkIndex();
  npu::construct_escape_carry_index(chunk, *chunk_index, true);

  REQUIRE(chunk_index->escape_carry_index[0] == true);

  delete chunk_index;
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

  auto chunk_index = new npu::ChunkIndex();
  npu::construct_escape_carry_index(chunk, *chunk_index, false);

  REQUIRE(chunk_index->escape_carry_index[1] == true);
  REQUIRE(chunk_index->escape_carry_index[2] == false);

  delete chunk_index;
  delete[] chunk;
}

TEST_CASE("sets index bit when chunk ends on escape (last block)") {
  auto chunk = new char[Engine::CHUNK_SIZE];
  memset(chunk, ' ', Engine::CHUNK_SIZE);

  auto end_of_chunk_idx = Engine::CHUNK_SIZE - 1;
  chunk[end_of_chunk_idx] = '\\';

  auto chunk_index = new npu::ChunkIndex();
  npu::construct_escape_carry_index(chunk, *chunk_index, false);

  auto blocks_in_chunk = Engine::CHUNK_SIZE / Engine::BLOCK_SIZE;
  REQUIRE(chunk_index->escape_carry_index[blocks_in_chunk] == true);

  delete chunk_index;
  delete[] chunk;
}
