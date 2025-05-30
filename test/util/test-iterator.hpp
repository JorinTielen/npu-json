#pragma once

#include <npu-json/npu/chunk-index.hpp>
#include <npu-json/engine.hpp>

// Test iterator to iterate over a single chunk.
class TestIterator {
public:
  TestIterator(npu::ChunkIndex & chunk_index);
  uint32_t * get_next_structural_character();
private:
  uint32_t * get_next_structural_character_in_block();
  npu::ChunkIndex & index;
  std::size_t current_pos_in_block = 0;
  std::size_t current_block = 0;
};
