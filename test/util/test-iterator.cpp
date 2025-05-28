#include "test-iterator.hpp"

TestIterator::TestIterator(npu::ChunkIndex & chunk_index)
  : index(chunk_index) {}

uint32_t* TestIterator::get_next_structural_character() {
  if (current_pos_in_chunk < index.structurals_count) {
    auto ptr = &index.structural_characters[current_pos_in_chunk];
    current_pos_in_chunk++;
    return ptr;
  }

  return nullptr;
}
