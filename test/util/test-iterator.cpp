#include <npu-json/npu/chunk-index.hpp>

#include "test-iterator.hpp"

TestIterator::TestIterator(npu::ChunkIndex & chunk_index)
  : index(chunk_index) {}

uint32_t* TestIterator::get_next_structural_character() {
  auto potential_structural = get_next_structural_character_in_block();
  if (potential_structural != nullptr) {
    return potential_structural;
  }

  while (current_block < npu::StructuralCharacterBlock::BLOCKS_PER_CHUNK) {
    current_block++;
    potential_structural = get_next_structural_character_in_block();
    if (potential_structural != nullptr) {
      return potential_structural;
    }
  }

  return nullptr;
}

uint32_t* TestIterator::get_next_structural_character_in_block() {
  if (current_pos_in_block < index.blocks[current_block].structural_characters_count) {
    auto ptr = &index.blocks[current_block].structural_characters[current_pos_in_block];
    current_pos_in_block++;
    return ptr;
  }

  return nullptr;
}
