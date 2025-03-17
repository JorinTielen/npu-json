#include <stdexcept>

#include <npu-json/npu/indexer.hpp>

namespace npu {

std::optional<StructuralIndex::StructuralCharacter> StructuralIndex::get_next_structural_character() {
  return std::optional<StructuralCharacter>();
}

void StructuralIndexer::construct_escape_carry_index(const char *chunk, std::bitset<CARRY_INDEX_SIZE> *index) {
  for (size_t i = 1; i <= Engine::CHUNK_SIZE / Engine::BLOCK_SIZE; i++) {
    auto is_escape_char = chunk[i * Engine::BLOCK_SIZE] == '\\';
    if (!is_escape_char) continue;

    auto escape_char_count = 1;
    while (chunk[i * Engine::BLOCK_SIZE - escape_char_count] == '\\') {
      is_escape_char = !is_escape_char;
    }

    index->operator[](i) = is_escape_char;
  }
}

std::unique_ptr<StructuralIndex> StructuralIndexer::construct_structural_index(const char *chunk) {
  return std::make_unique<StructuralIndex>();
}

} // namespace npu

