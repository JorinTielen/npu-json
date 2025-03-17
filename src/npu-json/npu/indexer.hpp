#pragma once

#include <bitset>
#include <vector>
#include <memory>
#include <optional>

#include <npu-json/engine.hpp>

namespace npu {

constexpr size_t CARRY_INDEX_SIZE = Engine::CHUNK_SIZE / Engine::BLOCK_SIZE;
constexpr size_t INDEX_SIZE = Engine::CHUNK_SIZE / 8;

class StructuralIndex {

  size_t current_pos = 0;

  struct StructuralCharacter {
    char c;
    size_t pos;
  };

 std::bitset<CARRY_INDEX_SIZE> escape_carry_index;

  std::vector<StructuralCharacter> structural_characters;
public:
  std::optional<StructuralCharacter> get_next_structural_character();
};

// Class used to build the indices required to provide a stream of structural
// characters to the JSONPath engine.
class StructuralIndexer {
  static constexpr bool use_npu = false;
public:
  std::unique_ptr<StructuralIndex> construct_structural_index(const char *chunk);
private:
  void construct_escape_carry_index(const char *chunk, std::bitset<CARRY_INDEX_SIZE> *index);
};

} // namespace npu
