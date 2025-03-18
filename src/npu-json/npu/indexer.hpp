#pragma once

#include <bitset>
#include <vector>
#include <memory>
#include <optional>

#include <npu-json/engine.hpp>
#include <npu-json/util/xrt.hpp>

namespace npu {

constexpr size_t CARRY_INDEX_SIZE = Engine::CHUNK_SIZE / Engine::BLOCK_SIZE;
constexpr size_t INDEX_SIZE = Engine::CHUNK_SIZE / 8;

class StructuralIndex {
  size_t current_pos = 0;
  bool chunk_carry_string = false;
  bool chunk_carry_escape = false;

  struct StructuralCharacter {
    char c;
    size_t pos;
  };

  std::vector<StructuralCharacter> structural_characters;
public:
  std::bitset<CARRY_INDEX_SIZE> escape_carry_index;
  std::array<uint64_t, INDEX_SIZE / 8> string_index;
  std::optional<StructuralCharacter> get_next_structural_character();
};

// Class used to build the indices required to provide a stream of structural
// characters to the JSONPath engine.
class StructuralIndexer {
  xrt::kernel kernel;

  xrt::bo bo_instr;
  xrt::bo bo_in;
  xrt::bo bo_out;
  size_t instr_size;
public:
  StructuralIndexer(std::string xclbin_path, std::string insts_path);
  std::unique_ptr<StructuralIndex> construct_structural_index(const char *chunk);
private:
  void construct_escape_carry_index(const char *chunk, std::bitset<CARRY_INDEX_SIZE> &index);
  void construct_string_index(const char *chunk, uint64_t *index);
};

} // namespace npu
