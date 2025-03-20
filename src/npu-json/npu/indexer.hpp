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

struct StructuralCharacter {
  char c;
  size_t pos;
};

class StructuralIndex {
  size_t current_pos = 0;
public:
  // Has to be 32-bit per carry boolean because of NPU data-transfer limitations
  std::array<uint32_t, CARRY_INDEX_SIZE> escape_carry_index;
  std::array<uint64_t, INDEX_SIZE / 8> string_index;
  std::vector<StructuralCharacter> structural_characters;

  // Methods
  std::optional<StructuralCharacter> get_next_structural_character();
  bool ends_in_string();
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
  std::unique_ptr<StructuralIndex> construct_structural_index(const char *chunk, bool, bool);
private:
  void construct_escape_carry_index(const char *chunk, std::array<uint32_t, CARRY_INDEX_SIZE> &index, bool);
  void construct_string_index(const char *chunk, uint64_t *index, uint32_t *escape_carries, bool);
  void construct_structural_character_index(const char *chunk, StructuralIndex &index);
};

} // namespace npu
