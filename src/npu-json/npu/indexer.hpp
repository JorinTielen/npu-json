#pragma once

#include <bitset>
#include <vector>
#include <memory>
#include <optional>

#include <npu-json/engine.hpp>
#include <npu-json/util/xrt.hpp>

namespace npu {

constexpr size_t CARRY_INDEX_SIZE = Engine::CHUNK_SIZE / Engine::BLOCK_SIZE + 1;
constexpr size_t INDEX_SIZE = Engine::CHUNK_SIZE / 8;

class StructuralIndex {
  size_t current_pos = 0;
public:
  // Has to be 32-bit per carry boolean because of NPU data-transfer limitations
  std::array<uint32_t, CARRY_INDEX_SIZE> escape_carry_index;
  std::array<uint64_t, INDEX_SIZE / 8> string_index;
  // TODO: Figure out proper storage size (check simdjson) or just use a generator style interface
  std::array<StructuralCharacter, Engine::CHUNK_SIZE> structural_characters;
  size_t structurals_count = 0;

  // Methods
  StructuralCharacter* get_next_structural_character();
  bool ends_in_string();
  bool ends_with_escape();
  void reset();
};

// Class used to build the indices required to provide a stream of structural
// characters to the JSONPath engine.
class StructuralIndexer {
  xrt::kernel kernel;

  xrt::bo bo_instr;
  xrt::bo bo_in;
  xrt::bo bo_out;
  size_t instr_size;

  bool npu_initialized = false;
public:
  StructuralIndexer(std::string xclbin_path, std::string insts_path, bool initialize_npu);
  std::shared_ptr<StructuralIndex> construct_structural_index(const char *chunk, bool, bool, size_t);
private:
  void construct_escape_carry_index(const char *chunk, std::array<uint32_t, CARRY_INDEX_SIZE> &index, bool);
  void construct_string_index(const char *chunk, uint64_t *index, uint32_t *escape_carries, bool);
  void construct_structural_character_index(const char *chunk, StructuralIndex &index, size_t chunk_idx);
};

} // namespace npu
