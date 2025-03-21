#include <cstring>
#include <stdexcept>
#include <immintrin.h>

#include <npu-json/npu/indexer.hpp>
#include <npu-json/structural/classifier.hpp>
#include <npu-json/util/debug.hpp>

namespace npu {

uint32_t trailing_zeroes(uint64_t mask) {
  return __builtin_ctzll(mask);
}

std::optional<StructuralCharacter> StructuralIndex::get_next_structural_character() {
  if (current_pos < structural_characters.size()) {
    auto c = structural_characters[current_pos];
    current_pos++;
    return std::optional<StructuralCharacter>(c);
  }
  return std::optional<StructuralCharacter>();
}

bool StructuralIndex::ends_in_string() {
  auto last_vector = string_index[INDEX_SIZE / 8 - 1];
  return (static_cast<int64_t>(last_vector) >> 63) & 1;
}

bool StructuralIndex::ends_with_escape() {
  return escape_carry_index[CARRY_INDEX_SIZE - 1];
}

StructuralIndexer::StructuralIndexer(std::string xclbin_path, std::string insts_path) {
  // Initialize NPU
  auto [device, k] = util::init_npu(xclbin_path);
  kernel = k;

  // Setup instruction buffer
  auto instr_v = util::load_instr_sequence(insts_path);
  instr_size = instr_v.size();
  bo_instr = xrt::bo(device, instr_size * sizeof(uint32_t),
                     XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));

  // Setup input/output buffers
  size_t in_buffer_size = Engine::CHUNK_SIZE + 4 * CARRY_INDEX_SIZE;
  bo_in  = xrt::bo(device, in_buffer_size, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  bo_out = xrt::bo(device, INDEX_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  // Copy instructions to buffer
  memcpy(bo_instr.map<void *>(), instr_v.data(), instr_v.size() * sizeof(uint32_t));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Zero out output buffer
  memset(bo_out.map<uint8_t *>(), 0, INDEX_SIZE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

void StructuralIndexer::construct_escape_carry_index(const char *chunk,
    std::array<uint32_t, CARRY_INDEX_SIZE> &index, bool first_escape_carry) {
  index[0] = first_escape_carry;
  for (size_t i = 1; i < Engine::CHUNK_SIZE / Engine::BLOCK_SIZE; i++) {
    auto is_escape_char = chunk[i * Engine::BLOCK_SIZE - 1] == '\\';
    if (!is_escape_char) continue;

    auto escape_char_count = 1;
    while (chunk[(i * Engine::BLOCK_SIZE - 1) - escape_char_count] == '\\') {
      is_escape_char = !is_escape_char;
      escape_char_count++;
    }

    index[i] = is_escape_char;
  }
}

void StructuralIndexer::construct_string_index(const char *chunk, uint64_t *index,
    uint32_t *escape_carries, bool first_string_carry) {
  // Copy input into buffer
  auto buf_in = bo_in.map<uint8_t *>();
  auto blocks_in_chunk_count = Engine::CHUNK_SIZE / Engine::BLOCK_SIZE;
  for (size_t block = 0; block < blocks_in_chunk_count; block++) {
    // Each block has 4 extra bytes
    auto idx = block * (Engine::BLOCK_SIZE + 4);
    memcpy(&buf_in[idx], &chunk[block * Engine::BLOCK_SIZE], Engine::BLOCK_SIZE);
    uint32_t *buf_in_carry = (uint32_t *)&buf_in[idx + Engine::BLOCK_SIZE];
    *buf_in_carry = escape_carries[block];
  }
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = kernel(3, bo_instr, instr_size, bo_in, bo_out);
  run.wait();

  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto buf_out = bo_out.map<uint64_t *>();

  // String rectification (merged into memcpy)
  bool last_block_inside_string = first_string_carry;
  for (size_t block = 0; block < Engine::CHUNK_SIZE / Engine::BLOCK_SIZE; block++) {
    auto vectors_in_block = Engine::BLOCK_SIZE / 64;
    for (size_t i = 0; i < vectors_in_block; i++) {
      auto idx = block * vectors_in_block + i;
      index[idx] = last_block_inside_string ? ~buf_out[idx] : buf_out[idx];
    }
    auto last_vector = index[(block + 1) * vectors_in_block - 1];
    last_block_inside_string = (static_cast<int64_t>(last_vector) >> 63) & 1;
  }
}

void StructuralIndexer::construct_structural_character_index(const char *chunk, StructuralIndex &index) {
  constexpr unsigned int N = 64;

  auto classifier = structural::Classifier();
  classifier.toggle_colons_and_commas();

  auto tail = index.structural_characters.data();

  for (size_t i = 0; i < index.string_index.size(); i++) {
    uint64_t structural1 = classifier.classify_block(&chunk[i * N]);
    uint64_t structural2 = classifier.classify_block(&chunk[i * N + N / 2]);
    uint64_t structural = (structural2 << 32) | (structural1);
    auto nonquoted_structural = structural & ~index.string_index[i];

    while (nonquoted_structural) {
      auto structural_idx = (i * N) + trailing_zeroes(nonquoted_structural);
      *tail++ = { chunk[structural_idx], structural_idx };
      nonquoted_structural = nonquoted_structural & (nonquoted_structural - 1);
    }
  }
}

// TODO: Clean up this ugly global shared pointer. Allocate it once in the engine
// Perhaps the entire engine/indexer concept has to be reconsidered. (chunk is copied twice)
auto index = std::make_shared<StructuralIndex>();

std::shared_ptr<StructuralIndex> StructuralIndexer::construct_structural_index(const char *chunk,
    bool first_escape_carry, bool first_string_carry) {

  construct_escape_carry_index(chunk, index->escape_carry_index, first_escape_carry);
  construct_string_index(chunk, index->string_index.data(), index->escape_carry_index.data(), first_string_carry);
  construct_structural_character_index(chunk, *index);

  return index;
}

} // namespace npu

