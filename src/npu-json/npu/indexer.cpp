#include <cstring>
#include <stdexcept>
#include <immintrin.h>

#include <npu-json/npu/indexer.hpp>

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
  bo_in    = xrt::bo(device, Engine::CHUNK_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  bo_out   = xrt::bo(device, INDEX_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  bo_carry = xrt::bo(device, CARRY_INDEX_SIZE * sizeof(uint32_t),
                     XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  // Copy instructions to buffer
  memcpy(bo_instr.map<void *>(), instr_v.data(), instr_v.size() * sizeof(uint32_t));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Zero out output buffer
  memset(bo_out.map<uint8_t *>(), 0, INDEX_SIZE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

void StructuralIndexer::construct_escape_carry_index(const char *chunk, std::array<uint32_t, CARRY_INDEX_SIZE> &index) {
  for (size_t i = 1; i <= Engine::CHUNK_SIZE / Engine::BLOCK_SIZE; i++) {
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

void StructuralIndexer::construct_string_index(const char *chunk, uint64_t *index, uint32_t *escape_carries) {
  // Copy input into buffer
  auto buf_in = bo_in.map<uint8_t *>();
  memcpy(buf_in, chunk, Engine::CHUNK_SIZE);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Copy carry index into buffer
  auto buf_carry = bo_carry.map<uint32_t *>();
  memcpy(buf_carry, escape_carries, CARRY_INDEX_SIZE * sizeof(uint32_t));
  bo_carry.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = kernel(3, bo_instr, instr_size, bo_in, bo_out, bo_carry);
  run.wait();

  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto buf_out = bo_out.map<uint64_t *>();

  // String rectification (merged into memcpy)
  bool last_block_inside_string = false;
  for (size_t block = 0; block < Engine::CHUNK_SIZE / Engine::BLOCK_SIZE; block++) {
    auto vectors_in_block = Engine::BLOCK_SIZE / 64;
    for (size_t i = 0; i < vectors_in_block; i++) {
      auto idx = block * vectors_in_block + i;
      index[idx] = last_block_inside_string ? ~buf_out[idx] : buf_out[idx];
    }
    last_block_inside_string = index[(block + 1) * vectors_in_block - 1] & 1;
  }
}

void StructuralIndexer::construct_structural_character_index(const char *chunk, StructuralIndex &index) {
  constexpr unsigned int N = 64;

  const __m512i open_brace_mask = _mm512_set1_epi8('{');
  const __m512i close_brace_mask = _mm512_set1_epi8('}');

  for (size_t i = 0; i < index.string_index.size(); i++) {
    __m512i data = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&chunk[i * N]));
    __mmask64 open_brace_res = _mm512_cmpeq_epu8_mask(data, open_brace_mask);
    __mmask64 close_brace_res = _mm512_cmpeq_epu8_mask(data, close_brace_mask);
    uint64_t structurals_mask = (open_brace_res | close_brace_res) & ~index.string_index[i];
    while (structurals_mask) {
      auto structural_idx = (i * N) + trailing_zeroes(structurals_mask);
      index.structural_characters.push_back({ chunk[structural_idx], structural_idx });
      structurals_mask = structurals_mask & (structurals_mask - 1);
    }
  }
}

std::unique_ptr<StructuralIndex> StructuralIndexer::construct_structural_index(const char *chunk) {
  auto index = std::make_unique<StructuralIndex>();

  construct_escape_carry_index(chunk, index->escape_carry_index);
  construct_string_index(chunk, index->string_index.data(), index->escape_carry_index.data());
  construct_structural_character_index(chunk, *index);

  return index;
}

} // namespace npu

