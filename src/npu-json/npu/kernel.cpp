#include <cstring>
#include <immintrin.h>

#include <npu-json/npu/chunk-index.hpp>
#include <npu-json/util/debug.hpp>
#include <npu-json/engine.hpp>
#include <npu-json/options.hpp>

#include <npu-json/npu/kernel.hpp>

namespace npu {

Kernel::Kernel(std::string &json) {
  // Initialize NPU
  auto xclbin = xrt::xclbin(XCLBIN_PATH);
  auto [device, context] = util::init_npu(xclbin);

  // Setup XRT kernel objects
  string_index.kernel = xrt::kernel(context, "STRINGINDEX");
  structural_index.kernel = xrt::kernel(context, "STRUCTURALCHARACTERINDEX");

  // Setup instruction buffer (string)
  auto string_index_instr_v = util::load_instr_sequence(STRING_INDEX_INSTS_PATH);
  string_index.instr_size = string_index_instr_v.size();
  string_index.instr = xrt::bo(device, string_index.instr_size * sizeof(uint32_t),
                               XCL_BO_FLAGS_CACHEABLE, string_index.kernel.group_id(1));

  // Setup instruction buffer (structural)
  auto structural_index_instr_v = util::load_instr_sequence(STRUCTURAL_CHARACTER_INDEX_INSTS_PATH);
  structural_index.instr_size = structural_index_instr_v.size();
  structural_index.instr = xrt::bo(device, structural_index.instr_size * sizeof(uint32_t),
                                   XCL_BO_FLAGS_CACHEABLE, structural_index.kernel.group_id(1));

  // Setup input/output buffers (string)
  size_t input_buffer_size_string = CHUNK_BIT_INDEX_SIZE * 2 + 4 * CHUNK_CARRY_INDEX_SIZE;
  string_index.input  = xrt::bo(device, input_buffer_size_string, XRT_BO_FLAGS_HOST_ONLY,
                                string_index.kernel.group_id(3));
  string_index.output = xrt::bo(device, CHUNK_BIT_INDEX_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                                string_index.kernel.group_id(4));

  // Setup input/output buffers (structural character)
  // We allocate a buffer for the entire JSON and use "sub-buffers" for each chunk kernel call.
  size_t input_buffer_size_structural = (json.length() + Engine::CHUNK_SIZE - 1) / Engine::CHUNK_SIZE * Engine::CHUNK_SIZE;
  structural_index.input  = xrt::bo(device, input_buffer_size_structural, XRT_BO_FLAGS_HOST_ONLY,
                                    structural_index.kernel.group_id(3));
  structural_index.output = xrt::bo(device, CHUNK_BIT_INDEX_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                                    structural_index.kernel.group_id(4));

  // Copy instructions to buffer
  memcpy(string_index.instr.map<void *>(), string_index_instr_v.data(),
         string_index.instr_size * sizeof(uint32_t));
  string_index.instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  memcpy(structural_index.instr.map<void *>(), structural_index_instr_v.data(),
         structural_index.instr_size * sizeof(uint32_t));
  structural_index.instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Copy JSON to input buffer
  memcpy(structural_index.input.map<uint8_t *>(), json.c_str(), json.length());
  memset(structural_index.input.map<uint8_t *>() + json.length(), ' ', input_buffer_size_structural - json.length());

  // Zero out output buffers
  memset(string_index.output.map<uint8_t *>(), 0, CHUNK_BIT_INDEX_SIZE);
  string_index.output.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  memset(structural_index.output.map<uint8_t *>(), 0, CHUNK_BIT_INDEX_SIZE);
  structural_index.output.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

void Kernel::call(const char * chunk, ChunkIndex &index, bool first_string_carry, size_t chunk_idx) {
  string_index.call(chunk, index, first_string_carry);
  structural_index.call(chunk, index, chunk_idx);
}

template<char C>
void build_character_index(const char *block, uint64_t *index, size_t n) {
  constexpr const size_t N = 64;

  const __m512i mask = _mm512_set1_epi8(C);

  for (size_t i = 0; i < n; i += N) {
    auto addr = reinterpret_cast<const __m512i *>(&block[i]);
    __m512i data = _mm512_loadu_si512(addr);
    *index++ = _mm512_cmpeq_epu8_mask(data, mask);;
  }
}

void Kernel::StringIndex::call(const char * chunk, ChunkIndex &index, bool first_string_carry) {
  auto input_buf = input.map<uint8_t *>();

  constexpr const auto vectors_in_block = Engine::BLOCK_SIZE / 64;
  constexpr const auto INDEX_BLOCK_SIZE = Engine::BLOCK_SIZE / 8;

  // Copy input into buffer
  auto blocks_in_chunk_count = Engine::CHUNK_SIZE / Engine::BLOCK_SIZE;
  for (size_t block = 0; block < blocks_in_chunk_count; block++) {
    // Each block has 4 extra bytes
    auto idx = block * (INDEX_BLOCK_SIZE * 2 + 4);
    auto first_index_block = reinterpret_cast<uint64_t *>(&input_buf[idx]);
    auto second_index_block = reinterpret_cast<uint64_t *>(&input_buf[idx + INDEX_BLOCK_SIZE]);
    build_character_index<'"'>(&chunk[block * Engine::BLOCK_SIZE], first_index_block, Engine::BLOCK_SIZE);
    build_character_index<'\\'>(&chunk[block * Engine::BLOCK_SIZE], second_index_block, Engine::BLOCK_SIZE);
    uint32_t *buf_in_carry = (uint32_t *)&input_buf[idx + INDEX_BLOCK_SIZE * 2];
    *buf_in_carry = index.escape_carry_index[block];
  }


  input.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = kernel(3, instr, instr_size, input, output);
  run.wait();

  output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  auto output_buf = output.map<uint64_t *>();

  // String rectification (merged into memcpy)
  bool last_block_inside_string = first_string_carry;
  for (size_t block = 0; block < blocks_in_chunk_count; block++) {
    for (size_t i = 0; i < vectors_in_block; i++) {
      auto idx = block * vectors_in_block + i;
      index.string_index[idx] = last_block_inside_string
        ? ~output_buf[idx] : output_buf[idx];
    }
    auto last_vector = index.string_index[(block + 1) * vectors_in_block - 1];
    last_block_inside_string = static_cast<int64_t>(last_vector) >> 63;
  }
}

inline uint64_t trailing_zeroes(uint64_t mask) {
  return __builtin_ctzll(mask);
}

void Kernel::StructuralIndexBuffers::call(const char *chunk, ChunkIndex &index, size_t chunk_idx) {
  // Use sub-buffer for input
  auto sub_input = xrt::bo(input, Engine::CHUNK_SIZE, chunk_idx);

  sub_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = kernel(3, instr, instr_size, sub_input, output);
  run.wait();

  output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  auto output_buf = output.map<uint8_t *>();

  auto tail = index.structural_characters.data();
  index.structurals_count = 0;

  constexpr const size_t N = 64;
  for (size_t i = 0; i < CHUNK_BIT_INDEX_SIZE / 8; i++) {
    auto nonquoted_structural = reinterpret_cast<uint64_t *>(output_buf)[i];

    nonquoted_structural = nonquoted_structural & ~index.string_index[i];

    while (nonquoted_structural) {
      auto structural_idx = (i * N) + trailing_zeroes(nonquoted_structural);
      *tail++ = { chunk[structural_idx], structural_idx + chunk_idx };
      index.structurals_count++;
      nonquoted_structural = nonquoted_structural & (nonquoted_structural - 1);
    }
  }
}

} // namespace npu
