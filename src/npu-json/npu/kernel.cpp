#include <cstring>
#include <immintrin.h>

#include <npu-json/npu/chunk-index.hpp>
#include <npu-json/util/debug.hpp>
#include <npu-json/util/tracer.hpp>
#include <npu-json/engine.hpp>
#include <npu-json/options.hpp>

#include <npu-json/npu/kernel.hpp>

namespace npu {

Kernel::Kernel(std::string &json) {
  // Initialize NPU
  auto xclbin = xrt::xclbin(XCLBIN_PATH);
  auto [device, context] = util::init_npu(xclbin);

  // Setup XRT kernel objects
  kernel = xrt::kernel(context, "MLIR_AIE");

  // Setup instruction buffer (string)
  auto instr_v = util::load_instr_sequence(INSTS_PATH);
  instr_size = instr_v.size();
  instr = xrt::bo(device, instr_size * sizeof(uint32_t),
                  XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));

  // Setup input/output buffers (string)
  size_t input_buffer_size_string = CHUNK_BIT_INDEX_SIZE * 2 + 4 * CHUNK_CARRY_INDEX_SIZE;
  string_index.input  = xrt::bo(device, input_buffer_size_string, XRT_BO_FLAGS_HOST_ONLY,
                                kernel.group_id(4));
  string_index.output = xrt::bo(device, CHUNK_BIT_INDEX_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                                kernel.group_id(5));

  // Setup input/output buffers (structural character)
  // We allocate a buffer for the entire JSON and use "sub-buffers" for each chunk kernel call.
  size_t input_buffer_size_structural = (json.length() + Engine::CHUNK_SIZE - 1) / Engine::CHUNK_SIZE * Engine::CHUNK_SIZE;
  structural_index.input  = xrt::bo(device, input_buffer_size_structural, XRT_BO_FLAGS_HOST_ONLY,
                                    kernel.group_id(3));
  structural_index.output = xrt::bo(device, CHUNK_BIT_INDEX_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                                    kernel.group_id(6));

  // Copy instructions to buffer
  memcpy(instr.map<void *>(), instr_v.data(), instr_size * sizeof(uint32_t));
  instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Copy JSON to input buffer
  memcpy(structural_index.input.map<uint8_t *>(), json.c_str(), json.length());
  memset(structural_index.input.map<uint8_t *>() + json.length(), ' ', input_buffer_size_structural - json.length());

  // Zero out output buffers
  memset(string_index.output.map<uint8_t *>(), 0, CHUNK_BIT_INDEX_SIZE);
  string_index.output.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  memset(structural_index.output.map<uint8_t *>(), 0, CHUNK_BIT_INDEX_SIZE);
  structural_index.output.sync(XCL_BO_SYNC_BO_TO_DEVICE);
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

inline uint64_t trailing_zeroes(uint64_t mask) {
  return __builtin_ctzll(mask);
}

void Kernel::call(const char * chunk, ChunkIndex &index, bool first_string_carry, size_t chunk_idx) {
  auto& tracer = util::Tracer::get_instance();
  auto trace = tracer.start_trace("construct_combined_index");

  constexpr const auto vectors_in_block = Engine::BLOCK_SIZE / 64;
  constexpr const auto INDEX_BLOCK_SIZE = Engine::BLOCK_SIZE / 8;

  auto input_buf = string_index.input.map<uint8_t *>();

  // // Copy string index input into buffer
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

  // Use sub-buffer for JSON data input
  auto sub_input = xrt::bo(structural_index.input, Engine::CHUNK_SIZE, chunk_idx);

  auto trace_npu = tracer.start_trace("construct_combined_index_npu");

  string_index.input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  sub_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // print_input_and_index(structural_index.input.map<const char *>(), string_index.input.map<uint64_t *>());

  auto run = kernel(3, instr, instr_size, sub_input, string_index.input,
                    string_index.output, structural_index.output);
  run.wait();

  string_index.output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  structural_index.output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  tracer.finish_trace(trace_npu);

  // String index rectification of (merged into memcpy)
  auto string_index_buf = string_index.output.map<uint64_t *>();
  bool last_block_inside_string = first_string_carry;
  for (size_t block = 0; block < blocks_in_chunk_count; block++) {
    for (size_t i = 0; i < vectors_in_block; i++) {
      auto idx = block * vectors_in_block + i;
      index.string_index[idx] = last_block_inside_string
        ? ~string_index_buf[idx] : string_index_buf[idx];
    }
    auto last_vector = index.string_index[(block + 1) * vectors_in_block - 1];
    last_block_inside_string = static_cast<int64_t>(last_vector) >> 63;
  }

  // Convert strurctural bit-index into structural character stream
  constexpr const size_t N = 64;
  index.structurals_count = 0;

  auto tail = index.structural_characters.data();
  auto structural_index_buf = structural_index.output.map<uint64_t *>();
  for (size_t i = 0; i < CHUNK_BIT_INDEX_SIZE / 8; i++) {
    auto nonquoted_structural = structural_index_buf[i];

    nonquoted_structural = nonquoted_structural & ~index.string_index[i];

    while (nonquoted_structural) {
      auto structural_idx = (i * N) + trailing_zeroes(nonquoted_structural);
      *tail++ = { chunk[structural_idx], structural_idx + chunk_idx };
      index.structurals_count++;
      nonquoted_structural = nonquoted_structural & (nonquoted_structural - 1);
    }
  }

  // print_input_and_index(chunk, string_index_buf);
  // print_input_and_index(chunk, structural_index_buf);

  // std::cout << "structural count: " << index.structurals_count << std::endl;
  tracer.finish_trace(trace);
}

} // namespace npu
