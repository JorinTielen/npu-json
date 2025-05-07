#include <cstring>
#include <stdexcept>
#include <immintrin.h>

#include <npu-json/npu/indexer.hpp>
#include <npu-json/structural/classifier.hpp>
#include <npu-json/util/debug.hpp>
#include <npu-json/util/tracer.hpp>
#include <npu-json/options.hpp>

namespace npu {

uint32_t trailing_zeroes(uint64_t mask) {
  return __builtin_ctzll(mask);
}

StructuralCharacter* StructuralIndex::get_next_structural_character() {
  if (current_pos < structurals_count) {
    auto ptr = &structural_characters[current_pos];
    current_pos++;
    return ptr;
  }
  return nullptr;
}

bool StructuralIndex::ends_in_string() {
  auto last_vector = string_index[INDEX_SIZE / 8 - 1];
  return (static_cast<int64_t>(last_vector) >> 63) & 1;
}

bool StructuralIndex::ends_with_escape() {
  return escape_carry_index[CARRY_INDEX_SIZE - 1];
}

void StructuralIndex::reset() {
  current_pos = 0;
  structurals_count = 0;
}

StructuralIndexer::StructuralIndexer(bool initialize_npu = true) {
  // If the flag is not passed we skip setting up the NPU. Used in unit tests.
  if (!initialize_npu) return;

  // Initialize NPU
  auto xclbin = xrt::xclbin(XCLBIN_PATH);
  auto [device, context] = util::init_npu(xclbin);

  string_index_kernel = xrt::kernel(context, "STRINGINDEX");
  structural_character_index_kernel = xrt::kernel(context, "STRUCTURALCHARACTERINDEX");

  // Setup instruction buffer (string)
  // auto instr1_v = util::load_instr_sequence(STRING_INDEX_INSTS_PATH);
  // instr1_size = instr1_v.size();
  // bo_instr1 = xrt::bo(device, instr1_size * sizeof(uint32_t),
  //                     XCL_BO_FLAGS_CACHEABLE, string_index_kernel.group_id(1));
  // auto instr2_v = util::load_instr_sequence(STRUCTURAL_CHARACTER_INDEX_INSTS_PATH);
  // instr2_size = instr2_v.size();
  // bo_instr2 = xrt::bo(device, instr2_size * sizeof(uint32_t),
  //                     XCL_BO_FLAGS_CACHEABLE, structural_character_index_kernel.group_id(1));

  // Setup input/output buffers (string)
  size_t in_buffer_size_string = Engine::CHUNK_SIZE + 4 * CARRY_INDEX_SIZE;
  bo_in1  = xrt::bo(device, in_buffer_size_string, XRT_BO_FLAGS_HOST_ONLY, string_index_kernel.group_id(3));
  bo_out1 = xrt::bo(device, INDEX_SIZE, XRT_BO_FLAGS_HOST_ONLY, string_index_kernel.group_id(4));

  // Setup input/output buffers (structural character)
  size_t in_buffer_size_structural = Engine::CHUNK_SIZE + INDEX_SIZE;
  bo_in2  = xrt::bo(device, in_buffer_size_structural, XRT_BO_FLAGS_HOST_ONLY, structural_character_index_kernel.group_id(3));
  bo_out2 = xrt::bo(device, INDEX_SIZE, XRT_BO_FLAGS_HOST_ONLY, structural_character_index_kernel.group_id(4));

  // Copy instructions to buffer
  // memcpy(bo_instr1.map<void *>(), instr1_v.data(), instr1_v.size() * sizeof(uint32_t));
  // bo_instr1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  // memcpy(bo_instr2.map<void *>(), instr2_v.data(), instr2_v.size() * sizeof(uint32_t));
  // bo_instr2.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Zero out output buffers
  memset(bo_out1.map<uint8_t *>(), 0, INDEX_SIZE);
  bo_out1.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  memset(bo_out2.map<uint8_t *>(), 0, INDEX_SIZE);
  bo_out2.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  npu_initialized = true;
}

void StructuralIndexer::construct_escape_carry_index(const char *chunk,
    std::array<uint32_t, CARRY_INDEX_SIZE> &index, bool first_escape_carry) {
  auto& tracer = util::Tracer::get_instance();
  auto trace = tracer.start_trace("construct_escape_carry_index");

  index[0] = first_escape_carry;
  for (size_t i = 1; i <= Engine::CHUNK_SIZE / Engine::BLOCK_SIZE; i++) {
    auto is_escape_char = chunk[i * Engine::BLOCK_SIZE - 1] == '\\';
    if (!is_escape_char) {
      index[i] = false;
      continue;
    }

    auto escape_char_count = 1;
    while (chunk[(i * Engine::BLOCK_SIZE - 1) - escape_char_count] == '\\') {
      is_escape_char = !is_escape_char;
      escape_char_count++;
    }

    index[i] = is_escape_char;
  }

  tracer.finish_trace(trace);
}

uint64_t prefix_xor_clmul(const uint64_t bitmask) {
  __m128i all_ones = _mm_set1_epi8('\xFF');
  __m128i result = _mm_clmulepi64_si128(_mm_set_epi64x(0ULL, bitmask), all_ones, 0);
  return _mm_cvtsi128_si64(result);
}

void construct_string_index_avx512(const char *chunk, uint64_t *index,
    uint32_t *escape_carries, bool first_string_carry) {
  static constexpr const uint64_t V = 64;
  static constexpr const uint64_t ODD_BITS = 0xAAAAAAAAAAAAAAAAULL;
  static constexpr const uint64_t ALL_ONES = 0xFFFFFFFFFFFFFFFFULL;

  auto& tracer = util::Tracer::get_instance();
  auto trace = tracer.start_trace("construct_string_index");

  // Static masks containing only quote or escape (backslash) characters
  const __m512i quotes_mask = _mm512_set1_epi8('"');
  const __m512i backslash_mask = _mm512_set1_epi8('\\');

  uint64_t prev_in_string = first_string_carry ? ALL_ONES : 0;
  uint64_t prev_is_escaped = escape_carries[0];

  size_t index_idx = 0;

  for (size_t i = 0; i < Engine::CHUNK_SIZE; i += V) {
    // Scan for quote and escape characters in input
    __m512i data = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&chunk[i]));
    __mmask64 quotes = _mm512_cmpeq_epu8_mask(data, quotes_mask);
    __mmask64 backslash = _mm512_cmpeq_epu8_mask(data, backslash_mask);

    // Find odd-length sequences of escape characters (Fig. 3)
    uint64_t potential_escape = backslash & ~prev_is_escaped;
    uint64_t maybe_escaped = potential_escape << 1;

    uint64_t maybe_escaped_and_odd_bits     = maybe_escaped | ODD_BITS;
    uint64_t even_series_codes_and_odd_bits = maybe_escaped_and_odd_bits - potential_escape;

    uint64_t escape_and_terminal_code = even_series_codes_and_odd_bits ^ ODD_BITS;
    uint64_t escaped = escape_and_terminal_code ^ (backslash | prev_is_escaped);
    uint64_t escape = escape_and_terminal_code & backslash;
    prev_is_escaped = escape >> 63;

    // Compute string-index (Fig. 4)
    uint64_t non_escaped_quotes = quotes & ~escaped;
    uint64_t string_index = prefix_xor_clmul(non_escaped_quotes);

    // Invert if we were in a string previously
    string_index = string_index ^ prev_in_string;
    prev_in_string = static_cast<int64_t>(string_index) >> 63;

    // Store result
    index[index_idx] = string_index;
    index_idx++;
  }

  tracer.finish_trace(trace);
}

void StructuralIndexer::construct_string_index(const char *chunk, uint64_t *index,
    uint32_t *escape_carries, bool first_string_carry) {
  // If the NPU is not initialized, we will not call a kernel here.
  if (!npu_initialized) throw std::logic_error("NPU was not initialized");

  auto& tracer = util::Tracer::get_instance();
  auto trace = tracer.start_trace("construct_string_index");

  // Copy input into buffer
  auto buf_in = bo_in1.map<uint8_t *>();
  auto blocks_in_chunk_count = Engine::CHUNK_SIZE / Engine::BLOCK_SIZE;
  for (size_t block = 0; block < blocks_in_chunk_count; block++) {
    // Each block has 4 extra bytes
    auto idx = block * (Engine::BLOCK_SIZE + 4);
    memcpy(&buf_in[idx], &chunk[block * Engine::BLOCK_SIZE], Engine::BLOCK_SIZE);
    uint32_t *buf_in_carry = (uint32_t *)&buf_in[idx + Engine::BLOCK_SIZE];
    *buf_in_carry = escape_carries[block];
  }
  bo_in1.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto trace_npu = tracer.start_trace("npu-string");

  auto run = string_index_kernel(3, bo_instr1, instr1_size, bo_in1, bo_out1);
  run.wait();

  tracer.finish_trace(trace_npu);

  bo_out1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto buf_out = bo_out1.map<uint64_t *>();

  // String rectification (merged into memcpy)
  bool last_block_inside_string = first_string_carry;
  for (size_t block = 0; block < blocks_in_chunk_count; block++) {
    auto vectors_in_block = Engine::BLOCK_SIZE / 64;
    for (size_t i = 0; i < vectors_in_block; i++) {
      auto idx = block * vectors_in_block + i;
      index[idx] = last_block_inside_string ? ~buf_out[idx] : buf_out[idx];
    }
    auto last_vector = index[(block + 1) * vectors_in_block - 1];
    last_block_inside_string = static_cast<int64_t>(last_vector) >> 63;
  }

  tracer.finish_trace(trace);
}

void construct_structural_character_index_avx512(
  const char *chunk,
  StructuralIndex &index,
  size_t chunk_idx
) {
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
      *tail++ = { chunk[structural_idx], structural_idx + chunk_idx };
      index.structurals_count++;
      nonquoted_structural = nonquoted_structural & (nonquoted_structural - 1);
    }
  }
}

void StructuralIndexer::construct_structural_character_index(
  const char *chunk,
  StructuralIndex &index,
  size_t chunk_idx
) {
  // If the NPU is not initialized, we will not call a kernel here.
  if (!npu_initialized) throw std::logic_error("NPU was not initialized");

  auto& tracer = util::Tracer::get_instance();
  auto trace = tracer.start_trace("construct_structural_character_index");

  // Copy input into buffer
  auto buf_in = bo_in2.map<uint8_t *>();
  auto blocks_in_chunk_count = Engine::CHUNK_SIZE / Engine::BLOCK_SIZE;
  for (size_t block = 0; block < blocks_in_chunk_count; block++) {
    // Each block has string_index after input
    auto idx = block * (Engine::BLOCK_SIZE + (Engine::BLOCK_SIZE / 8));
    memcpy(&buf_in[idx], &chunk[block * Engine::BLOCK_SIZE], Engine::BLOCK_SIZE);
    memcpy(&buf_in[idx + Engine::BLOCK_SIZE], &index.string_index.data()[block * (Engine::BLOCK_SIZE / 64)], Engine::BLOCK_SIZE / 8);
  }
  bo_in2.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto trace_npu = tracer.start_trace("npu-structural");

  auto run = structural_character_index_kernel(3, bo_instr2, instr2_size, bo_in2, bo_out2);
  run.wait();

  tracer.finish_trace(trace_npu);

  bo_out2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto buf_out = bo_out2.map<uint8_t *>();

  auto tail = index.structural_characters.data();
  // for (size_t block = 0; block < blocks_in_chunk_count; block++) {
  //   auto block_idx = block * Engine::BLOCK_SIZE;

  //   for (size_t i = 0; i < Engine::BLOCK_SIZE; i++) {
  //     auto structural_pos = *(uint32_t*)(buf_out + block_idx + i);
  //     std::cout << "chunk=" << chunk_idx << ", block=" << block_idx << ", i=" << i << ": " << structural_pos << std::endl;
  //     if (!structural_pos) break;
  //     *tail++ = { chunk[structural_pos], i + chunk_idx + block_idx };
  //   }
  // }

  constexpr unsigned int N = 64;
  auto classifier = structural::Classifier();
  classifier.toggle_colons_and_commas();

  for (size_t i = 0; i < INDEX_SIZE / 8; i++) {
    auto nonquoted_structural = reinterpret_cast<uint64_t *>(buf_out)[i];


    // uint64_t cpu_structural1 = classifier.classify_block(&chunk[i * N]);
    // uint64_t cpu_structural2 = classifier.classify_block(&chunk[i * N + N / 2]);
    // uint64_t cpu_structural = (cpu_structural2 << 32) | (cpu_structural1);
    // auto cpu_nonquoted_structural = cpu_structural & ~index.string_index[i];

    while (nonquoted_structural) {
      auto structural_idx = (i * N) + trailing_zeroes(nonquoted_structural);
      if (structural_idx + chunk_idx == 378740800) {
        std::cout << "structural_character_index:" << std::endl;
        // print_input_and_index(&chunk[(i - 1) * N], &reinterpret_cast<uint64_t *>(buf_out)[i - 1]);
        print_input_and_index(&chunk[i * N], &index.string_index[i]);
        print_input_and_index(&chunk[i * N], &reinterpret_cast<uint64_t *>(buf_out)[i]);
      }
      *tail++ = { chunk[structural_idx], structural_idx + chunk_idx };
      index.structurals_count++;
      nonquoted_structural = nonquoted_structural & (nonquoted_structural - 1);
    }
  }

  tracer.finish_trace(trace);
}

// TODO: Clean up this ugly global shared pointer. Allocate it once in the engine
// Perhaps the entire engine/indexer concept has to be reconsidered. (chunk is copied twice)
auto index = std::make_shared<StructuralIndex>();

std::shared_ptr<StructuralIndex> StructuralIndexer::construct_structural_index(
  const char *chunk,
  bool first_escape_carry,
  bool first_string_carry,
  size_t chunk_idx
) {
  index->reset();

  construct_escape_carry_index(chunk, index->escape_carry_index, first_escape_carry);
  if (npu_initialized) {
    construct_string_index(chunk, index->string_index.data(), index->escape_carry_index.data(), first_string_carry);
    construct_structural_character_index(chunk, *index, chunk_idx);
  } else {
    construct_string_index_avx512(chunk, index->string_index.data(), index->escape_carry_index.data(), first_string_carry);
    construct_structural_character_index_avx512(chunk, *index, chunk_idx);
  }

  return index;
}

} // namespace npu

