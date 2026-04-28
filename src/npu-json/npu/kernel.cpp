#include <cstring>
#include <immintrin.h>
#include <stdexcept>

#include <npu-json/npu/chunk-index.hpp>
#include <npu-json/util/debug.hpp>
#include <npu-json/util/tracer.hpp>
#include <npu-json/engine.hpp>
#include <npu-json/options.hpp>

#include <npu-json/npu/kernel.hpp>

namespace npu {

__attribute__((always_inline)) inline uint64_t count_ones(uint64_t mask) {
  return __builtin_popcountll(mask);
}

__attribute__((always_inline)) inline uint64_t prefix_xor(uint64_t bitmask) {
  bitmask ^= bitmask << 1;
  bitmask ^= bitmask << 2;
  bitmask ^= bitmask << 4;
  bitmask ^= bitmask << 8;
  bitmask ^= bitmask << 16;
  bitmask ^= bitmask << 32;
  return bitmask;
}

__attribute((always_inline)) inline void write_structural_index(
  uint32_t *tail,
  uint64_t bits,
  const size_t position,
  const size_t count
) {
  if (bits == 0) {
    return;
  }

  const __m512i indexes = _mm512_maskz_compress_epi8(bits, _mm512_set_epi32(
    0x3f3e3d3c, 0x3b3a3938, 0x37363534, 0x33323130,
    0x2f2e2d2c, 0x2b2a2928, 0x27262524, 0x23222120,
    0x1f1e1d1c, 0x1b1a1918, 0x17161514, 0x13121110,
    0x0f0e0d0c, 0x0b0a0908, 0x07060504, 0x03020100
  ));
  const __m512i start_index = _mm512_set1_epi32(position);

  __m512i t0 = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(indexes));
  _mm512_storeu_si512(tail, _mm512_add_epi32(t0, start_index));

  if (count > 16) {
    const __m512i t1 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(indexes, 1));
    _mm512_storeu_si512(tail + 16, _mm512_add_epi32(t1, start_index));
    if (count > 32) {
      const __m512i t2 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(indexes, 2));
      _mm512_storeu_si512(tail + 32, _mm512_add_epi32(t2, start_index));
      if (count > 48) {
        const __m512i t3 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(indexes, 3));
        _mm512_storeu_si512(tail + 48, _mm512_add_epi32(t3, start_index));
      }
    }
  }
}

void construct_escape_carry_index(const char *chunk, ChunkIndex &index, bool first_escape_carry) {
  auto &tracer = util::Tracer::get_instance();
  auto trace = tracer.start_trace("construct_escape_carry_index");

  index.escape_carry_index[0] = first_escape_carry;
  for (size_t i = 1; i <= Engine::CHUNK_SIZE / Engine::BLOCK_SIZE; i++) {
    auto is_escape_char = chunk[i * Engine::BLOCK_SIZE - 1] == '\\';
    if (!is_escape_char) {
      index.escape_carry_index[i] = false;
      continue;
    }

    auto escape_char_count = 1;
    while (chunk[(i * Engine::BLOCK_SIZE - 1) - escape_char_count] == '\\') {
      is_escape_char = !is_escape_char;
      escape_char_count++;
    }

    index.escape_carry_index[i] = is_escape_char;
  }

  tracer.finish_trace(trace);
}

static inline bool is_structural_char(char c) {
  switch (c) {
    case '{': case '}': case '[': case ']': case ':': case ',':
      return true;
    default:
      return false;
  }
}

#ifdef NPU_JSON_CPU_BACKEND

Kernel::Kernel(std::string_view json) {
  auto input_buffer_size_structural =
    (json.length() + Engine::CHUNK_SIZE - 1) / Engine::CHUNK_SIZE * Engine::CHUNK_SIZE;
  json_data.resize(input_buffer_size_structural, static_cast<uint8_t>(' '));

  if (!json.empty()) {
    memcpy(json_data.data(), json.begin(), json.length());
  }
}

void Kernel::construct_combined_index(
  const char *chunk,
  ChunkIndex &index,
  bool first_escape_carry,
  bool first_string_carry,
  size_t chunk_idx
) {
  auto &tracer = util::Tracer::get_instance();
  auto trace = tracer.start_trace("construct_combined_index_cpu");

  constexpr const size_t VECTOR_BYTES = 64;
  constexpr const size_t VECTORS_IN_CHUNK = CHUNK_BIT_INDEX_SIZE / 8;
  static constexpr const uint64_t ODD_BITS = 0xAAAAAAAAAAAAAAAAULL;

  const __m512i quote_mask = _mm512_set1_epi8('"');
  const __m512i slash_mask = _mm512_set1_epi8('\\');
  const __m512i brace_open_mask = _mm512_set1_epi8('{');
  const __m512i brace_close_mask = _mm512_set1_epi8('}');
  const __m512i bracket_open_mask = _mm512_set1_epi8('[');
  const __m512i bracket_close_mask = _mm512_set1_epi8(']');
  const __m512i colon_mask = _mm512_set1_epi8(':');
  const __m512i comma_mask = _mm512_set1_epi8(',');

  construct_escape_carry_index(chunk, index, first_escape_carry);

  index.block.structural_characters_count = 0;
  auto tail = index.block.structural_characters.data();

  uint64_t prev_in_string = first_string_carry ? ~uint64_t(0) : uint64_t(0);
  uint64_t prev_is_escaped = first_escape_carry ? 1 : 0;

  for (size_t i = 0; i < VECTORS_IN_CHUNK; i++) {
    const auto *addr = reinterpret_cast<const __m512i *>(&chunk[i * VECTOR_BYTES]);
    const __m512i data = _mm512_loadu_si512(addr);

    const uint64_t quotes = _mm512_cmpeq_epu8_mask(data, quote_mask);
    const uint64_t backslash = _mm512_cmpeq_epu8_mask(data, slash_mask);

    uint64_t potential_escape = backslash & ~prev_is_escaped;
    uint64_t maybe_escaped = potential_escape << 1;
    uint64_t maybe_escaped_and_odd_bits = maybe_escaped | ODD_BITS;
    uint64_t even_series_codes_and_odd_bits = maybe_escaped_and_odd_bits - potential_escape;

    uint64_t escape_and_terminal_code = even_series_codes_and_odd_bits ^ ODD_BITS;
    uint64_t escaped = escape_and_terminal_code ^ (backslash | prev_is_escaped);
    uint64_t escape = escape_and_terminal_code & backslash;
    prev_is_escaped = escape >> 63;

    uint64_t non_escaped_quotes = quotes & ~escaped;
    uint64_t string_index = prefix_xor(non_escaped_quotes);
    string_index ^= prev_in_string;
    prev_in_string = static_cast<int64_t>(string_index) >> 63;
    index.string_index[i] = string_index;

    uint64_t braces = _mm512_cmpeq_epu8_mask(data, brace_open_mask) |
                      _mm512_cmpeq_epu8_mask(data, brace_close_mask);
    uint64_t brackets = _mm512_cmpeq_epu8_mask(data, bracket_open_mask) |
                        _mm512_cmpeq_epu8_mask(data, bracket_close_mask);
    uint64_t colons_and_commas = _mm512_cmpeq_epu8_mask(data, colon_mask) |
                                 _mm512_cmpeq_epu8_mask(data, comma_mask);
    uint64_t structural_index = braces | brackets | colons_and_commas;
    uint64_t nonquoted_structural = structural_index & ~string_index;

    if (nonquoted_structural == 0) {
      continue;
    }

    const auto count = count_ones(nonquoted_structural);
    write_structural_index(tail, nonquoted_structural, i * VECTOR_BYTES + chunk_idx, count);
    index.block.structural_characters_count += count;
    tail += count;
  }

  tracer.finish_trace(trace);
}

void Kernel::call(ChunkIndex *index, size_t chunk_idx, std::function<void()> callback) {
  auto chunk = reinterpret_cast<const char *>(json_data.data() + chunk_idx);

  construct_combined_index(
    chunk,
    *index,
    previous_escape_carry,
    previous_string_carry,
    chunk_idx
  );

  previous_escape_carry = index->ends_with_escape();
  previous_string_carry = index->ends_in_string();

  callback();
}

void Kernel::wait_for_previous() {}

#else

static inline __attribute__((always_inline))
void build_dual_character_index(const char *block, uint64_t *idx_quote, uint64_t *idx_slash) {
  constexpr const size_t N = 64;

  const __m512i mask_quote = _mm512_set1_epi8('"');
  const __m512i mask_slash = _mm512_set1_epi8('\\');

  for (size_t i = 0; i < Engine::BLOCK_SIZE; i += N) {
    auto addr = reinterpret_cast<const __m512i *>(&block[i]);
    __m512i data = _mm512_loadu_si512(addr);

    *idx_quote++ = _mm512_cmpeq_epu8_mask(data, mask_quote);
    *idx_slash++ = _mm512_cmpeq_epu8_mask(data, mask_slash);
  }
}

Kernel::Kernel(std::string_view json) {
  // Compute page-alignment offset for zero-copy remapping
  auto addr = reinterpret_cast<uintptr_t>(json.data());
  prefix_size = (PAGE_ALIGN - (addr % PAGE_ALIGN)) % PAGE_ALIGN;
  if (prefix_size >= json.length()) {
    prefix_size = 0;
  }

  prefix_start = json.data();
  aligned_start = json.data() + prefix_size;
  aligned_length = json.length() - prefix_size;

  full_chunk_count = aligned_length / Engine::CHUNK_SIZE;
  size_t tail_size = aligned_length % Engine::CHUNK_SIZE;
  has_tail = tail_size > 0;
  size_t full_chunks_size = full_chunk_count * Engine::CHUNK_SIZE;

  // Initialize NPU
  auto xclbin = xrt::xclbin(XCLBIN_PATH);
  auto [device, context] = util::init_npu(xclbin);

  kernel = xrt::kernel(context, "MLIR_AIE");

  // Setup instruction buffer
  auto instr_v = util::load_instr_sequence(INSTS_PATH);
  instr_size = instr_v.size();
  instr = xrt::bo(device, instr_size * sizeof(uint32_t),
                  XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));

  // Setup input/output buffers (string)
  size_t input_buffer_size_string = CHUNK_BIT_INDEX_SIZE * 2 + 4 * CHUNK_CARRY_INDEX_SIZE;
  string_buffers[0].input = xrt::bo(device, input_buffer_size_string, XRT_BO_FLAGS_HOST_ONLY,
                                    kernel.group_id(4));
  string_buffers[0].output = xrt::bo(device, CHUNK_BIT_INDEX_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                                     kernel.group_id(5));
  string_buffers[1].input = xrt::bo(device, input_buffer_size_string, XRT_BO_FLAGS_HOST_ONLY,
                                    kernel.group_id(4));
  string_buffers[1].output = xrt::bo(device, CHUNK_BIT_INDEX_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                                     kernel.group_id(5));

  // Setup structural output buffers
  structural_buffers[0].output = xrt::bo(device, CHUNK_BIT_INDEX_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                                         kernel.group_id(6));
  structural_buffers[1].output = xrt::bo(device, CHUNK_BIT_INDEX_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                                         kernel.group_id(6));

  // Setup JSON data buffer (structural input) with zero-copy remapping
  size_t total_npu_chunks = full_chunk_count + (has_tail ? 1 : 0);
  if (full_chunks_size > 0) {
    json_data_input = xrt::bo(device, const_cast<char *>(aligned_start), full_chunks_size,
                              XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    json_chunk_inputs.reserve(total_npu_chunks);
    for (size_t i = 0; i < full_chunk_count; i++) {
      json_chunk_inputs.emplace_back(json_data_input, Engine::CHUNK_SIZE,
                                     i * Engine::CHUNK_SIZE);
    }
  } else {
    json_data_input = xrt::bo(device, Engine::CHUNK_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                              kernel.group_id(3));
  }

  // Setup tail buffer if needed
  if (has_tail) {
    tail_input = xrt::bo(device, Engine::CHUNK_SIZE, XRT_BO_FLAGS_HOST_ONLY,
                         kernel.group_id(3));
    tail_map = tail_input.map<uint8_t *>();
    auto src = aligned_start + full_chunks_size;
    memcpy(tail_map, src, tail_size);
    memset(tail_map + tail_size, ' ', Engine::CHUNK_SIZE - tail_size);
    tail_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  // Copy instructions to buffer
  memcpy(instr.map<void *>(), instr_v.data(), instr_size * sizeof(uint32_t));
  instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  json_data_map = json_data_input.map<uint8_t *>();
  string_input_maps[0] = string_buffers[0].input.map<uint8_t *>();
  string_input_maps[1] = string_buffers[1].input.map<uint8_t *>();
  string_output_maps[0] = string_buffers[0].output.map<uint64_t *>();
  string_output_maps[1] = string_buffers[1].output.map<uint64_t *>();
  structural_output_maps[0] = structural_buffers[0].output.map<uint64_t *>();
  structural_output_maps[1] = structural_buffers[1].output.map<uint64_t *>();

  // Zero out output buffers
  string_buffers[0].output.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  string_buffers[1].output.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

void Kernel::prepare_kernel_input(
  const char *chunk,
  ChunkIndex &index,
  bool first_escape_carry,
  size_t buffer
) {
  auto &tracer = util::Tracer::get_instance();

  auto input_buf = string_input_maps[buffer];
  constexpr const auto INDEX_BLOCK_SIZE = Engine::BLOCK_SIZE / 8;
  constexpr const auto BLOCKS_IN_CHUNK_COUNT = Engine::CHUNK_SIZE / Engine::BLOCK_SIZE;

  construct_escape_carry_index(chunk, index, first_escape_carry);

  auto trace = tracer.start_trace("prepare_kernel_input");

  for (size_t block = 0; block < BLOCKS_IN_CHUNK_COUNT; block++) {
    auto idx = block * (INDEX_BLOCK_SIZE * 2 + 4);
    auto first_index_block = reinterpret_cast<uint64_t *>(&input_buf[idx]);
    auto second_index_block = reinterpret_cast<uint64_t *>(&input_buf[idx + INDEX_BLOCK_SIZE]);
    build_dual_character_index(
      &chunk[block * Engine::BLOCK_SIZE],
      first_index_block,
      second_index_block
    );
    uint32_t *buf_in_carry = reinterpret_cast<uint32_t *>(&input_buf[idx + INDEX_BLOCK_SIZE * 2]);
    *buf_in_carry = index.escape_carry_index[block];
  }

  tracer.finish_trace(trace);
}

void Kernel::read_kernel_output(ChunkIndex &index, bool first_string_carry, size_t chunk_idx) {
  auto &tracer = util::Tracer::get_instance();

  constexpr const auto VECTORS_IN_BLOCK = Engine::BLOCK_SIZE / 64;
  constexpr const auto BLOCKS_IN_CHUNK_COUNT = Engine::CHUNK_SIZE / Engine::BLOCK_SIZE;

  auto trace = tracer.start_trace("read_kernel_output");

  auto output_buffer = !current;
  auto string_index_buf = string_output_maps[output_buffer];
  bool last_block_inside_string = first_string_carry;
  for (size_t block = 0; block < BLOCKS_IN_CHUNK_COUNT; block++) {
    for (size_t i = 0; i < VECTORS_IN_BLOCK; i++) {
      auto idx = block * VECTORS_IN_BLOCK + i;
      index.string_index[idx] =
        string_index_buf[idx] ^ (-static_cast<uint64_t>(last_block_inside_string));
    }
    auto last_vector = index.string_index[(block + 1) * VECTORS_IN_BLOCK - 1];
    last_block_inside_string = static_cast<int64_t>(last_vector) >> 63;
  }

  constexpr const size_t N = 64;

  auto structural_index_buf = structural_output_maps[output_buffer];
  auto tail = index.block.structural_characters.data();
  index.block.structural_characters_count = 0;
  constexpr auto total_size = CHUNK_BIT_INDEX_SIZE / 8;
  constexpr auto blocks_per_chunk = StructuralCharacterBlock::BLOCKS_PER_CHUNK;
  constexpr auto block_index_size = total_size / blocks_per_chunk;

  size_t i = 0;
  for (; i + 3 < block_index_size; i += 4) {
    auto pos = i;

    uint64_t s0 = structural_index_buf[pos];
    uint64_t q0 = index.string_index[pos];
    uint64_t r0 = s0 & ~q0;

    uint64_t s1 = structural_index_buf[pos + 1];
    uint64_t q1 = index.string_index[pos + 1];
    uint64_t r1 = s1 & ~q1;

    uint64_t s2 = structural_index_buf[pos + 2];
    uint64_t q2 = index.string_index[pos + 2];
    uint64_t r2 = s2 & ~q2;

    uint64_t s3 = structural_index_buf[pos + 3];
    uint64_t q3 = index.string_index[pos + 3];
    uint64_t r3 = s3 & ~q3;

    if ((r0 | r1 | r2 | r3) == 0) {
      continue;
    }

    if (r0) {
      const auto count = count_ones(r0);
      write_structural_index(tail, r0, pos * N + chunk_idx, count);
      index.block.structural_characters_count += count;
      tail += count;
    }

    if (r1) {
      const auto count = count_ones(r1);
      write_structural_index(tail, r1, (pos + 1) * N + chunk_idx, count);
      index.block.structural_characters_count += count;
      tail += count;
    }

    if (r2) {
      const auto count = count_ones(r2);
      write_structural_index(tail, r2, (pos + 2) * N + chunk_idx, count);
      index.block.structural_characters_count += count;
      tail += count;
    }

    if (r3) {
      const auto count = count_ones(r3);
      write_structural_index(tail, r3, (pos + 3) * N + chunk_idx, count);
      index.block.structural_characters_count += count;
      tail += count;
    }
  }

  for (; i < block_index_size; i++) {
    auto pos = i;
    auto nonquoted_structural = structural_index_buf[pos] & ~index.string_index[pos];

    if (nonquoted_structural == 0) {
      continue;
    }

    const auto count = count_ones(nonquoted_structural);
    write_structural_index(tail, nonquoted_structural, pos * N + chunk_idx, count);
    index.block.structural_characters_count += count;
    tail += count;
  }

  tracer.finish_trace(trace);
}

void Kernel::call(ChunkIndex *index, size_t chunk_idx, std::function<void()> callback) {
  auto &tracer = util::Tracer::get_instance();

  // Handle CPU prefix chunk
  if (chunk_idx == 0 && prefix_size > 0) {
    auto trace_prefix = tracer.start_trace("construct_cpu_prefix");

    const char *chunk = prefix_start;

    constexpr const size_t VECTOR_BYTES = 64;
    static constexpr const uint64_t ODD_BITS = 0xAAAAAAAAAAAAAAAAULL;

    const __m512i quote_mask = _mm512_set1_epi8('"');
    const __m512i slash_mask = _mm512_set1_epi8('\\');
    const __m512i brace_open_mask = _mm512_set1_epi8('{');
    const __m512i brace_close_mask = _mm512_set1_epi8('}');
    const __m512i bracket_open_mask = _mm512_set1_epi8('[');
    const __m512i bracket_close_mask = _mm512_set1_epi8(']');
    const __m512i colon_mask = _mm512_set1_epi8(':');
    const __m512i comma_mask = _mm512_set1_epi8(',');

    // Zero-initialize only escape_carry_index (small, 513 bytes)
    // string_index doesn't need initialization since prefix code doesn't read it
    index->escape_carry_index.fill(false);
    index->escape_carry_index[0] = previous_escape_carry;

    // Compute escape carry at end of prefix: check if last byte is an
    // unescaped backslash (odd-length backslash sequence)
    bool prefix_end_carry = false;
    if (prefix_size > 0 && chunk[prefix_size - 1] == '\\') {
      bool is_escaped = true;
      int count = 1;
      while (count < static_cast<int>(prefix_size) &&
             chunk[prefix_size - 1 - count] == '\\') {
        is_escaped = !is_escaped;
        count++;
      }
      prefix_end_carry = is_escaped;
    }

    index->block.structural_characters_count = 0;
    auto tail_pos = index->block.structural_characters.data();

    uint64_t prev_in_string = previous_string_carry ? ~uint64_t(0) : uint64_t(0);
    uint64_t prev_is_escaped = previous_escape_carry ? 1 : 0;

    size_t full_vectors = prefix_size / VECTOR_BYTES;
    size_t remainder = prefix_size % VECTOR_BYTES;

    for (size_t i = 0; i < full_vectors; i++) {
      const auto *addr = reinterpret_cast<const __m512i *>(&chunk[i * VECTOR_BYTES]);
      const __m512i data = _mm512_loadu_si512(addr);

      const uint64_t quotes = _mm512_cmpeq_epu8_mask(data, quote_mask);
      const uint64_t backslash = _mm512_cmpeq_epu8_mask(data, slash_mask);

      uint64_t potential_escape = backslash & ~prev_is_escaped;
      uint64_t maybe_escaped = potential_escape << 1;
      uint64_t maybe_escaped_and_odd_bits = maybe_escaped | ODD_BITS;
      uint64_t even_series_codes_and_odd_bits = maybe_escaped_and_odd_bits - potential_escape;

      uint64_t escape_and_terminal_code = even_series_codes_and_odd_bits ^ ODD_BITS;
      uint64_t escaped = escape_and_terminal_code ^ (backslash | prev_is_escaped);
      uint64_t escape = escape_and_terminal_code & backslash;
      prev_is_escaped = escape >> 63;

      uint64_t non_escaped_quotes = quotes & ~escaped;
      uint64_t string_idx = prefix_xor(non_escaped_quotes);
      string_idx ^= prev_in_string;
      prev_in_string = static_cast<int64_t>(string_idx) >> 63;

      uint64_t braces = _mm512_cmpeq_epu8_mask(data, brace_open_mask) |
                        _mm512_cmpeq_epu8_mask(data, brace_close_mask);
      uint64_t brackets = _mm512_cmpeq_epu8_mask(data, bracket_open_mask) |
                          _mm512_cmpeq_epu8_mask(data, bracket_close_mask);
      uint64_t colons_and_commas = _mm512_cmpeq_epu8_mask(data, colon_mask) |
                                   _mm512_cmpeq_epu8_mask(data, comma_mask);
      uint64_t struct_idx = braces | brackets | colons_and_commas;
      uint64_t nonquoted = struct_idx & ~string_idx;

      if (nonquoted == 0) {
        continue;
      }

      const auto count = count_ones(nonquoted);
      write_structural_index(tail_pos, nonquoted, i * VECTOR_BYTES + chunk_idx, count);
      index->block.structural_characters_count += count;
      tail_pos += count;
    }

    // Process remaining bytes (partial last vector) one byte at a time
    if (remainder > 0) {
      bool in_string = prev_in_string != 0;
      for (size_t i = 0; i < remainder; i++) {
        size_t pos = full_vectors * VECTOR_BYTES + i;
        char c = chunk[pos];

        if (in_string) {
          if (prev_is_escaped) {
            prev_is_escaped = 0;
          } else if (c == '\\') {
            prev_is_escaped = 1;
          } else if (c == '"') {
            in_string = false;
          }
          continue;
        }

        prev_is_escaped = 0;
        if (c == '"') {
          in_string = true;
        } else if (is_structural_char(c)) {
          index->block.structural_characters[tail_pos - index->block.structural_characters.data()] =
            static_cast<uint32_t>(chunk_idx + pos);
          index->block.structural_characters_count++;
          tail_pos++;
        }
      }
      prev_in_string = in_string ? ~uint64_t(0) : uint64_t(0);
    }

    // Set escape carry at end of prefix for cross-chunk propagation
    // ends_with_escape() reads escape_carry_index[CHUNK_CARRY_INDEX_SIZE - 1]
    index->escape_carry_index[0] = previous_escape_carry;
    index->escape_carry_index[CHUNK_CARRY_INDEX_SIZE - 1] = prefix_end_carry;

    tracer.finish_trace(trace_prefix);

    previous_escape_carry = prefix_end_carry;
    previous_string_carry = prev_in_string != 0;

    callback();
    return;
  }

  // Determine chunk data source: zero-copy aligned region or tail buffer
  size_t aligned_offset = chunk_idx - prefix_size;
  const char *chunk;
  bool is_tail = false;

  if (has_tail && aligned_offset + Engine::CHUNK_SIZE > aligned_length) {
    is_tail = true;
    chunk = reinterpret_cast<const char *>(tail_map);
  } else {
    chunk = aligned_start + aligned_offset;
  }

  // If we had a previous run, wait for it and process its output
  if (previous_run.has_value()) {
    prepare_kernel_input(chunk, *index, previous_escape_carry, !current);

    previous_run->handle.wait();

    string_buffers[current].output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    structural_buffers[current].output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    tracer.finish_trace(trace);

    current = !current;
  } else {
    prepare_kernel_input(chunk, *index, previous_escape_carry, current);
  }

  trace = tracer.start_trace("construct_combined_index_npu");

  string_buffers[current].input.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Look up the sub-buffer for this chunk
  size_t sub_buffer_idx = aligned_offset / Engine::CHUNK_SIZE;
  xrt::bo &sub_input = is_tail ? tail_input : json_chunk_inputs[sub_buffer_idx];

  auto run = kernel(3, instr, instr_size, sub_input, string_buffers[current].input,
                    string_buffers[current].output, structural_buffers[current].output);

  if (previous_run.has_value()) {
    read_kernel_output(
      *previous_run->index,
      previous_string_carry,
      previous_run->chunk_idx
    );

    previous_string_carry = previous_run->index->ends_in_string();

    previous_run->callback();

    previous_run.reset();
  }

  previous_escape_carry = index->ends_with_escape();

  previous_run = std::optional<RunHandle>({run, index, chunk_idx, callback});
}

void Kernel::wait_for_previous() {
  if (!previous_run.has_value()) {
    throw std::logic_error("Called wait for previous without previous run");
  }

  auto &tracer = util::Tracer::get_instance();

  previous_run->handle.wait();

  string_buffers[current].output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  structural_buffers[current].output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  tracer.finish_trace(trace);

  current = !current;

  read_kernel_output(
    *previous_run->index,
    previous_string_carry,
    previous_run->chunk_idx
  );

  previous_string_carry = previous_run->index->ends_in_string();

  previous_run->callback();

  previous_run.reset();
}

#endif

} // namespace npu
