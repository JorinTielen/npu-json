#include <cstring>
#include <stdexcept>

#include <npu-json/matrix/kernel.hpp>
#include <npu-json/util/debug.hpp>
#include <npu-json/util/tracer.hpp>

namespace matrix {

MatrixKernel::MatrixKernel(std::string_view json)
  : weight_matrix(build_weight_matrix()) {
  auto input_buffer_size_structural =
    (json.length() + Engine::CHUNK_SIZE - 1) / Engine::CHUNK_SIZE * Engine::CHUNK_SIZE;
  json_data.resize(input_buffer_size_structural, static_cast<uint8_t>(' '));

  if (!json.empty()) {
    memcpy(json_data.data(), json.begin(), json.length());
  }
}

void MatrixKernel::construct_combined_index(
  const char *chunk,
  npu::ChunkIndex &index,
  bool first_escape_carry,
  bool first_string_carry,
  size_t chunk_idx
) {
  auto &tracer = util::Tracer::get_instance();
  auto trace = tracer.start_trace("construct_combined_index_matrix");

  constexpr const size_t VECTOR_BYTES = 64;
  constexpr const size_t VECTORS_IN_CHUNK = npu::CHUNK_BIT_INDEX_SIZE / 8;
  static constexpr const uint64_t ODD_BITS = 0xAAAAAAAAAAAAAAAAULL;

  npu::construct_escape_carry_index(chunk, index, first_escape_carry);

  index.block.structural_characters_count = 0;
  auto tail = index.block.structural_characters.data();

  uint64_t prev_in_string = first_string_carry ? ~uint64_t(0) : uint64_t(0);
  uint64_t prev_is_escaped = first_escape_carry ? 1 : 0;

  uint64_t masks[CHAR_CLASS_COUNT];

  for (size_t i = 0; i < VECTORS_IN_CHUNK; i++) {
    const char *block = &chunk[i * VECTOR_BYTES];

    // Step 1: Character class detection via GEMM-equivalent lookup
    // This is equivalent to: M = one_hot(block) @ W, then step_activation(M)
    // where W is the 256x8 weight matrix and we pack each column into a bitmap.
    character_match(block, VECTOR_BYTES, weight_matrix, masks);

    // Step 2: Extract per-character masks from packed bitmasks
    uint64_t quotes = masks[static_cast<size_t>(CharClass::Quote)];
    uint64_t backslash = masks[static_cast<size_t>(CharClass::Backslash)];

    // Step 3: Escape sequence detection (element-wise bit operations)
    uint64_t potential_escape = backslash & ~prev_is_escaped;
    uint64_t maybe_escaped = potential_escape << 1;
    uint64_t maybe_escaped_and_odd_bits = maybe_escaped | ODD_BITS;
    uint64_t even_series_codes_and_odd_bits = maybe_escaped_and_odd_bits - potential_escape;

    uint64_t escape_and_terminal_code = even_series_codes_and_odd_bits ^ ODD_BITS;
    uint64_t escaped = escape_and_terminal_code ^ (backslash | prev_is_escaped);
    uint64_t escape = escape_and_terminal_code & backslash;
    prev_is_escaped = escape >> 63;

    // Step 4: String detection via prefix-XOR (inclusive scan with XOR)
    uint64_t non_escaped_quotes = quotes & ~escaped;
    uint64_t string_index = prefix_xor(non_escaped_quotes);
    string_index ^= prev_in_string;
    prev_in_string = static_cast<int64_t>(string_index) >> 63;
    index.string_index[i] = string_index;

    // Step 5: Structural mask combination (element-wise OR/AND operations)
    uint64_t braces = masks[static_cast<size_t>(CharClass::BraceOpen)] |
                      masks[static_cast<size_t>(CharClass::BraceClose)];
    uint64_t brackets = masks[static_cast<size_t>(CharClass::BracketOpen)] |
                        masks[static_cast<size_t>(CharClass::BracketClose)];
    uint64_t colons_and_commas = masks[static_cast<size_t>(CharClass::Colon)] |
                                  masks[static_cast<size_t>(CharClass::Comma)];
    uint64_t structural_index = braces | brackets | colons_and_commas;
    uint64_t nonquoted_structural = structural_index & ~string_index;

    // Step 6: Compress bitmask to position array (prefix-sum with ADD equivalent)
    if (nonquoted_structural == 0) {
      continue;
    }

    const auto count = count_ones(nonquoted_structural);
    compress_bitmask(tail, nonquoted_structural, i * VECTOR_BYTES + chunk_idx);
    index.block.structural_characters_count += count;
    tail += count;
  }

  tracer.finish_trace(trace);
}

void MatrixKernel::call(npu::ChunkIndex *index, size_t chunk_idx, std::function<void()> callback) {
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

void MatrixKernel::wait_for_previous() {}

}