#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

#ifndef __STRINGIFY
#define __STRINGIFY(a) #a
#endif
#define AIE_LOOP_MIN_ITERATION_COUNT(x)                                        \
  _Pragma(__STRINGIFY(clang loop min_iteration_count(x)))
#define AIE_LOOP_MAX_ITERATION_COUNT(x)                                        \
  _Pragma(__STRINGIFY(clang loop max_iteration_count(x)))
#define AIE_LOOP_RANGE(a, ...)                                                 \
  AIE_LOOP_MIN_ITERATION_COUNT(a)                                              \
  __VA_OPT__(AIE_LOOP_MAX_ITERATION_COUNT(__VA_ARGS__))
#define AIE_PREPARE_FOR_PIPELINING

__attribute__((inline)) uint64_t prefix_xor(uint64_t bitmask) {
    bitmask ^= bitmask << 1;
    bitmask ^= bitmask << 2;
    bitmask ^= bitmask << 4;
    bitmask ^= bitmask << 8;
    bitmask ^= bitmask << 16;
    bitmask ^= bitmask << 32;
    return bitmask;
}

// Combined index kernel: performs character classification (GEMM-equivalent
// via aie::eq / step activation), escape detection, prefix-XOR string
// detection, and structural combination in a single pass over raw JSON data.
//
// This fuses the two separate string_index and structural_character_index
// kernels into one, eliminating the need for CPU-side quote/backslash
// bitmask pre-processing.
//
// Input layout (in_buffer):
//   - carry_data[4]: uint32_t carry flags for string/escape state
//   - json_data[n]:  raw JSON bytes
//
// Output layout (out_buffer):
//   - string_index[n/8]:       uint64_t string bitmask per 64-byte lane
//   - structural_index[n/8]:   uint64_t non-quoted structural bitmask per lane
void combined_index_aie(
  uint8_t *__restrict in_buffer,
  uint64_t *__restrict string_out,
  uint64_t *__restrict structural_out,
  const int32_t n
) {
  static constexpr unsigned int V = 64;
  static constexpr const uint64_t ODD_BITS = 0xAAAAAAAAAAAAAAAAULL;

  uint32_t *carry_ptr = (uint32_t *)in_buffer;
  uint8_t *data_ptr = (uint8_t *)(in_buffer + 4);

  bool carry_in_string = (carry_ptr[0] & 1) != 0;
  bool carry_is_escaped = (carry_ptr[0] & 2) != 0;
  uint64_t prev_in_string = carry_in_string ? ~uint64_t(0) : uint64_t(0);
  uint64_t prev_is_escaped = carry_is_escaped ? 1 : 0;

  // Broadcast masks for character classification (GEMM weight columns)
  const aie::vector<uint8_t, V> quote_mask = aie::broadcast<uint8_t, V>('"');
  const aie::vector<uint8_t, V> backslash_mask = aie::broadcast<uint8_t, V>('\\');
  const aie::vector<uint8_t, V> brace_open_mask = aie::broadcast<uint8_t, V>('{');
  const aie::vector<uint8_t, V> brace_close_mask = aie::broadcast<uint8_t, V>('}');
  const aie::vector<uint8_t, V> bracket_open_mask = aie::broadcast<uint8_t, V>('[');
  const aie::vector<uint8_t, V> bracket_close_mask = aie::broadcast<uint8_t, V>(']');
  const aie::vector<uint8_t, V> colon_mask = aie::broadcast<uint8_t, V>(':');
  const aie::vector<uint8_t, V> comma_mask = aie::broadcast<uint8_t, V>(',');

  AIE_PREPARE_FOR_PIPELINING
  for (size_t i = 0; i < n; i += V) {
    const aie::vector<uint8_t, V> data = aie::load_v<V>(data_ptr);
    data_ptr += V;

    // Step 1: Character classification via high-level AIE vector API
    //         Equivalent to step(one_hot(x) @ W) where W is the
    //         256x8 character class weight matrix.
    auto quotes = (aie::eq(data, quote_mask)).to_uint64();
    auto backslash = (aie::eq(data, backslash_mask)).to_uint64();

    // Step 2: Escape sequence detection (element-wise bit operations)
    uint64_t potential_escape = backslash & ~prev_is_escaped;
    uint64_t maybe_escaped = potential_escape << 1;
    uint64_t maybe_escaped_and_odd_bits = maybe_escaped | ODD_BITS;
    uint64_t even_series_codes_and_odd_bits = maybe_escaped_and_odd_bits - potential_escape;

    uint64_t escape_and_terminal_code = even_series_codes_and_odd_bits ^ ODD_BITS;
    uint64_t escaped = escape_and_terminal_code ^ (backslash | prev_is_escaped);
    uint64_t escape = escape_and_terminal_code & backslash;
    prev_is_escaped = escape >> 63;

    // Step 3: String detection via prefix-XOR (inclusive scan with XOR)
    uint64_t non_escaped_quotes = quotes & ~escaped;
    uint64_t string_index = prefix_xor(non_escaped_quotes);
    string_index ^= prev_in_string;
    prev_in_string = static_cast<int64_t>(string_index) >> 63;

    // Step 4: Structural mask combination (OR reduction of character classes)
    auto brace_open = (aie::eq(data, brace_open_mask)).to_uint64();
    auto brace_close = (aie::eq(data, brace_close_mask)).to_uint64();
    auto braces = brace_open | brace_close;

    auto bracket_open = (aie::eq(data, bracket_open_mask)).to_uint64();
    auto bracket_close = (aie::eq(data, bracket_close_mask)).to_uint64();
    auto brackets = bracket_open | bracket_close;

    auto colon = (aie::eq(data, colon_mask)).to_uint64();
    auto comma = (aie::eq(data, comma_mask)).to_uint64();
    auto colons_and_commas = colon | comma;

    auto structurals = braces | brackets | colons_and_commas;

    // Step 5: Remove structural characters inside strings
    uint64_t nonquoted_structural = structurals & ~string_index;

    *string_out++ = string_index;
    *structural_out++ = nonquoted_structural;
  }
}

extern "C" {

void combined_index(uint8_t *in_buffer, uint64_t *string_out, uint64_t *structural_out, int32_t n) {
  combined_index_aie(in_buffer, string_out, structural_out, n);
}

}