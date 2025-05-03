#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

__attribute__((inline)) uint64_t prefix_xor(uint64_t bitmask) {
    bitmask ^= bitmask << 1;
    bitmask ^= bitmask << 2;
    bitmask ^= bitmask << 4;
    bitmask ^= bitmask << 8;
    bitmask ^= bitmask << 16;
    bitmask ^= bitmask << 32;
    return bitmask;
}

__attribute__((inline)) uint64_t trailing_zeroes(uint64_t n) {
  // return aie::(bitmask);
  int zeros = 0;
  if((n % 100000000) == 0)
  {
      zeros += 8;
      n /= 100000000;
  }
  if((n % 10000) == 0)
  {
      zeros += 4;
      n /= 10000;
  }
  if((n % 100) == 0)
  {
      zeros += 2;
      n /= 100;
  }
  if((n % 10) == 0)
  {
      zeros++;
  }
  return zeros;
  // return __builtin_ctzll(bitmask);
}

// TODO: Refactor to use structs and std::array for input/output buffers instead of byte pointers.

void string_index_aie(uint8_t *__restrict in_buffer, uint64_t *__restrict index_buffer, const int32_t n) {
  uint32_t *__restrict carry_ptr = (uint32_t *)(in_buffer + n);
  v64uint8 *__restrict in_ptr = (v64uint8 *)in_buffer;

  static constexpr unsigned int V = 64;

  static constexpr const uint64_t ODD_BITS = 0xAAAAAAAAAAAAAAAAULL;

  const v64uint8 quote_mask = broadcast_to_v64uint8('"');
  const v64uint8 backslash_mask = broadcast_to_v64uint8('\\');

  uint64_t prev_in_string = 0;
  uint64_t prev_is_escaped = uint64_t(*carry_ptr);

  for (unsigned int i = 0; i < n; i += V)
      chess_prepare_for_pipelining chess_loop_range(16,) {
    v64uint8 data = *in_ptr++;

    // Scan for quote and escape characters in input
    uint64_t quotes = eq(data, quote_mask);
    uint64_t backslash = eq(data, backslash_mask);

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
    uint64_t string_index = prefix_xor(non_escaped_quotes);

    // Invert if we were in a string previously
    string_index = string_index ^ prev_in_string;
    prev_in_string = static_cast<int64_t>(string_index) >> 63;

    // Store result
    *index_buffer++ = string_index;
  }
}

void structural_character_index_aie(
  uint8_t *__restrict in_buffer,
  uint64_t *__restrict index_buffer,
  const int32_t n
) {
  static constexpr unsigned int V = 64;

  // We pass the string index as second parameter through the same buffer as
  // the data to avoid NPU limitations (mem-tile channels).
  uint64_t *__restrict string_index_ptr = (uint64_t *)(in_buffer + n);
  // v64uint8 *__restrict data_ptr = (v64uint8 *)in_buffer;
  uint8_t *__restrict data_ptr = (uint8_t *)in_buffer;

  const aie::vector<uint8_t, V> brace_open_mask = aie::broadcast<uint8_t, V>('{');
  const aie::vector<uint8_t, V> brace_close_mask = aie::broadcast<uint8_t, V>('}');

  const aie::vector<uint8_t, V> bracket_open_mask = aie::broadcast<uint8_t, V>('[');
  const aie::vector<uint8_t, V> bracket_close_mask = aie::broadcast<uint8_t, V>(']');

  const aie::vector<uint8_t, V> colon_mask = aie::broadcast<uint8_t, V>(':');
  const aie::vector<uint8_t, V> comma_mask = aie::broadcast<uint8_t, V>(',');

  for (size_t i = 0; i < n; i += V) {
    const aie::vector<uint8_t, V> data = aie::load_v<V>(data_ptr);
    data_ptr += V;

    const uint64_t string_index = *string_index_ptr++;

    auto brace_open = aie::eq(data, brace_open_mask).to_uint64();
    auto brace_close = aie::eq(data, brace_close_mask).to_uint64();
    uint64_t braces = brace_open | brace_close;

    auto bracket_open = aie::eq(data, bracket_open_mask).to_uint64();
    auto bracket_close = aie::eq(data, bracket_close_mask).to_uint64();
    uint64_t brackets = bracket_open | bracket_close;

    auto colon = aie::eq(data, colon_mask).to_uint64();
    auto comma = aie::eq(data, comma_mask).to_uint64();
    uint64_t colons_and_commas = colon | comma;

    uint64_t structurals = braces | brackets | colons_and_commas;
    uint64_t nonquoted_structurals = structurals & ~string_index;

    *index_buffer++ = nonquoted_structurals;
    // size_t j = 0;
    // while (nonquoted_structurals) {
    //   // auto structural_idx = (i * V) + trailing_zeroes(nonquoted_structurals);
    //   auto structural_idx = i + j;
    //   *index_buffer++ = structural_idx;
    //   j++;

    //   // Remove structural we just processed from the mask
    //   nonquoted_structurals = nonquoted_structurals & (nonquoted_structurals - 1);
    // }
  }
}

extern "C" {

void string_index(uint8_t *in_buffer, uint64_t *index_buffer, int32_t n) {
  string_index_aie(in_buffer, index_buffer, n);
}

void structural_character_index(uint8_t *in_buffer, uint64_t *index_buffer, int32_t n) {
  structural_character_index_aie(in_buffer, index_buffer, n);
}

}
