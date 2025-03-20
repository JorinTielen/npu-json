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

void stringindexer_aie(uint8_t *__restrict in_buffer, uint64_t *__restrict index_buffer, const int32_t n) {
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

extern "C" {

void stringindexer(uint8_t *in_buffer, uint64_t *index_buffer, int32_t n) {
  stringindexer_aie(in_buffer, index_buffer, n);
}

}
