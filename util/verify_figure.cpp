#include <algorithm>
#include <bitset>
#include <iostream>
#include <stdint.h>
#include <immintrin.h>

template <char C>
uint64_t match(const char * data) {
  const __m512i mask = _mm512_set1_epi8(C);
  __m512i vector = _mm512_loadu_si512(&data[0]);
  __mmask64 index = _mm512_cmpeq_epu8_mask(vector, mask);
  return index;
}

void print_bitmask(std::string name, uint64_t bitmask) {
  auto index_bitset = std::bitset<18>(bitmask);
  auto index_bitset_str = index_bitset.to_string();
  std::reverse(index_bitset_str.begin(), index_bitset_str.end());

  std::cout << name << ":\t|" << index_bitset_str << "|" << std::endl;
}

uint64_t prefix_xor(uint64_t bitmask) {
  bitmask ^= bitmask << 1;
  bitmask ^= bitmask << 2;
  bitmask ^= bitmask << 4;
  bitmask ^= bitmask << 8;
  bitmask ^= bitmask << 16;
  bitmask ^= bitmask << 32;
  return bitmask;
}

int main(int argc, char *argv[]) {
  const char data[64] = "{\"\\\\\\\"Quoted\\\"\":\"\\";
  std::cout << "data:\t|" << std::string(&data[0], 18) << "|" << std::endl;

  uint64_t p = 0;

  constexpr uint64_t ODD_BITS = 0xAAAAAAAAAAAAAAAAULL;
  print_bitmask("ob", ODD_BITS);

  uint64_t q = match<'"'>(data);
  print_bitmask("q", q);
  uint64_t b = match<'\\'>(data);
  print_bitmask("b", b);

  uint64_t pe = b & ~p;
  print_bitmask("pe", pe);

  uint64_t m = pe << 1;
  print_bitmask("m", m);

  uint64_t mo = m | ODD_BITS;
  print_bitmask("mo", mo);

  uint64_t s = mo - pe;
  print_bitmask("s", s);

  uint64_t t = s ^ ODD_BITS;
  print_bitmask("t", t);

  uint64_t e = t ^ (b | p);
  print_bitmask("e", e);

  uint64_t ec = t & b;
  print_bitmask("ec", ec);

  uint64_t uq = q & ~e;
  print_bitmask("uq", uq);

  uint64_t si = prefix_xor(uq);
  print_bitmask("si", si);
  return 0;
}
