#pragma once

#include <algorithm>
#include <cstdint>
#include <bitset>
#include <iostream>

void print_input_and_index(const char* input, const uint64_t *index, const size_t at = 0) {
  auto index_bitset = std::bitset<64>(index[at]);
  auto index_bitset_str = index_bitset.to_string();
  std::reverse(index_bitset_str.begin(), index_bitset_str.end());
  auto input_str = std::string(input + at * 64, 64);
  std::replace(input_str.begin(), input_str.end(), '\n', ' ');
  std::cout << "data (64 bytes at position " << at << "):" << std::endl;
  std::cout << "input: |" << input_str << "|" << std::endl;
  std::cout << "index: |" << index_bitset_str << "|" << std::endl;
}
