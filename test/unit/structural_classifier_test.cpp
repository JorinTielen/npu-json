#include <array>
#include <algorithm>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

#include <catch2/catch_all.hpp>

#include <npu-json/structural/classifier.hpp>
#include <npu-json/engine.hpp>


// Helper to write a JSON string into a vector block at the end for easy testing
void write_string_into_block_end(std::array<uint8_t, 32> &block, const std::string &str) {
  assert(str.size() <= block.size());

  auto start_idx = block.size() - str.size();
  for (size_t i = start_idx; i < block.size(); i++) {
    block[i] = str[i - start_idx];
  }
}

bool structural_block_is_equal_to_bits(uint32_t structural, const char *bits) {
  auto bits_str = std::string(bits);

  // Pad passed in bitstring with leading zeroes
  std::string bits_with_leading_zeros;
  assert(bits_str.size() <= 32);
  if (bits_str.size() != 32) {
    auto leading_count = 32 - bits_str.length();
    bits_with_leading_zeros = std::string(leading_count, '0').append(bits_str);
  } else {
    bits_with_leading_zeros = bits;
  }

  // Compare bitsets
  auto structural_bitset = std::bitset<32>(structural);
  std::reverse(bits_with_leading_zeros.begin(), bits_with_leading_zeros.end());
  auto bits_bitset = std::bitset<32>(bits_with_leading_zeros);
  return structural_bitset == bits_bitset;
}

TEST_CASE("recognizes all types of structural characters") {
  auto block = std::array<uint8_t, 32> {};
  std::fill(block.begin(), block.end(), ' ');

  auto json = std::string("{\"a\": 123, \"b\": []}");

  write_string_into_block_end(block, json);

  auto classifier = structural::Classifier();
  classifier.toggle_colons_and_commas();

  auto block_ptr = reinterpret_cast<const char *>(block.data());

  auto structurals = classifier.classify_block(block_ptr);

  REQUIRE(structural_block_is_equal_to_bits(structurals, "1000100001000010111"));
}

