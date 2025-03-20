#include <array>
#include <immintrin.h>

#include <npu-json/structural/classifier.hpp>
#include "classifier.hpp"

// Structural Classifier based on the one in rsonpath, which is licensed MIT.

namespace structural {

constexpr const std::array<uint8_t, 32> LOWER_NIBBLE_MASK_ARRAY = {
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x03, 0x01, 0x02, 0x01, 0xff, 0xff, 0xff, 0xff, 0xff,
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x03, 0x01, 0x02, 0x01, 0xff, 0xff,
};
constexpr const std::array<uint8_t, 32> UPPER_NIBBLE_MASK_ARRAY = {
  0xfe, 0xfe, 0x10, 0x10, 0xfe, 0x01, 0xfe, 0x01, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0x10,
  0x10, 0xfe, 0x01, 0xfe, 0x01, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe,
};
constexpr const std::array<uint8_t, 32> COMMA_TOGGLE_MASK_ARRAY = {
  0x00, 0x00, 0x12, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x12,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};
constexpr const std::array<uint8_t, 32> COLON_TOGGLE_MASK_ARRAY = {
  0x00, 0x00, 0x00, 0x13, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x13, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};

inline const __m256i set_upper_nibble_zeroing_mask() {
  return _mm256_set1_epi8(0x0F);
}

inline const __m256i load_lower_nibble_mask() {
  return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(LOWER_NIBBLE_MASK_ARRAY.data()));
}

inline const __m256i load_upper_nibble_mask() {
  return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(UPPER_NIBBLE_MASK_ARRAY.data()));
}

inline const __m256i comma_toggle_mask() {
  return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(COMMA_TOGGLE_MASK_ARRAY.data()));
}

inline const __m256i colon_toggle_mask() {
  return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(COLON_TOGGLE_MASK_ARRAY.data()));
}

inline const __m256i colon_and_comma_toggle_mask() {
  return _mm256_or_si256(colon_toggle_mask(), comma_toggle_mask());
}

Classifier::Classifier() {
  upper_nibble_mask = load_upper_nibble_mask();
}

inline void Classifier::toggle_commas() {
  upper_nibble_mask = _mm256_xor_si256(upper_nibble_mask, comma_toggle_mask());
}

inline void Classifier::toggle_colons() {
  upper_nibble_mask = _mm256_xor_si256(upper_nibble_mask, colon_toggle_mask());
}

void Classifier::toggle_colons_and_commas() {
  upper_nibble_mask = _mm256_xor_si256(upper_nibble_mask, colon_and_comma_toggle_mask());
}

uint32_t Classifier::classify_block(const char *block) {
  auto byte_vector = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(block));
  auto shifted_byte_vector = _mm256_srli_epi16(byte_vector, 4);
  auto upper_nibble_byte_vector = _mm256_and_si256(shifted_byte_vector, set_upper_nibble_zeroing_mask());
  auto lower_nibble_lookup = _mm256_shuffle_epi8(load_lower_nibble_mask(), byte_vector);
  auto upper_nibble_lookup = _mm256_shuffle_epi8(upper_nibble_mask, upper_nibble_byte_vector);
  auto structural_vector = _mm256_cmpeq_epi8(lower_nibble_lookup, upper_nibble_lookup);
  uint32_t structural = _mm256_movemask_epi8(structural_vector);

  return structural;
}

} // namespace structural
