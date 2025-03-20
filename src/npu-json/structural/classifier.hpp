#pragma once

#include <cstdint>
#include <immintrin.h>

// Structural Classifier based on the one in rsonpath, which is licensed MIT.

namespace structural {

class Classifier {
  __m256i upper_nibble_mask;

  void toggle_commas();
  void toggle_colons();
public:
  Classifier();
  void toggle_colons_and_commas();
  uint32_t classify_block(const char *block);
};

} // namespace structural
