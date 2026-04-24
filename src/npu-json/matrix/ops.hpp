#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <npu-json/matrix/matrix.hpp>

namespace matrix {

enum class CharClass : size_t {
  BraceOpen = 0,
  BraceClose = 1,
  BracketOpen = 2,
  BracketClose = 3,
  Colon = 4,
  Comma = 5,
  Quote = 6,
  Backslash = 7,
  CharClassCount = 8
};

constexpr size_t CHAR_CLASS_COUNT = static_cast<size_t>(CharClass::CharClassCount);

constexpr size_t VECTOR_BYTES = 64;

Matrix build_weight_matrix();

void gemm(const Matrix& A, const Matrix& B, Matrix& C, float alpha = 1.0f, float beta = 0.0f);

void step_activation(Matrix& M, float threshold = 0.5f);

uint64_t prefix_xor(uint64_t bitmask);

void prefix_xor_inplace(std::vector<uint64_t>& vec);

uint64_t pack_column_to_bitmask(const Matrix& M, size_t col, size_t row_offset = 0, size_t num_rows = 0);

void character_match(const char* block, size_t block_size, const Matrix& W, uint64_t* masks);

uint64_t count_ones(uint64_t mask);

size_t compress_bitmask(uint32_t* tail, uint64_t bitmask, size_t position);

}