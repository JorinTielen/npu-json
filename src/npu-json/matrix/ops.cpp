#include <cassert>
#include <cstring>

#include <npu-json/matrix/ops.hpp>

namespace matrix {

Matrix build_weight_matrix() {
  Matrix W(256, CHAR_CLASS_COUNT, 0.0f);

  W(static_cast<size_t>('{'),  static_cast<size_t>(CharClass::BraceOpen))    = 1.0f;
  W(static_cast<size_t>('}'),  static_cast<size_t>(CharClass::BraceClose))   = 1.0f;
  W(static_cast<size_t>('['),  static_cast<size_t>(CharClass::BracketOpen))  = 1.0f;
  W(static_cast<size_t>(']'),  static_cast<size_t>(CharClass::BracketClose)) = 1.0f;
  W(static_cast<size_t>(':'),  static_cast<size_t>(CharClass::Colon))        = 1.0f;
  W(static_cast<size_t>(','),  static_cast<size_t>(CharClass::Comma))        = 1.0f;
  W(static_cast<size_t>('"'),  static_cast<size_t>(CharClass::Quote))         = 1.0f;
  W(static_cast<size_t>('\\'), static_cast<size_t>(CharClass::Backslash))    = 1.0f;

  return W;
}

void gemm(const Matrix& A, const Matrix& B, Matrix& C, float alpha, float beta) {
  assert(A.cols() == B.rows());
  assert(C.rows() == A.rows());
  assert(C.cols() == B.cols());

  const size_t M = A.rows();
  const size_t K = A.cols();
  const size_t N = B.cols();

  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      float sum = 0.0f;
      for (size_t k = 0; k < K; k++) {
        sum += A(i, k) * B(k, j);
      }
      C(i, j) = alpha * sum + beta * C(i, j);
    }
  }
}

void step_activation(Matrix& M, float threshold) {
  for (size_t i = 0; i < M.rows(); i++) {
    for (size_t j = 0; j < M.cols(); j++) {
      M(i, j) = M(i, j) > threshold ? 1.0f : 0.0f;
    }
  }
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

void prefix_xor_inplace(std::vector<uint64_t>& vec) {
  for (size_t i = 1; i < vec.size(); i++) {
    vec[i] ^= vec[i - 1];
  }
}

uint64_t pack_column_to_bitmask(const Matrix& M, size_t col, size_t row_offset, size_t num_rows) {
  const size_t rows = (num_rows == 0) ? M.rows() : num_rows;
  assert(rows <= 64);
  assert(col < M.cols());
  assert(row_offset + rows <= M.rows());

  uint64_t bitmask = 0;
  for (size_t i = 0; i < rows; i++) {
    if (M(row_offset + i, col) > 0.5f) {
      bitmask |= (1ULL << i);
    }
  }
  return bitmask;
}

void character_match(const char* block, size_t block_size, const Matrix& W, uint64_t* masks) {
  assert(block_size <= VECTOR_BYTES);

  for (size_t k = 0; k < CHAR_CLASS_COUNT; k++) {
    masks[k] = 0;
  }

  for (size_t i = 0; i < block_size; i++) {
    uint8_t byte = static_cast<uint8_t>(block[i]);

    for (size_t k = 0; k < CHAR_CLASS_COUNT; k++) {
      if (W(byte, k) > 0.5f) {
        masks[k] |= (1ULL << i);
      }
    }
  }
}

uint64_t count_ones(uint64_t mask) {
  return static_cast<uint64_t>(__builtin_popcountll(mask));
}

size_t compress_bitmask(uint32_t* tail, uint64_t bitmask, size_t position) {
  if (bitmask == 0) return 0;

  size_t count = 0;
  while (bitmask != 0) {
    auto bit = __builtin_ctzll(bitmask);
    tail[count] = static_cast<uint32_t>(position + bit);
    count++;
    bitmask &= bitmask - 1;
  }

  return count;
}

}