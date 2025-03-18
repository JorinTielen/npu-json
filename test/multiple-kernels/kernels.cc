#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

template <typename T, const size_t V, const int N>
void vector_scalar_add_aie(T *a, T *b, T w) {
  T *__restrict ptr_a = a;
  T *__restrict ptr_b = b;
  const int F = N / V;
  for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(16, ) {
      aie::vector<T, V> a = aie::load_v<V>(ptr_a);
      ptr_a += V;
      aie::vector<T, V> b = aie::add(a, w);
      aie::store_v(ptr_b, b);
      ptr_b += V;
    }
}

template <typename T, const size_t V, const int N>
void vector_scalar_mul_aie(T *a, T *b, T w) {
  T *__restrict ptr_a = a;
  T *__restrict ptr_b = b;
  const int F = N / V;
  for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(16, ) {
      aie::vector<T, V> a = aie::load_v<V>(ptr_a);
      ptr_a += V;
      aie::accum<acc64, V> b = aie::mul(a, w);
      aie::store_v(ptr_b, b.template to_vector<T>());
      ptr_b += V;
    }
}

extern "C" {

void vector_scalar_add(int32_t *a, int32_t *b) {
  vector_scalar_add_aie<int32_t, 64, 1024>(a, b, 1);
}

void vector_scalar_mul(int32_t *a, int32_t *b) {
  vector_scalar_mul_aie<int32_t, 64, 1024>(a, b, 3);
}

} // extern "C"
