#include "result-set.hpp"

void ResultSet::record_result(size_t idx_start, size_t idx_end) {
  result_count++;
}

size_t ResultSet::get_result_count() {
  return result_count;
}
