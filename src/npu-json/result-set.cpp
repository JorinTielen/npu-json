#include <iostream>

#include <npu-json/result-set.hpp>

void ResultSet::record_result(const std::string &json, size_t idx_start, size_t idx_end) {
  // std::cout << "record_result(" << idx_start << "," << idx_end << "): ";
  // std::cout << std::string(json, idx_start, (idx_end - idx_start) + 1) << std::endl;
  result_count++;
}

size_t ResultSet::get_result_count() {
  return result_count;
}
