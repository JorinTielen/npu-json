#include <stdexcept>
#include <iostream>

#include <npu-json/result-set.hpp>

void ResultSet::record_result(size_t idx_start, size_t idx_end) {
  results.emplace_back(idx_start, idx_end);
}

size_t ResultSet::get_result_count() {
  return results.size();
}

std::string ResultSet::extract_result(size_t i, const std::string & json) {
  if (results.size() < i) throw std::out_of_range("Tried to extract result outside of valid set");

  auto [start, end] = results[i];
  return std::string(json, start, end - start + 1);
}
