#pragma once

#include <vector>
#include <string>
#include <cstddef>

// Simple class to record the results of a query.
// TODO: Record different types of result (complex, value)
class ResultSet {
public:
  // Records a new result at position idx in the JSON.
  void record_result(size_t idx_start, size_t idx_end);

  // Returns the total number of results.
  size_t get_result_count();

  std::string extract_result(size_t i, const std::string & json);
private:
  std::vector<std::pair<std::size_t, std::size_t>> results;
};
