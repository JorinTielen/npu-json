#pragma once

#include <cstddef>

// Simple class to record the results of a query.
// TODO: Actually record results instead of count
// TODO: Record different types of result (complex, value)
class ResultSet {
  size_t result_count = 0;
public:
  // Records a new result at position idx in the JSON.
  void record_result(size_t idx_start, size_t idx_end);

  // Returns the total number of results.
  size_t get_result_count();
};
