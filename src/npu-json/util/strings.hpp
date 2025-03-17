#pragma once

#include <string>

namespace util {

std::string pad_to_multiple(std::string s, size_t k, char fill = ' ') {
  s.resize((s.size() + k - 1) / k * k, fill);
  return s;
}

} // namespace util
