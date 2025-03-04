#pragma once

#include <fstream>
#include <string>
#include <sstream>
#include <iostream>

namespace util {

std::string load_file_content(std::string filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Could not open file: " << filename << std::endl;
    return "";
  }

  std::stringstream buffer;
  buffer << file.rdbuf();

  file.close();
  return buffer.str();
}

std::string pad_to_multiple(std::string s, size_t k, char fill = ' ') {
  s.resize((s.size() + k - 1) / k * k, fill);
  return s;
}

} // namespace util
