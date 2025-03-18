#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

namespace test_utils {

std::vector<uint32_t> load_instr_sequence(std::string instr_path) {
  std::ifstream instr_file(instr_path);
  std::string line;
  std::vector<uint32_t> instr_v;
  while (std::getline(instr_file, line)) {
    std::istringstream iss(line);
    uint32_t a;
    if (!(iss >> std::hex >> a)) {
      throw std::runtime_error("Unable to parse instruction file");
    }
    instr_v.push_back(a);
  }
  return instr_v;
}

} // namespace test_utils
