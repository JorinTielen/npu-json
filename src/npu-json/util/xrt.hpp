#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <utility>

#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

namespace util {

inline std::pair<xrt::device, xrt::kernel> init_npu(std::string xclbin_name) {
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);
  auto xclbin = xrt::xclbin(xclbin_name);
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, "MLIR_AIE");
  return std::make_pair(device, kernel);
}

inline std::vector<uint32_t> load_instr_sequence(std::string instr_path) {
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

} // namespace util
