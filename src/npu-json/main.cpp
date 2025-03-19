#include <bitset>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <iostream>

#include <npu-json/jsonpath/query.hpp>
#include <npu-json/util/files.hpp>
#include <npu-json/engine.hpp>
#include <npu-json/options.hpp>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: ./nj [json]" << std::endl;
    return -1;
  }

  // Read in JSON file
  std::string data = util::load_file_content(argv[1]);

  // TODO: Parse query from string
  auto query = new jsonpath::Query {
    {
      jsonpath::segments::Name { "user" },
      jsonpath::segments::Name { "lang" }
    }
  };

  auto engine = Engine();

  auto start = std::chrono::high_resolution_clock::now();

  engine.run_query_on(*query, data);

  auto end = std::chrono::high_resolution_clock::now();
  auto runtime = (end - start);

  auto seconds = std::chrono::duration<double>(runtime).count();
  double gigabytes = (double)data.size() / 1000 / 1000 / 1000;
  std::cout << "performed query in " << seconds << "s:" << std::endl;
  std::cout << "size: " << gigabytes << "GB" << std::endl;
  std::cout << "GB/s: " << gigabytes / seconds << std::endl;

  delete query;
  return 0;
}
