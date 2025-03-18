#include <bitset>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <iostream>

#include <npu-json/jsonpath/query.hpp>
#include <npu-json/util/files.hpp>
#include <npu-json/engine.hpp>

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
  engine.run_query_on(*query, data);

  delete query;
  return 0;
}
