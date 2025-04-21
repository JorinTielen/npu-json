#include <bitset>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <iostream>

#include <npu-json/jsonpath/query.hpp>
#include <npu-json/util/files.hpp>
#include <npu-json/util/tracer.hpp>
#include <npu-json/engine.hpp>
#include <npu-json/options.hpp>

void run_bench(std::string data, Engine &engine) {
  std::cout << "Starting benchmark..." << std::endl;

  constexpr size_t WARMUP_ITERS = 3;
  constexpr size_t BENCH_ITERS = 10;

  for (size_t i = 0; i < WARMUP_ITERS; i++) {
    engine.run_query_on(data);
  }

  std::cout << "Finished warmup..." << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < BENCH_ITERS; i++) {
    std::cout << "Iteration " << i << "..." << std::endl;
    engine.run_query_on(data);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto avg_runtime = (end - start) / BENCH_ITERS;

  auto seconds = std::chrono::duration<double>(avg_runtime).count();
  double gigabytes = (double)data.size() / 1000 / 1000 / 1000;
  std::cout << "Finished benchmark!" << std::endl;
  std::cout << "performed query on average in " << seconds << "s:" << std::endl;
  std::cout << "size: " << gigabytes << "GB" << std::endl;
  std::cout << "GB/s: " << gigabytes / seconds << std::endl;
}

void run_single(std::string data, Engine &engine) {
  auto results_set = engine.run_query_on(data);
  std::cout << "Found " << results_set->get_result_count() << " results!" << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: ./nj [json] [--bench]" << std::endl;
    return -1;
  }

  bool bench = false;
  if ((argc == 3 && std::string(argv[2]) == "--bench") ||
      (argc == 4 && std::string(argv[3]) == "--bench")) {
    bench = true;
  }

  bool trace = false;
  if ((argc == 3 && std::string(argv[2]) == "--trace") ||
      (argc == 4 && std::string(argv[3]) == "--trace")) {
    trace = true;
  }

  // Read in JSON file
  std::string data = util::load_file_content(argv[1]);

  // TODO: Parse query from string
  auto query = new jsonpath::Query {
    {
      // jsonpath::segments::Name { "statuses" },
      // jsonpath::segments::Wildcard {},
      // jsonpath::segments::Name { "user" },
      // jsonpath::segments::Name { "lang" }

      jsonpath::segments::Name { "products" },
      jsonpath::segments::Wildcard {},
      jsonpath::segments::Name { "videoChapters" },
      jsonpath::segments::Wildcard {},
      jsonpath::segments::Name { "chapter" }
    }
  };

  auto engine = Engine(*query);

  if (bench) {
    run_bench(data, engine);
  } else {
    run_single(data, engine);
  }

  if (trace) {
    auto& tracer = util::Tracer::get_instance();
    tracer.export_traces("traces.csv");
  }

  delete query;
  return 0;
}
