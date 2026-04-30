#include <bitset>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <iostream>

#include <npu-json/jsonpath/parser.hpp>
#include <npu-json/jsonpath/query.hpp>
#include <npu-json/util/files.hpp>
#include <npu-json/util/tracer.hpp>
#include <npu-json/engine.hpp>
#include <npu-json/options.hpp>

void run_bench_warm(std::string &data, Engine &engine) {
  std::cout << "Starting benchmark..." << std::endl;

  constexpr size_t WARMUP_ITERS = 25;
  constexpr size_t BENCH_ITERS = 100;

  for (size_t i = 0; i < WARMUP_ITERS; i++) {
    engine.run_query();
  }

  auto start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < BENCH_ITERS; i++) {
    engine.run_query();
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

void run_single(Engine &engine) {
  auto results_set = engine.run_query();
  std::cout << "Found " << results_set->get_result_count() << " results!" << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << "Usage: ./nj json query [--bench [cold|warm]] [--trace]" << std::endl;
    return -1;
  }

  bool bench = false;
  bool cold = false;
  bool trace = false;

  for (int i = 3; i < argc; i++) {
    std::string arg(argv[i]);
    if (arg == "--bench") {
      bench = true;
      if (i + 1 < argc) {
        std::string next(argv[i + 1]);
        if (next == "cold") {
          cold = true;
          i++;
        } else if (next == "warm") {
          i++;
        }
      }
    } else if (arg == "--trace") {
      trace = true;
    }
  }

  if (cold) {
    std::cout << "=== Cold Benchmark ===" << std::endl;
    std::cout << "File: " << argv[1] << std::endl;
    std::cout << "Query: " << argv[2] << std::endl;

    auto file_start = std::chrono::high_resolution_clock::now();
    std::string data = util::load_file_content(argv[1]);
    auto file_end = std::chrono::high_resolution_clock::now();
    auto file_read_ms = std::chrono::duration<double, std::milli>(file_end - file_start).count();

    double gigabytes = (double)data.size() / 1000.0 / 1000.0 / 1000.0;

    std::cout << "Data size: " << gigabytes << " GB" << std::endl;
    std::cout << "File read time: " << file_read_ms << " ms" << std::endl;

    auto cold_start = std::chrono::high_resolution_clock::now();

    auto parser = jsonpath::Parser();
    auto query = parser.parse(argv[2]);
    auto engine = Engine(*query, data);
    engine.run_query();

    auto cold_end = std::chrono::high_resolution_clock::now();
    auto cold_ms = std::chrono::duration<double, std::milli>(cold_end - cold_start).count();

    double total_ms = file_read_ms + cold_ms;
    double cold_seconds = cold_ms / 1000.0;
    double total_seconds = total_ms / 1000.0;

    std::cout << "Cold start (excl. file read): " << cold_ms << " ms" << std::endl;
    std::cout << "Cold start (incl. file read): " << total_ms << " ms" << std::endl;
    std::cout << "Throughput (excl. file read): " << gigabytes / cold_seconds << " GB/s" << std::endl;
    std::cout << "Throughput (incl. file read): " << gigabytes / total_seconds << " GB/s" << std::endl;

    return 0;
  }

  // Read in JSON file
  std::string data = util::load_file_content(argv[1]);

  // Parse query from string
  auto parser = jsonpath::Parser();
  auto query = parser.parse(argv[2]);

  auto engine = Engine(*query, data);

  if (bench) {
    run_bench_warm(data, engine);
  } else {
    run_single(engine);
  }

  if (trace) {
    auto& tracer = util::Tracer::get_instance();
    tracer.export_traces("traces.csv");
  }

  return 0;
}
