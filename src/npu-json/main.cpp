#include <bitset>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <iostream>
#include <iomanip>

#include <npu-json/jsonpath/parser.hpp>
#include <npu-json/jsonpath/query.hpp>
#include <npu-json/util/files.hpp>
#include <npu-json/util/tracer.hpp>
#include <npu-json/engine.hpp>
#include <npu-json/options.hpp>

using Clock = std::chrono::high_resolution_clock;
using Ms = std::chrono::duration<double, std::milli>;

void run_bench_warm(size_t data_size, Engine &engine) {
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
  double gigabytes = (double)data_size / 1000 / 1000 / 1000;
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
  bool cold_detail = false;
  bool trace = false;

  for (int i = 3; i < argc; i++) {
    std::string arg(argv[i]);
    if (arg == "--bench") {
      bench = true;
      if (i + 1 < argc) {
        std::string next(argv[i + 1]);
        if (next == "cold-detail") {
          cold = true;
          cold_detail = true;
          i++;
        } else if (next == "cold") {
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

    auto file_start = Clock::now();
    auto buf = util::load_file_hugepage(argv[1]);
    auto file_end = Clock::now();
    auto file_read_ms = Ms(file_end - file_start).count();

    double gigabytes = (double)buf.size / 1000.0 / 1000.0 / 1000.0;

    std::cout << "Data size: " << gigabytes << " GB" << std::endl;
    std::cout << "File read time: " << file_read_ms << " ms" << std::endl;

    if (cold_detail) {
      extern bool g_engine_cold_detail;
      g_engine_cold_detail = true;

      std::cout << "\n--- Detailed Cold Start Breakdown ---" << std::endl;
      std::cout << std::fixed << std::setprecision(3);

      auto t0 = Clock::now();

      auto parser = jsonpath::Parser();
      auto t1 = Clock::now();

      auto query = parser.parse(argv[2]);
      auto t2 = Clock::now();

      std::string_view json_sv(static_cast<const char*>(buf.data), buf.size);
      auto engine = Engine(*query, json_sv);
      auto t3 = Clock::now();

      engine.run_query();
      auto t4 = Clock::now();

      double d_parser_ctor  = Ms(t1 - t0).count();
      double d_query_parse  = Ms(t2 - t1).count();
      double d_engine_ctor  = Ms(t3 - t2).count();
      double d_first_query  = Ms(t4 - t3).count();
      double d_total         = Ms(t4 - t0).count();

      std::cout << "  1. Parser construct              : " << std::setw(8) << d_parser_ctor  << " ms" << std::endl;
      std::cout << "  2. Query parse                   : " << std::setw(8) << d_query_parse  << " ms" << std::endl;
      std::cout << "  3. Engine construct              : " << std::setw(8) << d_engine_ctor  << " ms" << std::endl;
      std::cout << "  4. First run_query()             : " << std::setw(8) << d_first_query  << " ms" << std::endl;
      std::cout << "  -------------------------------------------" << std::endl;
      std::cout << "     Total (excl. file read)       : " << std::setw(8) << d_total         << " ms" << std::endl;

      double total_ms = file_read_ms + d_total;
      double cold_seconds = d_total / 1000.0;
      double total_seconds = total_ms / 1000.0;

      std::cout << "\nCold start (excl. file read): " << d_total << " ms" << std::endl;
      std::cout << "Cold start (incl. file read): " << total_ms << " ms" << std::endl;
      std::cout << "Throughput (excl. file read): " << gigabytes / cold_seconds << " GB/s" << std::endl;
      std::cout << "Throughput (incl. file read): " << gigabytes / total_seconds << " GB/s" << std::endl;
    } else {
      auto cold_start = Clock::now();

      auto parser = jsonpath::Parser();
      auto query = parser.parse(argv[2]);
      std::string_view json_sv(static_cast<const char*>(buf.data), buf.size);
      auto engine = Engine(*query, json_sv);
      engine.run_query();

      auto cold_end = Clock::now();
      auto cold_ms = Ms(cold_end - cold_start).count();

      double total_ms = file_read_ms + cold_ms;
      double cold_seconds = cold_ms / 1000.0;
      double total_seconds = total_ms / 1000.0;

      std::cout << "Cold start (excl. file read): " << cold_ms << " ms" << std::endl;
      std::cout << "Cold start (incl. file read): " << total_ms << " ms" << std::endl;
      std::cout << "Throughput (excl. file read): " << gigabytes / cold_seconds << " GB/s" << std::endl;
      std::cout << "Throughput (incl. file read): " << gigabytes / total_seconds << " GB/s" << std::endl;
    }

    return 0;
  }

  // Read JSON file via huge-page-backed mmap
  auto buf = util::load_file_hugepage(argv[1]);
  size_t data_size = buf.size;

  // Parse query from string
  auto parser = jsonpath::Parser();
  auto query = parser.parse(argv[2]);

  std::string_view json_sv(static_cast<const char*>(buf.data), buf.size);
  auto engine = Engine(*query, json_sv);

  if (bench) {
    run_bench_warm(data_size, engine);
  } else {
    run_single(engine);
  }

  if (trace) {
    auto& tracer = util::Tracer::get_instance();
    tracer.export_traces("traces.csv");
  }

  return 0;
}
