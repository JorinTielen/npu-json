#include <chrono>
#include <fstream>
#include <iostream>

#include <npu-json/util/tracer.hpp>

namespace util {

void Tracer::set_enabled(bool value) {
  enabled.store(value, std::memory_order_relaxed);
}

bool Tracer::is_enabled() const {
  return enabled.load(std::memory_order_relaxed);
}

trace_id util::Tracer::start_trace(std::string_view task) {
  if (!is_enabled()) {
    return INVALID_TRACE_ID;
  }

  std::lock_guard<std::mutex> guard(tracer_mutex);

  auto start = std::chrono::high_resolution_clock::now();
  auto epoch = start.time_since_epoch().count();
  auto start_ns = std::chrono::duration<uint64_t, std::nano>(epoch);
  traces.emplace_back(std::string(task), start_ns.count());
  return traces.size() - 1;
}

void Tracer::finish_trace(trace_id id) {
  if (id == INVALID_TRACE_ID) {
    return;
  }

  std::lock_guard<std::mutex> guard(tracer_mutex);

  auto& trace = traces[id];

  auto end = std::chrono::high_resolution_clock::now();
  auto epoch = end.time_since_epoch().count();
  auto end_ns = std::chrono::duration<uint64_t, std::nano>(epoch);
  trace.duration_ns = end_ns.count() - trace.start_ns;
}

void Tracer::export_traces(const std::string &file_name) {
  std::lock_guard<std::mutex> guard(tracer_mutex);

  std::ofstream output(file_name);

  output << "task,start_ns,duration_ns" << std::endl;

  if (traces.empty()) {
    return;
  }

  auto first_start_ns = traces[0].start_ns;

  for (auto trace : traces) {
    auto start_ns = trace.start_ns - first_start_ns;
    output << trace.task << "," << start_ns << "," << trace.duration_ns << std::endl;
  }
}

} // namespace util
