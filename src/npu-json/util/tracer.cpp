#include <chrono>
#include <sstream>

#include <npu-json/util/tracer.hpp>

namespace util {

trace_id util::Tracer::start_trace(std::string task) {
  auto start = std::chrono::high_resolution_clock::now();
  auto epoch = start.time_since_epoch().count();
  auto start_ns = std::chrono::duration<uint64_t, std::nano>(epoch);
  traces.emplace_back(task, start_ns.count());
  return traces.size() - 1;
}

void Tracer::finish_trace(trace_id id) {
  auto& trace = traces[id];

  auto end = std::chrono::high_resolution_clock::now();
  auto epoch = end.time_since_epoch().count();
  auto end_ns = std::chrono::duration<uint64_t, std::nano>(epoch);
  trace.duration_ns = end_ns.count() - trace.start_ns;
}

std::string Tracer::export_traces() {
  std::stringstream output;

  if (traces.empty()) {
    return std::string();
  }

  output << "task,start,duration" << std::endl;

  auto first_start_ns = traces[0].start_ns;

  for (auto trace : traces) {
    auto start_ns = trace.start_ns - first_start_ns;
    output << trace.task << "," << start_ns << "," << trace.duration_ns << std::endl;
  }

  return output.str();
}

} // namespace util
