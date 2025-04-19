#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace util {

using trace_id = size_t;

struct Trace {
  std::string task;
  uint64_t start_ns;
  uint64_t duration_ns;

  Trace(std::string task, uint64_t start_ns)
    : task(task), start_ns(start_ns) {}
};

class Tracer {
public:
  static Tracer& get_instance() {
    static Tracer instance;
    return instance;
  }

  trace_id start_trace(std::string task);
  void finish_trace(trace_id id);

  std::string export_traces();
private:
  Tracer() {}
  // Tracer(Tracer const&);         // Don't Implement
  // void operator=(Tracer const&); // Don't implement

  std::vector<Trace> traces;
public:
  Tracer(Tracer const&)          = delete;
  void operator=(Tracer const&)  = delete;
};

} // namespace util
