#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>
#include <mutex>

namespace util {

using trace_id = size_t;
static constexpr trace_id INVALID_TRACE_ID = std::numeric_limits<trace_id>::max();

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

  void set_enabled(bool value);
  bool is_enabled() const;

  trace_id start_trace(std::string_view task);
  void finish_trace(trace_id id);

  void export_traces(const std::string &file_name);
private:
  Tracer() {}

  std::vector<Trace> traces;

  std::mutex tracer_mutex;
  std::atomic_bool enabled = false;
public:
  Tracer(Tracer const&)          = delete;
  void operator=(Tracer const&)  = delete;
};

} // namespace util
