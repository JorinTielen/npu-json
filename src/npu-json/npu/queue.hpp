#pragma once

#include <cassert>
#include <cstdint>
#include <thread>
#include <memory>
#include <mutex>

#include <npu-json/engine.hpp>

namespace npu {

// Thread-safe queue implementation.
// The queue owns the records inside of a pool of size N.
template <class T, std::size_t N>
class Queue {
  using RecordPool = std::array<T, N>;
public:
  Queue(const Queue&) = delete;
  Queue& operator=(const Queue&) = delete;

  static_assert(N >= 2);

  explicit Queue() : read_idx(0), write_idx(0) {
    record_pool = std::make_unique<RecordPool>();
  }

  // Reserve a space to write into. Waits for a free space if the queue is full.
  // It is not possible to reserve more than one space at a time. Multiple calls
  // return the same space.
  T* reserve_write_space() {
    std::unique_lock<std::mutex> guard(queue_mutex);

    auto pool = record_pool.get();

    auto next_write_idx = write_idx + 1;
    if (next_write_idx == N) next_write_idx = 0;

    // Wait until a space is free if the queue is full.
    queue_full_condition.wait(guard, [this, next_write_idx]{
      return next_write_idx != read_idx;
    });

    return &pool->data()[write_idx];
  }

  // Release the reserved space to signal writing is finished.
  // This transfers ownership of the space to the consumer.
  void release_write_space(T* space) {
    std::lock_guard<std::mutex> guard(queue_mutex);

    auto pool = record_pool.get();

    auto next_write_idx = write_idx + 1;
    if (next_write_idx == N) next_write_idx = 0;

    assert(space == &pool->data()[write_idx]);

    write_idx = next_write_idx;

    // Notify consumer thread waiting for non-empty queue.
    queue_empty_condition.notify_one();
  }

  // Claim the next token to read from. Waits if the queue is empty.
  // You can only claim a single token at once. Multiple calls return
  // the same token.
  T* claim_read_token() {
    std::unique_lock<std::mutex> guard(queue_mutex);

    auto pool = record_pool.get();

    // Wait until a token is produced if the queue is empty.
    queue_empty_condition.wait(guard, [this]{
      return read_idx != write_idx;
    });

    return &pool->data()[read_idx];
  }

  // Release the token to free the space.
  void release_token(T* token) {
    std::lock_guard<std::mutex> guard(queue_mutex);

    auto pool = record_pool.get();

    assert(token == &pool->data()[read_idx]);

    auto next_read_idx = read_idx + 1;
    if (next_read_idx == N) next_read_idx = 0;

    read_idx = next_read_idx;

    // Notify producer thread waiting for free space.
    queue_full_condition.notify_one();
  }

  bool is_empty() {
    std::lock_guard<std::mutex> guard(queue_mutex);

    return read_idx == write_idx;
  }

  void reset() {
    read_idx = 0;
    write_idx = 0;
  }
private:
  std::mutex queue_mutex;
  std::condition_variable queue_full_condition;
  std::condition_variable queue_empty_condition;

  std::unique_ptr<RecordPool> record_pool;

  std::size_t read_idx;
  std::size_t write_idx;
};

} // namespace npu
