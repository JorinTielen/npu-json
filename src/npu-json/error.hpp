#pragma once

#include <stdexcept>

struct QueryError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct EngineError : std::runtime_error {
  using std::runtime_error::runtime_error;
};
