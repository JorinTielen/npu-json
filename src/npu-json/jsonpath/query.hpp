#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace jsonpath {

enum class SegmentType {
  Name,
  Wildcard,
  Index,
  Slice,
  Filter
};

namespace segments {
struct Name {
  std::string member;
};

struct Wildcard {};
struct Index { size_t value; };
} // namespace segments

using Segment = std::variant<
  segments::Name,
  segments::Wildcard,
  segments::Index
>;

struct Query {
  std::vector<Segment> segments;
};

} // namespace jsonpath
