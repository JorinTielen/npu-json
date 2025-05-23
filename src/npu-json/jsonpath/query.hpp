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
struct Member {
  std::string name;
};

struct Descendant {
  std::string name;
};

struct Wildcard {};
struct Index { int64_t value; };
} // namespace segments

using Segment = std::variant<
  segments::Member,
  segments::Descendant,
  segments::Wildcard,
  segments::Index
>;

struct Query {
  std::vector<Segment> segments = {};
};

} // namespace jsonpath
