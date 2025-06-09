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

struct Range { int64_t start; int64_t end; };
} // namespace segments

using Segment = std::variant<
  segments::Member,
  segments::Descendant,
  segments::Wildcard,
  segments::Index,
  segments::Range
>;

struct Query {
  std::vector<Segment> segments = {};
};

} // namespace jsonpath
