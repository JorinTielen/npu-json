#include <variant>
#include <stdexcept>

#include <npu-json/jsonpath/byte-code.hpp>
#include <npu-json/error.hpp>

namespace jsonpath {

void ByteCode::compile_from_query(Query &query) {
  instructions.clear();

  for (auto segment : query.segments) {
    std::visit([this](auto &&arg) -> void {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<std::monostate, T>) {
        throw std::logic_error("Invalid query segment type.");
      } else if constexpr (std::is_same_v<segments::Member, T>) {
        instructions.emplace_back(Opcode::OpenObject);
        instructions.emplace_back(Opcode::FindKey, arg.name);
      } else if constexpr (std::is_same_v<segments::Index, T>) {
        instructions.emplace_back(Opcode::OpenArray);
        instructions.emplace_back(Opcode::FindIndex, arg.value);
      } else if constexpr (std::is_same_v<segments::Range, T>) {
        instructions.emplace_back(Opcode::OpenArray);
        instructions.emplace_back(Opcode::FindRange, arg.start, arg.end);
      } else if constexpr (std::is_same_v<segments::Wildcard, T>) {
        instructions.emplace_back(Opcode::WildCard);
      } else {
        throw QueryError("Unsupported segment type in query");
      }
    }, segment);
  }

  // TODO: Check if trailing wildcards are allowed in JSONPath.
  // TODO: If so, remove trailing wildcard, as it is useless.

  instructions.emplace_back(Opcode::RecordResult);
  calculate_query_depth();
}

void ByteCode::calculate_query_depth() {
  auto depth = 0;
  for (auto instruction : instructions) {
    switch (instruction.opcode) {
    case Opcode::OpenArray:
    case Opcode::OpenObject:
    case Opcode::WildCard:
      depth++;
      break;
    default:
      break;
    }
    query_instruction_depth.emplace_back(depth);
  }
}

} // namespace jsonpath
