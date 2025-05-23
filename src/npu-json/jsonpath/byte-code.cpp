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
}

} // namespace jsonpath
