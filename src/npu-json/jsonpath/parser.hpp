#pragma once

#include <memory>
#include <string>

#include <npu-json/jsonpath/lexer.hpp>
#include <npu-json/jsonpath/query.hpp>
#include <npu-json/error.hpp>

namespace jsonpath {

class Parser {
public:
  std::shared_ptr<Query> parse(const std::string & query);
private:
  Segment parse_segment(Lexer & lexer);
  Segment parse_member_segment(Lexer & lexer);
  Segment parse_descendant_segment(Lexer & lexer);
  Segment parse_selector_segment(Lexer & lexer);

  void expect(Token & token, TokenType expected_type);

  QueryError make_unexpected_token_error(Token & token);
};

} // namespace jsonpath
