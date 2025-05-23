#include <string>
#include <format>
#include <iostream>

#include <npu-json/jsonpath/lexer.hpp>
#include <npu-json/error.hpp>

#include <npu-json/jsonpath/parser.hpp>

namespace jsonpath {

std::shared_ptr<jsonpath::Query> Parser::parse(const std::string & input) {
  auto query = std::make_shared<jsonpath::Query>();

  Lexer lexer(input);

  auto first_token = lexer.consume();
  if (first_token.type != TokenType::Root) {
    throw QueryError("Query should start with root ($)");
  }

  while (!lexer.is_at_end()) {
    auto segment = parse_segment(lexer);
    query->segments.emplace_back(segment);
  }

  return query;
}

std::string token_type_name(TokenType type) {
  switch (type) {
    case TokenType::Root: return "root";
    case TokenType::Member: return "member";
    case TokenType::Descendant: return "member";
    case TokenType::Wildcard: return "wildcard";
    case TokenType::OpenBracket: return "open bracket";
    case TokenType::CloseBracket: return "close bracket";
    case TokenType::Name: return "name";
    case TokenType::Number: return "number";
    default: throw std::logic_error("Unknown TokenType");
  }
}

Segment Parser::parse_segment(Lexer & lexer) {
  auto token = lexer.consume();
  std::cout << "parse_segment: " << token_type_name(token.type) << std::endl;
  switch (token.type) {
    case TokenType::Member: {
      auto next_token = lexer.peek();
      if (next_token.type == TokenType::OpenBracket) {
        lexer.consume();
        return parse_selector_segment(lexer);
      }
      return parse_member_segment(lexer);
    }
    case TokenType::Descendant: {
      return parse_descendant_segment(lexer);
    }
    case TokenType::OpenBracket: {
      auto segment = parse_selector_segment(lexer);
      auto next_token = lexer.consume();
      expect(next_token, TokenType::CloseBracket);
      return segment;
    }
    default: {
      throw QueryError("Unexpected token");
    }
  }
}

Segment Parser::parse_member_segment(Lexer & lexer) {
  auto token = lexer.consume();
  std::cout << "parse_member_segment: " << token_type_name(token.type) << std::endl;
  switch (token.type) {
    case TokenType::Name: {
      return segments::Member { token.text };
    }
    case TokenType::Wildcard: {
      return segments::Wildcard {};
    }
    default: {
      throw QueryError("Unexpected token");
    }
  }
}

Segment Parser::parse_descendant_segment(Lexer & lexer) {
  auto token = lexer.consume();
  std::cout << "parse_descendant_segment: " << token_type_name(token.type) << std::endl;
  switch (token.type) {
    case TokenType::Name: {
      return segments::Descendant { token.text };
    }
    default: {
      throw QueryError("Unexpected token");
    }
  }
}

Segment Parser::parse_selector_segment(Lexer & lexer) {
  auto token = lexer.consume();
  std::cout << "parse_selector_segment: " << token_type_name(token.type) << std::endl;
  switch (token.type) {
    case TokenType::Number: {
      return segments::Index { std::stol(token.text) };
    }
    case TokenType::Wildcard: {
      return segments::Wildcard {};
    }
    default: {
      throw make_unexpected_token_error(token);
    }
  }
}

void Parser::expect(Token & token, TokenType expected_type) {
  if (token.type != expected_type) {
    throw make_unexpected_token_error(token);
  }
}

QueryError Parser::make_unexpected_token_error(Token & token) {
  return QueryError(std::format("Unexpected {} token at {}", token_type_name(token.type), token.pos));
}

} // namespace jsonpath
