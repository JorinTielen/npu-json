#pragma once

#include <optional>
#include <string>

namespace jsonpath {

enum class TokenType {
  Root,
  Member,
  Descendant,
  Name,
  Number,
  OpenBracket,
  CloseBracket,
  Wildcard,
};

struct Token {
  TokenType type;
  size_t pos;
  std::string text;
};

class Lexer {
public:
  Lexer(const std::string & query)
    : input(query) {}

  bool is_at_end();
  Token consume();
  Token peek();
private:
  size_t pos = 0;
  const std::string & input;
  std::optional<Token> peeked_token;

  void advance();
  Token next_token();

  Token single_character_token(TokenType type);
  Token two_character_token(TokenType type);

  Token build_token(TokenType type, size_t start, size_t end);
};

} // namespace jsonpath
