#include <cstdint>

#include <catch2/catch_all.hpp>

#include <npu-json/jsonpath/parser.hpp>
#include <npu-json/jsonpath/query.hpp>

TEST_CASE("expects JSONPath query to start at root") {
  auto parser = jsonpath::Parser();

  CHECK_THROWS_AS(parser.parse(".hello.world"), QueryError);
}

TEST_CASE("parses JSONPath member expressions") {
  auto parser = jsonpath::Parser();

  auto query = parser.parse("$.hello.world");

  REQUIRE(query->segments.size() == 2);

  REQUIRE(std::holds_alternative<jsonpath::segments::Member>(query->segments[0]));
  REQUIRE(std::get<jsonpath::segments::Member>(query->segments[0]).name == "hello");

  REQUIRE(std::holds_alternative<jsonpath::segments::Member>(query->segments[1]));
  REQUIRE(std::get<jsonpath::segments::Member>(query->segments[1]).name == "world");
}

TEST_CASE("parses JSONPath descendant expressions") {
  auto parser = jsonpath::Parser();

  auto query = parser.parse("$..hello..world");

  REQUIRE(query->segments.size() == 2);

  REQUIRE(std::holds_alternative<jsonpath::segments::Descendant>(query->segments[0]));
  REQUIRE(std::get<jsonpath::segments::Descendant>(query->segments[0]).name == "hello");

  REQUIRE(std::holds_alternative<jsonpath::segments::Descendant>(query->segments[1]));
  REQUIRE(std::get<jsonpath::segments::Descendant>(query->segments[1]).name == "world");
}

TEST_CASE("parses JSONPath wildcard expressions") {
  auto parser = jsonpath::Parser();

  auto query = parser.parse("$[*]");

  REQUIRE(query->segments.size() == 1);

  REQUIRE(std::holds_alternative<jsonpath::segments::Wildcard>(query->segments[0]));

  query = parser.parse("$.[*]");

  REQUIRE(query->segments.size() == 1);

  REQUIRE(std::holds_alternative<jsonpath::segments::Wildcard>(query->segments[0]));

  query = parser.parse("$.*");

  REQUIRE(query->segments.size() == 1);

  REQUIRE(std::holds_alternative<jsonpath::segments::Wildcard>(query->segments[0]));
}

TEST_CASE("parses JSONPath index expressions") {
  auto parser = jsonpath::Parser();

  auto query = parser.parse("$.[123]");

  REQUIRE(query->segments.size() == 1);

  REQUIRE(std::holds_alternative<jsonpath::segments::Index>(query->segments[0]));
  REQUIRE(std::get<jsonpath::segments::Index>(query->segments[0]).value == 123);
}

TEST_CASE("parses twitter JSONPath query (T1)") {
  auto parser = jsonpath::Parser();
  auto query = parser.parse("$.statuses[*].user.lang");

  REQUIRE(query->segments.size() == 4);

  REQUIRE(std::holds_alternative<jsonpath::segments::Member>(query->segments[0]));
  REQUIRE(std::get<jsonpath::segments::Member>(query->segments[0]).name == "statuses");

  REQUIRE(std::holds_alternative<jsonpath::segments::Wildcard>(query->segments[1]));

  REQUIRE(std::holds_alternative<jsonpath::segments::Member>(query->segments[2]));
  REQUIRE(std::get<jsonpath::segments::Member>(query->segments[2]).name == "user");

  REQUIRE(std::holds_alternative<jsonpath::segments::Member>(query->segments[3]));
  REQUIRE(std::get<jsonpath::segments::Member>(query->segments[3]).name == "lang");
}

TEST_CASE("parses twitter JSONPath query (T3)") {
  auto parser = jsonpath::Parser();
  auto query = parser.parse("$[*].entities.urls[*].url");

  REQUIRE(query->segments.size() == 5);

  REQUIRE(std::holds_alternative<jsonpath::segments::Wildcard>(query->segments[0]));

  REQUIRE(std::holds_alternative<jsonpath::segments::Member>(query->segments[1]));
  REQUIRE(std::get<jsonpath::segments::Member>(query->segments[1]).name == "entities");

  REQUIRE(std::holds_alternative<jsonpath::segments::Member>(query->segments[2]));
  REQUIRE(std::get<jsonpath::segments::Member>(query->segments[2]).name == "urls");

  REQUIRE(std::holds_alternative<jsonpath::segments::Wildcard>(query->segments[3]));

  REQUIRE(std::holds_alternative<jsonpath::segments::Member>(query->segments[4]));
  REQUIRE(std::get<jsonpath::segments::Member>(query->segments[4]).name == "url");
}

TEST_CASE("parses bestbuy JSONPath query (B2)") {
  auto parser = jsonpath::Parser();
  auto query = parser.parse("$.products[*].videoChapters[*].chapter");

  REQUIRE(query->segments.size() == 5);

  REQUIRE(std::holds_alternative<jsonpath::segments::Member>(query->segments[0]));
  REQUIRE(std::get<jsonpath::segments::Member>(query->segments[0]).name == "products");

  REQUIRE(std::holds_alternative<jsonpath::segments::Wildcard>(query->segments[1]));

  REQUIRE(std::holds_alternative<jsonpath::segments::Member>(query->segments[2]));
  REQUIRE(std::get<jsonpath::segments::Member>(query->segments[2]).name == "videoChapters");

  REQUIRE(std::holds_alternative<jsonpath::segments::Wildcard>(query->segments[3]));

  REQUIRE(std::holds_alternative<jsonpath::segments::Member>(query->segments[4]));
  REQUIRE(std::get<jsonpath::segments::Member>(query->segments[4]).name == "chapter");
}

TEST_CASE("parses nspl JSONPath query (N1)") {
  auto parser = jsonpath::Parser();
  auto query = parser.parse("$.meta.view.columns[*].name");

  REQUIRE(query->segments.size() == 5);

  REQUIRE(std::holds_alternative<jsonpath::segments::Member>(query->segments[0]));
  REQUIRE(std::get<jsonpath::segments::Member>(query->segments[0]).name == "meta");

  REQUIRE(std::holds_alternative<jsonpath::segments::Member>(query->segments[1]));
  REQUIRE(std::get<jsonpath::segments::Member>(query->segments[1]).name == "view");

  REQUIRE(std::holds_alternative<jsonpath::segments::Member>(query->segments[2]));
  REQUIRE(std::get<jsonpath::segments::Member>(query->segments[2]).name == "columns");

  REQUIRE(std::holds_alternative<jsonpath::segments::Wildcard>(query->segments[3]));

  REQUIRE(std::holds_alternative<jsonpath::segments::Member>(query->segments[4]));
  REQUIRE(std::get<jsonpath::segments::Member>(query->segments[4]).name == "name");
}

TEST_CASE("parses walmart JSONPath query (W1)") {
  auto parser = jsonpath::Parser();
  auto query = parser.parse("$.items[*].bestMarketplacePrice.price");

  REQUIRE(query->segments.size() == 4);

  REQUIRE(std::holds_alternative<jsonpath::segments::Member>(query->segments[0]));
  REQUIRE(std::get<jsonpath::segments::Member>(query->segments[0]).name == "items");

  REQUIRE(std::holds_alternative<jsonpath::segments::Wildcard>(query->segments[1]));

  REQUIRE(std::holds_alternative<jsonpath::segments::Member>(query->segments[2]));
  REQUIRE(std::get<jsonpath::segments::Member>(query->segments[2]).name == "bestMarketplacePrice");

  REQUIRE(std::holds_alternative<jsonpath::segments::Member>(query->segments[3]));
  REQUIRE(std::get<jsonpath::segments::Member>(query->segments[3]).name == "price");
}

TEST_CASE("parses ast JSONPath query (A1)") {
  auto parser = jsonpath::Parser();
  auto query = parser.parse("$..decl.name");

  REQUIRE(query->segments.size() == 2);

  REQUIRE(std::holds_alternative<jsonpath::segments::Descendant>(query->segments[0]));
  REQUIRE(std::get<jsonpath::segments::Descendant>(query->segments[0]).name == "decl");

  REQUIRE(std::holds_alternative<jsonpath::segments::Member>(query->segments[1]));
  REQUIRE(std::get<jsonpath::segments::Member>(query->segments[1]).name == "name");
}

TEST_CASE("parses ast JSONPath query (A2)") {
  auto parser = jsonpath::Parser();
  auto query = parser.parse("$..inner..inner..type.qualType");

  REQUIRE(query->segments.size() == 4);

  REQUIRE(std::holds_alternative<jsonpath::segments::Descendant>(query->segments[0]));
  REQUIRE(std::get<jsonpath::segments::Descendant>(query->segments[0]).name == "inner");

  REQUIRE(std::holds_alternative<jsonpath::segments::Descendant>(query->segments[1]));
  REQUIRE(std::get<jsonpath::segments::Descendant>(query->segments[1]).name == "inner");

  REQUIRE(std::holds_alternative<jsonpath::segments::Descendant>(query->segments[2]));
  REQUIRE(std::get<jsonpath::segments::Descendant>(query->segments[2]).name == "type");

  REQUIRE(std::holds_alternative<jsonpath::segments::Member>(query->segments[3]));
  REQUIRE(std::get<jsonpath::segments::Member>(query->segments[3]).name == "qualType");
}
