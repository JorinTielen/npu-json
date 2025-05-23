# Build & run unit tests (CPU side)
test_files = [
  'unit/escape_carry_index_test.cpp',
  'unit/jsonpath_parser_test.cpp',
  'unit/structural_classifier_test.cpp',
  'util/test-iterator.cpp',
]

test_exe = executable(
  'unit_tests',
  test_files,
  include_directories : '../src',
  dependencies : project_dependencies + [catch2_dep],
  link_with : npu_json_lib,
)

test('unit_test', test_exe)

# Build NPU end-to-end tests
executable(
  'npu_tests',
  'npu/string_index_test.cpp',
  include_directories: '../src',
  dependencies : project_dependencies,
  link_with : npu_json_lib
)
