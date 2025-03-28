
test_files = [
  'unit/escape_carry_index_test.cpp',
  'unit/structural_classifier_test.cpp',
]

test_exe = executable(
  'unit_tests',
  test_files,
  include_directories : '../src',
  dependencies : project_dependencies + [catch2_dep],
  link_with : npu_json_lib,
)

test('unit_test', test_exe)
