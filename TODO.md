# TODO

- [x] Setup benchmark download script
- [x] List of benchmark documents + queries
- [x] Run the queries using `jq` or `rsonpath` as a test
- [x] Document why these queries
- [ ] Document sequence of kernels with inputs and outputs required
- [ ] Research/Document which indices the JSONPath automaton runs on and how
  - Critical to know before implementing kernels and automaton
  - Document architecture considerations w.r.t performance
- [x] Setup MLIR-AIE build process
- [x] Setup MLIR-AIE hello world (string indexer)
- [x] Setup benchmark for string indexer
- [ ] Setup test for string indexer (FileCheck or other i/o test)
  - Small JSON file and expected index specified in test
  - (Optional) Set up linters
    - `clangd-format` or/and `clangd-tidy` for C++
    - Something for Python
    - Shellcheck
  - (Optional) How to run tests on GitHub Actions (SSH or simulator?)
