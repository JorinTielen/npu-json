# TODO

- [ ] Setup benchmark download script
- [ ] List of benchmark documents + queries
- [ ] Run the queries using `jq` or `rsonpath` as a test
- [ ] Document why these queries
- [ ] Document sequence of kernels with inputs and outputs required
- [ ] Research/Document which indices the JSONPath automaton runs on and how
  - Critical to know before implementing kernels and automaton
  - Document architecture considerations w.r.t performance
- [ ] Setup MLIR-AIE build process
- [ ] Setup MLIR-AIE hello world (string indexer)
- [ ] Setup benchmark for string indexer
- [ ] Setup test for string indexer (FileCheck or other i/o test)
  - Small JSON file and expected index specified in test
  - (Optional) How to run these on GitHub Actions (SSH or simulator?)
