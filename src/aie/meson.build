# Compile C++ kernels into object files
aie_compiler = find_program('clang++')
aie_tools_root = get_option('aie-tools-root')
kernel = custom_target(
  'kernel',
  input : 'json.cc',
  output : 'json.cc.o',
  command : [
    aie_compiler, '-O2', '-std=c++23', '--target=aie2-none-unknown-elf',
    '-Wno-parentheses', '-Wno-attributes', '-Wno-macro-redefined',
    '-Wno-deprecated-declarations', '-Wno-logical-not-parentheses', '-DNDEBUG',
    '-I', aie_tools_root + '/include', '-c', '@INPUT@', '-o', '@OUTPUT@'
  ]
)

# Compile Python NPU application description to MLIR
python = find_program('python')
mlir = custom_target(
  'mlir',
  input : ['gen_mlir_design.py', kernel],
  output : 'json_aie.mlir',
  command : [python, '@INPUT0@', '--kernel-obj', '@INPUT1@', '--out', '@OUTPUT@']
)

# Build insts file and Xclbin
aiecc = find_program('aiecc.py')
insts = custom_target(
  'json-insts.txt',
  input : mlir,
  output : 'json-insts.txt',
  install : true,
  install_dir : 'insts',
  command : [aiecc, '--aie-only-generate-npu', '--npu-insts-name=@OUTPUT@', '@INPUT@']
)

custom_target('json.xclbin',
  input : [kernel, mlir, insts],
  output : 'json.xclbin',
  install : true,
  install_dir : 'xclbins',
  command : [
    aiecc, '--aie-generate-cdo', '--aie-generate-npu',
    '--no-compile-host', '--no-xchesscc', '--no-xbridge',
    '--xclbin-name=@OUTPUT@', '--npu-insts-name=@INPUT2@', '@INPUT1@'
  ]
)
