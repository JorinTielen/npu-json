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
  'json.mlir',
  input : ['gen_mlir_design.py', kernel],
  output : 'json.mlir',
  command : [python, '@INPUT0@', '--kernel-obj', '@INPUT1@', '--out', '@OUTPUT@']
)

# Build insts file and Xclbin
aiecc = find_program('aiecc.py')

custom_target('json.xclbin',
  input : [kernel, mlir],
  output : 'json.xclbin',
  install : true,
  install_dir : 'xclbins',
  command : [
    aiecc, '--aie-generate-npu', '--no-xchessc', '--no-xbridge', '--aie-generate-cdo',
    '--no-compile-host', '--xclbin-name=@OUTPUT@', '--npu-insts-name=insts.txt', '@INPUT1@'
  ]
)
