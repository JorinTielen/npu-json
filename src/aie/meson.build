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
string_index_mlir = custom_target(
  'string_index.mlir',
  input : ['gen_mlir_design.py', kernel],
  output : 'string_index.mlir',
  command : [
    python, '@INPUT0@', '--kernel', 'string_index',
    '--kernel-obj', '@INPUT1@',
    '--out', '@OUTPUT@'
  ]
)

structural_character_index_mlir = custom_target(
  'structural_character_index.mlir',
  input : ['gen_mlir_design.py', kernel],
  output : 'structural_character_index.mlir',
  command : [
    python, '@INPUT0@', '--kernel', 'structural_character_index',
    '--kernel-obj', '@INPUT1@',
    '--out', '@OUTPUT@'
  ]
)

# Build insts file and Xclbin
aiecc = find_program('aiecc.py')

initial_xclbin = custom_target('string_index.xclbin',
  input : [kernel, string_index_mlir],
  output : 'string_index.xclbin',
  command : [
    aiecc, '--xclbin-kernel-name=STRINGINDEX', '--xclbin-kernel-id=0x901', '--xclbin-instance-name=STRINGINDEXINST',
    '--aie-generate-npu', '--no-xchessc', '--no-xbridge',
    '--aie-generate-cdo', '--no-compile-host', '--xclbin-name=@OUTPUT@', '--npu-insts-name=string-index-insts.txt', '@INPUT1@'
  ]
)

custom_target('multi_kernel.xclbin',
  input : [kernel, initial_xclbin, structural_character_index_mlir],
  output : 'json.xclbin',
  install : true,
  install_dir : 'xclbins',
  command : [
    aiecc, '--xclbin-kernel-name=STRUCTURALCHARACTERINDEX', '--xclbin-kernel-id=0x902', '--xclbin-instance-name=STRUCTURALCHARACTERINDEXINST',
    '--aie-generate-npu', '--no-xchessc', '--no-xbridge', '--xclbin-input=@INPUT1@',
    '--aie-generate-cdo', '--no-compile-host', '--xclbin-name=@OUTPUT@', '--npu-insts-name=structural-character-index-insts.txt', '@INPUT2@'
  ]
)
