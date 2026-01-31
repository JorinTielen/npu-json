# Compile C++ kernels into object files
aie_compiler = find_program('clang++')
aie_tools_root = get_option('aie-tools-root')

# 1. Find Python
python = find_program('python')

# 2. Dynamically find the mlir_aie include path using __path__ for namespace packages
#    We cast __path__ to a list and take the first entry, then append 'include'
cmd = 'import os; import mlir_aie; print(os.path.join(list(mlir_aie.__path__)[0], "include"))'

r = run_command(python, '-c', cmd, check: false)

if r.returncode() != 0
  error('Could not find mlir_aie python package. Make sure the venv is active.')
endif

mlir_aie_inc = r.stdout().strip()
# Optional: Print it to debug if the build fails again
message('Found mlir-aie include path: ' + mlir_aie_inc)

kernel = custom_target(
  'kernel',
  input : 'json.cc',
  output : 'json.cc.o',
  command : [
    aie_compiler, '-O2', '-std=c++23', '--target=aie2p-none-unknown-elf',
    '-Wno-parentheses', '-Wno-attributes', '-Wno-macro-redefined',
    '-Wno-deprecated-declarations', '-Wno-logical-not-parentheses', '-DNDEBUG',
    '-DBIT_WIDTH=8',
    '-I', aie_tools_root + '/include',
    '-I', mlir_aie_inc,
    '-c', '@INPUT@', '-o', '@OUTPUT@'
  ]
)

# Compile Python NPU application description to MLIR
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
    aiecc, '--aie-generate-xclbin', '--aie-generate-npu-insts', '--no-compile-host', 
    '--no-xchessc', '--no-xbridge',
    '--xclbin-name=@OUTPUT@', '--npu-insts-name=insts.txt', '@INPUT1@'
  ]
)