project('xrt')

cpp = meson.get_compiler('cpp')

xrt_dep = declare_dependency(
  dependencies : [
    cpp.find_library('xrt_coreutil', dirs : ['/opt/xilinx/xrt/lib']),
    dependency('uuid')
  ],
  include_directories : '/opt/xilinx/xrt/include'
)
