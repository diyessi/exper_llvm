
set_property(DIRECTORY . APPEND PROPERTY COMPILE_OPTIONS "-fno-rtti")

add_executable(mlirfact
  ast.cpp
  context.cpp
  mlirfact.cpp
  my.cpp
  type_info.cpp
  )

add_dependencies(mlirfact MyIncGen)


set(LLVM_LINK_COMPONENTS
  Support
  )

include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)

target_link_libraries(mlirfact
PRIVATE
MLIRAnalysis
MLIRIR
MLIRDialect
MLIRParser
MLIRSideEffectInterfaces
MLIRTransforms)
