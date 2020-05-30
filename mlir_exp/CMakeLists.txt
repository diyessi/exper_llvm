add_subdirectory(include)

set_property(DIRECTORY . APPEND PROPERTY COMPILE_OPTIONS "-fno-rtti")

add_executable(mlirfact mlirfact.cpp my.cpp)

add_dependencies(mlirfact MyIncGen)


set(LLVM_LINK_COMPONENTS
  Support
  )

include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)

target_link_libraries(mlirfact
PRIVATE
MLIRAnalysis
MLIRIR
MLIRDialect
MLIRParser
MLIRSideEffectInterfaces
MLIRTransforms)
