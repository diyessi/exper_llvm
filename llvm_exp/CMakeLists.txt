add_executable(buildir main.cpp)

# Find the libraries that correspond to the LLVM components
# that we wish to use
llvm_map_components_to_libnames(llvm_libs bitwriter support core irreader orcjit nativecodegen)

# Link against LLVM libraries
target_link_libraries(buildir ${llvm_libs})
