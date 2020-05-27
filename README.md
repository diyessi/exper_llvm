# Experiments with llvm/mlir

To prepare llvm:
```bash
cmake -G Ninja ../llvm/ -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD="X86"
cmake --build .
cmake -DCMAKE_INSTALL_PREFIX=$HOME/llvm -P cmake_install.cmake
```
