#include <iostream>

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/FormattedStream.h"

int main() {
  llvm::LLVMContext context;
  llvm::Module module("top", context);
  llvm::IRBuilder<> builder(context);
  llvm::verifyModule(module, &llvm::outs());
  WriteBitcodeToFile(module, llvm::outs());
  module.dump();

  std::cout << "Hello, world!\n";
}
