#include <iostream>

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

ExitOnError &getExitOnErr() {
  static ExitOnError ExitOnErr;
  return ExitOnErr;
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();

  auto context = std::make_unique<LLVMContext>();
  auto module = std::make_unique<Module>("top", *context);
  llvm::IRBuilder<> builder(*context);

  std::vector<Type *> argtypes{Type::getInt32Ty(*context),
                               Type::getInt32Ty(*context)};
  FunctionType *FT = FunctionType::get(Type::getInt32Ty(*context), argtypes,
                                       /*not vararg*/ false);
  Function *F =
      Function::Create(FT, Function::ExternalLinkage, "fred", *module);
  BasicBlock *BB = BasicBlock::Create(*context, "EntryBlock", F);
  Value *Two = ConstantInt::get(Type::getInt32Ty(*context), 2);
  Value *Three = ConstantInt::get(Type::getInt32Ty(*context), 3);
  Instruction *Add = BinaryOperator::Create(Instruction::Add, Two, Three);
  BB->getInstList().push_back(Add);
  Instruction *Add1 =
      BinaryOperator::Create(Instruction::Add, Add, F->getArg(0));
  BB->getInstList().push_back(Add1);

  // Create the return instruction and add it to the basic block
  BB->getInstList().push_back(ReturnInst::Create(*context, Add1));

  llvm::verifyModule(*module, &llvm::outs());
  WriteBitcodeToFile(*module, llvm::outs());
  module->dump();

  auto TSM = orc::ThreadSafeModule(std::move(module), std::move(context));

  auto JIT = getExitOnErr()(orc::LLJITBuilder().create());
  getExitOnErr()(JIT->addIRModule(std::move(TSM)));

  auto fredSym = getExitOnErr()(JIT->lookup("fred"));
  int (*fred)(int) = (int (*)(int))fredSym.getAddress();

  outs() << "fred(42) = " << fred(42) << "\n";
}
