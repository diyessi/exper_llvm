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

  auto Context = std::make_unique<LLVMContext>();
  auto TopModule = std::make_unique<Module>("top", *Context);

  FunctionType *FT = FunctionType::get(
      Type::getInt32Ty(*Context),
      {Type::getInt32Ty(*Context), Type::getInt32Ty(*Context)},
      /*not vararg*/ false);
  Function *F =
      Function::Create(FT, Function::ExternalLinkage, "fred", *TopModule);
  BasicBlock *BBEntry = BasicBlock::Create(*Context, "EntryBlock", F);
  BasicBlock *BBTrue = BasicBlock::Create(*Context, "TrueBlock", F);
  BasicBlock *BBFalse = BasicBlock::Create(*Context, "FalseBlock", F);

  auto Cmp = new ICmpInst(CmpInst::Predicate::ICMP_NE, F->getArg(1),
                          ConstantInt::get(Type::getInt32Ty(*Context), 0));
  BBEntry->getInstList().push_back(Cmp);
  BBEntry->getInstList().push_back(BranchInst::Create(BBTrue, BBFalse, Cmp));

  Value *Two = ConstantInt::get(Type::getInt32Ty(*Context), 2);
  Value *Three = ConstantInt::get(Type::getInt32Ty(*Context), 3);
  Instruction *Add = BinaryOperator::Create(Instruction::Add, Two, Three);
  BBTrue->getInstList().push_back(Add);
  Instruction *Mul1 =
      BinaryOperator::Create(Instruction::Mul, Add, F->getArg(0));
  BBTrue->getInstList().push_back(Mul1);

  // Create the return instruction and add it to the basic block
  BBTrue->getInstList().push_back(ReturnInst::Create(*Context, Mul1));

  BBFalse->getInstList().push_back(ReturnInst::Create(*Context, F->getArg(0)));

  llvm::verifyModule(*TopModule, &llvm::outs());
  WriteBitcodeToFile(*TopModule, llvm::outs());
  TopModule->dump();

  auto TSM = orc::ThreadSafeModule(std::move(TopModule), std::move(Context));
  auto JIT = getExitOnErr()(orc::LLJITBuilder().create());
  getExitOnErr()(JIT->addIRModule(std::move(TSM)));

  auto fredSym = getExitOnErr()(JIT->lookup("fred"));
  int (*fred)(int, int) = (int (*)(int, int))fredSym.getAddress();

  outs() << "fred(42, 0) = " << fred(42, 0) << "\n";
  outs() << "fred(42, 1) = " << fred(42, 1) << "\n";
}
