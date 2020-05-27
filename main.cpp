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

Function *createFact(Module &M) {
  auto &C = M.getContext();
  Function *F = Function::Create(
      FunctionType::get(Type::getInt32Ty(C), {Type::getInt32Ty(C)}, false),
      Function::ExternalLinkage, "fact", M);
  auto BBEntry = BasicBlock::Create(C, "Entry", F);
  auto BBLoop = BasicBlock::Create(C, "Loop", F);
  auto BBLoop1 = BasicBlock::Create(C, "Loop1", F);
  auto BBExit = BasicBlock::Create(C, "Exit", F);

  BBEntry->getInstList().push_back(BranchInst::Create(BBLoop));

  auto LoopResult = PHINode::Create(Type::getInt32Ty(C), 2);
  LoopResult->addIncoming(ConstantInt::get(Type::getInt32Ty(C), 1), BBEntry);
  auto LoopCount = PHINode::Create(Type::getInt32Ty(C), 2);
  LoopCount->addIncoming(F->getArg(0), BBEntry);
  BBLoop->getInstList().push_back(LoopResult);
  BBLoop->getInstList().push_back(LoopCount);
  auto ICmp = new ICmpInst(ICmpInst::Predicate::ICMP_SGT, LoopCount,
                           ConstantInt::get(Type::getInt32Ty(C), 0));
  BBLoop->getInstList().push_back(ICmp);
  BBLoop->getInstList().push_back(BranchInst::Create(BBLoop1, BBExit, ICmp));

  auto LoopResult1 = PHINode::Create(Type::getInt32Ty(C), 1);
  LoopResult1->addIncoming(LoopResult, BBLoop);
  auto LoopCount1 = PHINode::Create(Type::getInt32Ty(C), 1);
  LoopCount1->addIncoming(LoopCount, BBLoop);
  BBLoop1->getInstList().push_back(LoopResult1);
  BBLoop1->getInstList().push_back(LoopCount1);
  auto NextResult =
      BinaryOperator::Create(Instruction::Mul, LoopResult1, LoopCount1);
  BBLoop1->getInstList().push_back(NextResult);
  auto NextCount = BinaryOperator::Create(
      Instruction::Sub, LoopCount1, ConstantInt::get(Type::getInt32Ty(C), 1));
  BBLoop1->getInstList().push_back(NextCount);
  BBLoop1->getInstList().push_back(BranchInst::Create(BBLoop));
  LoopResult->addIncoming(NextResult, BBLoop1);
  LoopCount->addIncoming(NextCount, BBLoop1);

  auto ExitResult = PHINode::Create(Type::getInt32Ty(C), 1);
  ExitResult->addIncoming(LoopResult, BBLoop);
  BBExit->getInstList().push_back(ExitResult);
  BBExit->getInstList().push_back(ReturnInst::Create(C, ExitResult));

  return F;
}

Function *createFred(Module &M) {
  auto &C = M.getContext();

  FunctionType *FT = FunctionType::get(
      Type::getInt32Ty(C), {Type::getInt32Ty(C), Type::getInt32Ty(C)},
      /*not vararg*/ false);
  Function *F = Function::Create(FT, Function::ExternalLinkage, "fred", M);
  BasicBlock *BBEntry = BasicBlock::Create(C, "EntryBlock", F);
  BasicBlock *BBTrue = BasicBlock::Create(C, "TrueBlock", F);
  BasicBlock *BBFalse = BasicBlock::Create(C, "FalseBlock", F);

  auto Cmp = new ICmpInst(CmpInst::Predicate::ICMP_NE, F->getArg(1),
                          ConstantInt::get(Type::getInt32Ty(C), 0));
  BBEntry->getInstList().push_back(Cmp);
  BBEntry->getInstList().push_back(BranchInst::Create(BBTrue, BBFalse, Cmp));

  Value *Two = ConstantInt::get(Type::getInt32Ty(C), 2);
  Value *Three = ConstantInt::get(Type::getInt32Ty(C), 3);
  Instruction *Add = BinaryOperator::Create(Instruction::Add, Two, Three);
  BBTrue->getInstList().push_back(Add);
  Instruction *Mul1 =
      BinaryOperator::Create(Instruction::Mul, Add, F->getArg(0));
  BBTrue->getInstList().push_back(Mul1);

  // Create the return instruction and add it to the basic block
  BBTrue->getInstList().push_back(ReturnInst::Create(C, Mul1));

  BBFalse->getInstList().push_back(ReturnInst::Create(C, F->getArg(0)));
  return F;
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();

  auto Context = std::make_unique<LLVMContext>();
  auto TopModule = std::make_unique<Module>("top", *Context);

  createFred(*TopModule);
  createFact(*TopModule);

  llvm::verifyModule(*TopModule, &llvm::outs());
  WriteBitcodeToFile(*TopModule, llvm::outs());
  TopModule->dump();

  auto TSM = orc::ThreadSafeModule(std::move(TopModule), std::move(Context));
  auto JIT = getExitOnErr()(orc::LLJITBuilder().create());
  getExitOnErr()(JIT->addIRModule(std::move(TSM)));

  auto fredSym = getExitOnErr()(JIT->lookup("fred"));
  int (*fred)(int32_t, int32_t) =
      (int32_t(*)(int32_t, int32_t))fredSym.getAddress();

  outs() << "fred(42, 0) = " << fred(42, 0) << "\n";
  outs() << "fred(42, 1) = " << fred(42, 1) << "\n";

  auto factSym = getExitOnErr()(JIT->lookup("fact"));
  int32_t (*fact)(int32_t) = (int32_t(*)(int32_t))factSym.getAddress();
  for (int32_t i = 0; i < 10; ++i) {
    outs() << "fact(" << i << ") = " << fact(i) << "\n";
  }
}
