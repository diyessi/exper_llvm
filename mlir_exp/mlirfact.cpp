//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "ast/ast.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "my/my.hpp"

namespace cl = llvm::cl;

int main(int argc, char **argv) {

  cl::ParseCommandLineOptions(argc, argv, "MLIR experiments\n");
  mlir::registerDialect<mlir::my::MyDialect>();
  mlga::core::AstContext AstContext;
  mlga::ast::ParameterOp ParamX(AstContext, "X");
  mlga::ast::AstOp ZZ = ParamX;
  mlga::ast::ParameterOp ParamX2(ZZ);
  auto x = ParamX.getResults().at(0);
  auto y = x + x;

  mlir::MLIRContext context;
  mlir::OpBuilder builder(&context);
  mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());
  auto func_type = builder.getFunctionType({}, builder.getF64Type());
  mlir::FuncOp function =
      mlir::FuncOp::create(builder.getUnknownLoc(), "name", func_type);
  auto &entryBlock = *function.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  auto val = builder.create<mlir::my::OneOp>(builder.getUnknownLoc(),
                                             builder.getF64Type());

  module.push_back(function);

  module.dump();

  return 0;
}
