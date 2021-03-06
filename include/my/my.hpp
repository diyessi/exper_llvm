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

#ifndef MLGA_MY_MY_HPP_
#define MLGA_MY_MY_HPP_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace my {

class MyDialect : public mlir::Dialect {
public:
  explicit MyDialect(mlir::MLIRContext *ctx);
  static llvm::StringRef getDialectNamespace() { return "my"; }
};

#define GET_OP_CLASSES
#include "my/Ops.h.inc"
}
}

#endif
