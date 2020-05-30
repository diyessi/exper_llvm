#ifndef MY_DIALECT_HPP_
#define MY_DIALECT_HPP_

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