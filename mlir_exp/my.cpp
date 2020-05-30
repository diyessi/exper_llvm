#include "my/my.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace mlir::my;

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
MyDialect::MyDialect(mlir::MLIRContext *ctx) : mlir::Dialect("my", ctx) {
  addOperations<
#define GET_OP_LIST
#include "my/Ops.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "my/Ops.cpp.inc"
