#ifndef TOY_OPS
#define TOY_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def My_Dialect : Dialect {
  let name = "my";
  let cppNamespace = "my";
}

class My_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<My_Dialect, mnemonic, traits>;

def OneOp : My_Op<"one", [NoSideEffect]> {
    let summary = "one operation";
    let description = [{
        One operation returns a literal with value 1.
    }];
    let arguments = (ins);
    let results = (outs AnyTensor);
}

#endif
