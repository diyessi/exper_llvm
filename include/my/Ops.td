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

#ifndef MY_OPS
#define MY_OPS

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
