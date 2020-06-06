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

mlga::ast::Node::Node(const Ops &Operands) : Operands(Operands) {
  for (auto Operand : Operands) {
    auto C = Operand.getContext();
    if (C) {
      setContext(C);
      return;
    }
  }
}

void mlga::ast::Node::setContext(mlga::core::Context *Context) {
  std::vector<Node *> ToDo{this};
  while (!ToDo.empty()) {
    auto Node = ToDo.back();
    ToDo.pop_back();
    for (auto &Op : Node->getOperands()) {
      if (!Op.getNode()->getContext()) {
        ToDo.push_back(Op.getNode());
      }
    }
    Node->mlga::core::ContextManaged::setContext(Context);
  }
}
