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

#ifndef MLGA_AST_AST_HPP
#define MLGA_AST_AST_HPP

#include "ast/context.hpp"
#include "ast/exports.hpp"
#include "ast/type_info.hpp"

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mlga {
namespace ast {

class Node;
class Op;
using Ops = std::vector<Op>;
using TypeInfo = mlga::core::DiscreteTypeInfo;

class Node : public mlga::core::ContextManaged {
public:
  virtual ~Node() {}
  Node(const Ops &Operands);
  virtual const TypeInfo &getTypeInfo() = 0;
  void setContext(mlga::core::Context *Context) override;
  const Ops &getOperands() const { return Operands; };

protected:
  Ops Operands;
};

class Op {
public:
  Op(Node *Node) : Node(Node) {}
  Node *getNode() const { return Node; }
  mlga::core::Context *getContext() { return Node->getContext(); }
  void setContext(mlga::core::Context *Context) { Node->setContext(Context); }
  const Ops &getOperands() { return Node->getOperands(); }

protected:
  Node *Node{nullptr};
};

using Ops = std::vector<Op>;

template <typename T> Ops asOps(T Values) {
  Ops Result;
  for (auto &Value : Values) {
    Result.push_back(Value);
  }
  return Result;
}

struct DefaultAttributes {};

template <typename OpType, typename AttributesType = DefaultAttributes>
class OpImp : public Op {
public:
  class NodeImp : public Node {

  public:
    using Node::Node;
    const TypeInfo &getTypeInfo() { return OpType::TypeInfo; }
    static const TypeInfo &getStaticTypeInfo() { return OpType::TypeInfo; }
    AttributesType Attributes;
  };

  OpImp() {}

  OpImp(const Op &Op) : OpImp(Op.getNode()) {}

  OpImp(const Ops &Operands) : Op(new NodeImp(Operands)) {}

  AttributesType &getAttributes() const {
    return static_cast<NodeImp *>(Node)->Attributes;
  }
};
}
}
#endif
