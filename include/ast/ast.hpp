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

class AstNode;
using TypeInfo = mlga::core::DiscreteTypeInfo;

class AstOp {
public:
  AstOp(const AstOp &Op);
  AstOp(AstNode *AstNode) : AstNode(AstNode) {}
  AstNode *getNode() const { return AstNode; }

protected:
  AstNode *AstNode{nullptr};
};

/// \brief A handle to a particular result of an AstNode.
class AstResult {
public:
  AstResult(AstNode *Node, size_t Index) : AstNode(Node), Index(Index) {}
  AstNode *getNode() const { return AstNode; }
  size_t getIndex() const { return Index; }

protected:
  AstNode *AstNode;
  size_t Index;
};

class AstNode : public mlga::core::AstContextManaged {
public:
  virtual ~AstNode() {}
  AstNode(mlga::core::AstContext &Context,
          const std::vector<AstResult> &Operands);
  virtual const TypeInfo &getTypeInfo() = 0;
  void setResultsSize(size_t size);
  const std::vector<AstResult> &getOperands() const { return Operands; };
  const std::vector<AstResult> &getResults() const { return Results; }
  operator AstOp() { return AstOp(this); }

protected:
  std::vector<AstResult> Operands;
  std::vector<AstResult> Results;
};

struct DefaultAttributes {};

template <typename AstOpType, typename AttributesType = DefaultAttributes>
class AstOpImp : public AstOp {
public:
  class AstNodeImp : public AstNode {

  public:
    using AstNode::AstNode;
    const TypeInfo &getTypeInfo() { return AstOpType::TypeInfo; }
    static const TypeInfo &getStaticTypeInfo() { return AstOpType::TypeInfo; }
    AttributesType Attributes;
  };

  AstOpImp() {}

  AstOpImp(const AstOp &Op) : AstOpImp(Op.getNode()) {}

  AstOpImp(class AstNode *Node) : AstOp(mlga::core::asType<AstNodeImp>(Node)) {}

  AstOpImp(const AstOpType &Op) : AstOp(Op->AstNode) {}

  AstOpImp(mlga::core::AstContext &Context,
           const std::vector<AstResult> &Operands)
      : AstOp(create(Context, Operands)) {}

  AttributesType *getAttributes() const {
    return AstNode ? &static_cast<AstNodeImp *>(AstNode)->Attributes : nullptr;
  }

  operator AstOp() const { return AstOp(AstNode); }

  static AstNodeImp *create(mlga::core::AstContext &Context,
                            const std::vector<AstResult> &Operands) {
    AstNodeImp *Value = new AstNodeImp(Context, Operands);
    AstOpType::initialize(Value);
    return Value;
  }
  const std::vector<AstResult> &getResults() { return AstNode->getResults(); }
  static void initialize(AstNodeImp *node) { node->setResultsSize(1); }
};

struct ParameterAttributes {
  std::string Name;
};

class ParameterOp : public AstOpImp<ParameterOp, ParameterAttributes> {
public:
  using AstOpImp<ParameterOp, ParameterAttributes>::AstOpImp;
  ParameterOp(mlga::core::AstContext &Context, const std::string &Name)
      : ParameterOp(Context, std::vector<AstResult>{}) {
    getAttributes()->Name = Name;
  }
  static constexpr TypeInfo TypeInfo{"Parameter", 0};
};

class AddOp : public AstOpImp<AddOp> {
public:
  static constexpr TypeInfo TypeInfo{"Add", 0};
};

class MultiplyOp : public AstOpImp<MultiplyOp> {
public:
  static constexpr TypeInfo TypeInfo{"Multiply"};
};

inline AstResult operator+(const AstResult &x, const AstResult &y) {
  return AddOp::create(x.getNode()->getContext(), {x, y})->getResults().at(0);
}

inline AstResult operator*(const AstResult &x, const AstResult &y) {
  return MultiplyOp::create(x.getNode()->getContext(), {x, y})
      ->getResults()
      .at(0);
}
}
}
#endif
